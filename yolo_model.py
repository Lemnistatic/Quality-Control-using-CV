#18-08-2025damn i just realised I didnt upload the backend file, my team has it let me get back to them and update the file. Sorry Joseph sir 😭
#19-08-2025 I have updated it now, sorry for the delay!
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from ultralytics import YOLO
import torch
import os
from PIL import Image

app = FastAPI()

# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize YOLOv8 model
model = YOLO('yolov8s.pt')

# Load custom defect detection model if available
defect_model = None
if os.path.exists('leather_defect_model.pt'):
    defect_model = YOLO('leather_defect_model.pt')

@app.post("/analyze")
async def analyze_bag(image: UploadFile = File(...)):
    bag_type = "black"  # Force black bag analysis

    contents = await image.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original = img_rgb.copy()
    img_resized = cv2.resize(img_rgb, (640, 640))

    temp_img_path = "temp_image.jpg"
    cv2.imwrite(temp_img_path, cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))

    results = model(temp_img_path)

    bag_detected = False
    bag_confidence = 0
    bag_class = None

    for r in results:
        bag_classes = ['handbag', 'backpack', 'suitcase', 'purse']
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = r.names[cls]
            if class_name in bag_classes and conf > 0.5:
                bag_detected = True
                bag_confidence = conf
                bag_class = class_name
                break

    hsv = cv2.cvtColor(img_resized, cv2.COLOR_RGB2HSV)

    lower_color = np.array([0, 0, 0])
    upper_color = np.array([180, 255, 50])

    mask = cv2.inRange(hsv, lower_color, upper_color)
    leather_percentage = np.sum(mask > 0) / (640 * 640) * 100

    defect_percentage = 0
    defect_results = None

    if defect_model:
        defect_results = defect_model(temp_img_path)
        defects_area = 0
        for r in defect_results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                defects_area += (x2 - x1) * (y2 - y1)
        defect_percentage = (defects_area / (640 * 640)) * 100
    else:
        gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        defect_percentage = np.sum(edges > 0) / (640 * 640) * 100

    is_leather = leather_percentage > 15
    has_defects = defect_percentage > 2

    if not bag_detected:
        is_leather = False

    result = "pass" if is_leather and not has_defects else "fail"
    issues = []

    if not bag_detected:
        issues.append("No bag detected in image")
    elif not is_leather:
        issues.append("Material may not be genuine leather")
    if has_defects:
        issues.append("Detected surface imperfections or scratches")

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs[0, 0].imshow(original)
    axs[0, 0].set_title('Original Image')

    if os.path.exists(temp_img_path):
        yolo_result_img = Image.fromarray(results[0].plot())
        axs[0, 1].imshow(yolo_result_img)
        axs[0, 1].set_title(f'YOLO Detection: {bag_class if bag_detected else "No bag"}')
    else:
        axs[0, 1].imshow(original)
        axs[0, 1].set_title('YOLO Detection: Failed')

    axs[0, 2].imshow(mask, cmap='gray')
    axs[0, 2].set_title(f'Leather Material Detection: {leather_percentage:.1f}%')

    if defect_model and defect_results:
        defect_img = Image.fromarray(defect_results[0].plot())
        axs[1, 0].imshow(defect_img)
        axs[1, 0].set_title(f'Defect Detection: {defect_percentage:.1f}%')
    else:
        gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        axs[1, 0].imshow(edges, cmap='gray')
        axs[1, 0].set_title(f'Edge Detection: {defect_percentage:.1f}%')

    axs[1, 1].axis('off')
    confidence_display = f"{bag_confidence:.2f}" if bag_detected else "0.00"
    axs[1, 1].text(0, 0.5,
        f"Result: {result.upper()}\n"
        f"Bag detected: {'Yes' if bag_detected else 'No'}\n"
        f"Bag type: {bag_class if bag_detected else 'Unknown'}\n"
        f"Confidence: {confidence_display}\n"
        f"Leather area: {leather_percentage:.1f}%\n"
        f"Defect density: {defect_percentage:.1f}%\n"
        f"Issues: {'None' if not issues else ', '.join(issues)}",
        fontsize=12
    )

    axs[1, 2].axis('off')
    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode()
    plt.close()

    if os.path.exists(temp_img_path):
        os.remove(temp_img_path)

    return JSONResponse(content={
        "result": result,
        "bag_detected": bag_detected,
        "bag_type": bag_class if bag_detected else "Unknown",
        "confidence": float(bag_confidence) if bag_detected else 0.0,
        "leather_percentage": float(leather_percentage),
        "defect_percentage": float(defect_percentage),
        "issues": issues,
        "annotated_image": f"data:image/png;base64,{img_base64}"
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

