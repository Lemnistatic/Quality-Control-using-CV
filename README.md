# Quality-Control-using-CV
Final Year Project- Presented at ICISD 

Frameworks Used: YOLO, OpenCV

This can be taken as a POC for other fields and is very scalable.

Trained the YOLO model with a certain leather bag ( this reference image is taken for comparison with the other bags).
Now, the other mass produced leather bags are compared with the reference model. Using YOLO and canny edge detection, any faults, blemishes, scracthes, stiching issues and discolourations are noted and picked up by the model. This is also displayed as heatmaps to give a further understanding of where the discrepancies are found.
The model then gives confidence rating for each of the parameters and then decides whether the object has failed or passed based on the thresholds we provide. It can be strict or lenient.




/h2 Scalability

It can be used in the medical field - say, if we train the model with healthy cells, it can spot irregularity in cells and fish out any underlying causes or diseases way before it worsens. The only true limit is the budget. This project's POC has near 
infinite scalability.
