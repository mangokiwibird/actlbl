import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.predict("./data.jpg", save=True, conf=0.9)
results[0].boxes.data.tolist()
plots = results[0].plot()
cv2.imshow("plot", plots)
cv2.waitKey(0)
cv2.destroyAllWindows()
