import sys
import torch

sys.path.insert(0, "C:/YOLOv8-multi-task/ultralytics")

from ultralytics import YOLO

number = 3 #input how many tasks in your work
model = YOLO('C:/YOLOv8-multi-task/runs/multi/yolopm/weights/best.pt')  # Validate the model
#image size only -> width : 1280, height : 720
model.predict(source='./drive', imgsz=(384,672), device=0,name='drive_predict', save=True, conf=0.25, iou=0.45, show_labels=False, speed=True)
