from ultralytics import YOLO

# Load model
model = YOLO("Models/yolo26l-pose.pt")

# Train
model.train(
    data="/home/kaushik/Football_Detection/Datasets/football-field-detection.v16i.yolov8/data.yaml",
    epochs=100,
    batch=8,
    imgsz=640,
    mosaic=0.0,
    plots=True,
    )