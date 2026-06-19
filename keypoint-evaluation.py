from ultralytics import YOLO

model = YOLO("/home/kaushik/Football_Detection/runs/pose/train-2/weights/best.pt")

val = model.val(
    data="/home/kaushik/Football_Detection/Datasets/football-field-detection.v16i.yolov8/data.yaml",
    split="val",
    imgsz=640,
    batch=4,
)
print(f"Val mAP50:    {val.box.map50:.4f}")
print(f"Val mAP50-95: {val.box.map:.4f}")
