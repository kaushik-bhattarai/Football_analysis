from ultralytics import YOLO

model = YOLO("/home/kaushik/Football_Detection/runs/detect/train-4/weights/best.pt")

val = model.val(data="/home/kaushik/Football_Detection/Datasets/football-players-detection.v20-rf-detr-m.yolo26/data.yaml", split="val",batch=4,device=0)
print(f"Val  mAP50: {val.box.map50:.4f}")

test = model.val(data="/home/kaushik/Football_Detection/Datasets/football-players-detection.v20-rf-detr-m.yolo26/data.yaml", split="test",batch=4,device=0)
print(f"Test mAP50: {test.box.map50:.4f}")