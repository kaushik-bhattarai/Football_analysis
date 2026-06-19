from ultralytics import YOLO

model = YOLO("/home/kaushik/Football_Detection/runs/detect/train/weights/best.pt")

val = model.val(data="/home/kaushik/Football_Detection/Datasets/football-players-detection.v20-rf-detr-m.yolo26/data.yaml", split="val",batch=4,device=0)
print(f"Val  mAP50: {val.box.map50:.4f}")
