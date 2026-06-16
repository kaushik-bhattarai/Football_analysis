from ultralytics import YOLO

model = YOLO("yolo26l.pt")

model.train(data="/home/kaushik/Football_Detection/Datasets/football-players-detection.v20-rf-detr-m.yolo26/data.yaml",
            epochs=50,
            imgsz=1280,
            batch=4,
            device=0
            )