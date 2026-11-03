from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model.train(
    data='/content/dataset/data.yaml',
    epochs=30,
    imgsz=640,
    batch=16,
    device=0,
    project='/content/runs/detect',
    name='ship_detection_sea1'
)