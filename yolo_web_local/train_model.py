import os
from google.colab import drive
from ultralytics import YOLO

# 1. Buộc ngắt kết nối cũ và xóa sạch thư mục mount nếu có tệp kẹt bên trong
if os.path.exists('/content/drive'):
    try:
        drive.flush_and_unmount()
    except:
        pass
    # Xóa sạch thư mục /content/drive để tránh lỗi chứa file cũ
    !rm -rf /content/drive

# 2. Tiến hành kết nối lại Google Drive một cách sạch sẽ
drive.mount('/content/drive')

# 3. Tải mô hình YOLOv8 pre-trained
model = YOLO('yolov8n.pt')

# 4. Huấn luyện mô hình
results = model.train(
    data='/content/drive/MyDrive/YOLO_Dataset/ROD-Dataset/dataset/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,
    project='/content/drive/MyDrive/YOLO_Results',
    name='obstacle_run_1'
)

