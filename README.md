# 🚢 Hệ Thống Nhận Diện Chướng Ngại Vật & Tàu Thuyền Trên Mặt Biển

> Ứng dụng thị giác máy tính kết hợp **Xử lý ảnh số (DIP)** và **Mạng nơ-ron sâu YOLOv8** để phát hiện chướng ngại vật và tàu thuyền trong môi trường hàng hải — hỗ trợ cả ảnh tĩnh và luồng camera thời gian thực.

---

## 👥 Thành Viên Nhóm

| Họ và Tên | Vai trò |
|---|---|
| **Phạm Minh Quang** | Thành viên |
| **Phan Sĩ Thành** | Thành viên |
| **Nguyễn Phúc Tâm Huy** | Nhóm trưởng|

---

## 📌 Giới Thiệu Dự Án

Dự án xây dựng một web application tích hợp pipeline xử lý ảnh 4 bước:

1. **Nhập ảnh gốc** — đọc ảnh thô từ camera hoặc tệp tải lên.
2. **Bộ lọc cải thiện (DIP Filter)** — áp dụng CLAHE + Unsharp Masking để khử sương mù, tăng độ nét.
3. **Tiền xử lý hình học (Letterbox)** — chuẩn hóa kích thước ảnh về 640×640 px giữ nguyên tỷ lệ.
4. **Suy luận & Hậu xử lý** — YOLOv8 phát hiện đối tượng, lọc NMS và trả về bounding box.

### 📊 Chỉ Số Mô Hình (best checkpoint)

| Chỉ số | Giá trị |
|---|---|
| mAP@0.5 | **0.884** |
| mAP@0.5:0.95 | **0.612** |
| Precision | **0.895** |
| Recall | **0.841** |
| Số epoch huấn luyện | 30 (tốt nhất tại epoch 28) |

---

## 📁 Cấu Trúc Thư Mục

```
obstacle-detection-on-traffic/
│
├── app.py                          # Flask server chính — routes, pipeline xử lý ảnh, webcam stream
├── train.py                        # Script huấn luyện YOLOv8 (chạy trên Google Colab/GPU)
├── README.md                       # Tài liệu dự án
│
├── sea_detection_sea1/             # Kết quả & artifacts sau quá trình huấn luyện
│   ├── weights/
│   │   ├── best.pt                 # ✅ Model checkpoint tốt nhất (dùng để inference)
│   │   └── last.pt                 # Model checkpoint cuối cùng
│   ├── args.yaml                   # Cấu hình siêu tham số huấn luyện
│   ├── results.csv                 # Lịch sử loss & metrics theo từng epoch
│   ├── results.png                 # Biểu đồ training curves
│   ├── confusion_matrix.png        # Ma trận nhầm lẫn
│   ├── confusion_matrix_normalized.png
│   ├── BoxF1_curve.png             # F1 theo ngưỡng confidence
│   ├── BoxPR_curve.png             # Precision-Recall curve
│   ├── BoxP_curve.png              # Precision curve
│   ├── BoxR_curve.png              # Recall curve
│   ├── labels.jpg                  # Phân phối nhãn trong dataset
│   ├── train_batch*.jpg            # Mẫu batch huấn luyện
│   └── val_batch*_labels/pred.jpg  # Kết quả validation
│
├── static/
│   ├── style/
│   │   └── style.css               # CSS giao diện web
│   └── uploads/                    # Thư mục lưu ảnh tải lên & kết quả xử lý
│       ├── raw_<filename>          # Ảnh gốc
│       ├── enh_<filename>          # Ảnh sau CLAHE + Unsharp Mask
│       ├── padded_<filename>       # Ảnh sau Letterbox
│       └── final_<filename>        # Ảnh kết quả với bounding box
│
└── templates/
    └── index.html                  # Giao diện web (Jinja2 template)
```

---

## ⚙️ Yêu Cầu Hệ Thống

### Phần Mềm

| Yêu cầu | Phiên bản khuyến nghị |
|---|---|
| Python | ≥ 3.9 |
| pip | ≥ 23.0 |

### Thư Viện Python

Cài đặt toàn bộ dependencies bằng lệnh:

```bash
pip install flask ultralytics opencv-python numpy pandas
```

| Thư viện | Mô tả |
|---|---|
| `flask` | Web framework để chạy server |
| `ultralytics` | YOLOv8 — phát hiện đối tượng |
| `opencv-python` | Xử lý ảnh (CLAHE, Unsharp Mask, Letterbox, vẽ bounding box) |
| `numpy` | Xử lý mảng số |
| `pandas` | Đọc file `results.csv` để hiển thị metrics |

### Phần Cứng

- **Camera** (tùy chọn): Cần có webcam nếu muốn dùng tính năng live stream.
- **GPU** (tùy chọn): CUDA-compatible GPU sẽ tăng tốc đáng kể tốc độ inference; CPU vẫn hoạt động bình thường.

---

## 🚀 Hướng Dẫn Chạy Ứng Dụng

### Bước 1 — Clone repository

```bash
git clone <repository-url>
cd obstacle-detection-on-traffic
```

### Bước 2 — Cài đặt thư viện

```bash
pip install flask ultralytics opencv-python numpy pandas
```

### Bước 3 — Chạy server

```bash
python app.py
```

### Bước 4 — Mở trình duyệt

Truy cập địa chỉ:

```
http://127.0.0.1:5000
```

> **Lưu ý:** File model `sea_detection_sea1/weights/best.pt` phải tồn tại trước khi chạy `app.py`. File này là kết quả sau quá trình huấn luyện và phải được đặt đúng đường dẫn.

---

## 🧠 Huấn Luyện Lại Mô Hình (Tùy chọn)

Script `train.py` được thiết kế để chạy trên **Google Colab** với GPU:

```bash
# Chỉnh sửa đường dẫn dataset trong train.py trước khi chạy
python train.py
```

**Cấu hình mặc định:**
- Model gốc: `yolov8n.pt` (YOLOv8 Nano)
- Epochs: 30
- Image size: 640×640
- Batch size: 16
- Device: GPU (device=0)

---

## 🌐 Tính Năng Chính

| Tính năng | Mô tả |
|---|---|
| 📷 **Upload ảnh tĩnh** | Tải ảnh lên, xem kết quả pipeline 4 bước và bảng thống kê chi tiết |
| 🎥 **Live camera stream** | Bật/tắt webcam để nhận diện đối tượng theo thời gian thực |
| 🎛️ **Điều chỉnh ngưỡng** | Thanh trượt Confidence (0.1–0.9) và IoU/NMS (0.1–0.9) |
| ⚡ **Đo độ trễ** | Hiển thị thời gian từng giai đoạn: tiền xử lý, suy luận, hậu xử lý |
| 📊 **Hiển thị metrics** | mAP, Precision, Recall của mô hình ngay trên giao diện |

---

## 🔧 API Endpoints

| Method | Endpoint | Mô tả |
|---|---|---|
| `GET/POST` | `/` | Trang chính — upload ảnh và xem kết quả |
| `GET` | `/video_feed` | MJPEG stream từ webcam |
| `POST` | `/toggle_camera` | Bật/tắt camera (`{"action": "start"\|"stop"}`) |
| `POST` | `/update_thresholds` | Cập nhật ngưỡng Conf & IoU cho camera live |
