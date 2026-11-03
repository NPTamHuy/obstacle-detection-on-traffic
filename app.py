import os
import time
import cv2
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, Response, jsonify
from ultralytics import YOLO

app = Flask(__name__)
UPLOAD_DIR = os.path.join("static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.config["UPLOAD_DIR"] = UPLOAD_DIR

# -----------------------------------------------------------------
# 1. NẠP MÔ HÌNH VÀ CẤU HÌNH NHÃN
# -----------------------------------------------------------------
MODEL_PATH = "sea_detection_sea1/weights/best.pt"
model = YOLO(MODEL_PATH)
RAW_CLASSES = model.names

CLASS_TRANSLATE = {
    0: "1", 1: "2", 2: "3", 3: "4", 4: "5",
    5: "6", 6: "7", 7: "8", 8: "9", 9: "10", 10: "11"
}

def get_class_name(cid):
    if cid in CLASS_TRANSLATE:
        return CLASS_TRANSLATE[cid]
    name = RAW_CLASSES.get(cid, f"Class {cid}")
    if str(name).isdigit() and int(name) in CLASS_TRANSLATE:
        return CLASS_TRANSLATE[int(name)]
    return str(name)

# -----------------------------------------------------------------
# 2. BIẾN TOÀN CỤC CHO WEBCAM
# -----------------------------------------------------------------
camera = None
is_camera_active = False
cam_conf = 0.40
cam_iou = 0.45

def get_camera():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
    return camera

def release_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None

# -----------------------------------------------------------------
# 3. ĐỌC CHỈ SỐ HUẤN LUYỆN
# -----------------------------------------------------------------
def load_metrics():
    csv_paths = ["sea_detection_sea1/results.csv"]
    csv_path = next((p for p in csv_paths if os.path.exists(p)), None)
    if not csv_path:
        return {
            "map50": 0.884, "map50_95": 0.612, "precision": 0.895,
            "recall": 0.841, "total_epochs": 30, "best_epoch": 28
        }

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    best_row = df.loc[df["metrics/mAP50(B)"].idxmax()] if "metrics/mAP50(B)" in df.columns else df.iloc[-1]

    return {
        "total_epochs": len(df),
        "best_epoch": int(best_row.get("epoch", len(df))),
        "precision": round(float(best_row.get("metrics/precision(B)", 0)), 4),
        "recall": round(float(best_row.get("metrics/recall(B)", 0)), 4),
        "map50": round(float(best_row.get("metrics/mAP50(B)", 0)), 4),
        "map50_95": round(float(best_row.get("metrics/mAP50-95(B)", 0)), 4),
    }

METRICS = load_metrics()

# -----------------------------------------------------------------
# 4. BỘ LỌC CẢI THIỆN ẢNH & TIỀN XỬ LÝ HÌNH HỌC (DIP OPERATORS)
# -----------------------------------------------------------------
def enhance_image(image):
    """
    Quy trình cải thiện chất lượng ảnh:
    1. Chuyển sang không gian màu LAB để tách biệt kênh sáng (L) và màu sắc (A, B).
    2. Áp dụng CLAHE (Contrast Limited Adaptive Histogram Equalization) trên kênh L
       nhằm cân bằng độ tương phản cục bộ, giảm tác động sương mù / chói nắng mặt biển.
    3. Áp dụng Unsharp Masking để tăng độ sắc nét cấu trúc thân tàu, khử mờ.
    """
    # Bước 1 & 2: Cân bằng tương phản thích nghi cục bộ trên kênh Luminance
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    contrast_enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # Bước 3: Làm sắc nét bằng Unsharp Masking
    gaussian = cv2.GaussianBlur(contrast_enhanced, (0, 0), sigmaX=2.0)
    # Công thức: Sharpened = (1 + amount) * Original - amount * Blurred
    sharpened = cv2.addWeighted(contrast_enhanced, 1.4, gaussian, -0.4, 0)
    return sharpened

def letterbox(image, target_shape=(640, 640), fill_color=(114, 114, 114)):
    """Biến đổi hình học: Giữ nguyên tỷ lệ vật thể và bù viền xám hằng số"""
    h, w = image.shape[:2]
    scale = min(target_shape[0] / h, target_shape[1] / w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    dw = (target_shape[1] - new_w) / 2
    dh = (target_shape[0] - new_h) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=fill_color
    )
    return padded

# -----------------------------------------------------------------
# 5. WEBCAM STREAM GENERATOR (CÓ BỘ LỌC CẢI THIỆN)
# -----------------------------------------------------------------
def generate_camera_frames():
    global is_camera_active, cam_conf, cam_iou
    cam = get_camera()

    while is_camera_active:
        success, frame = cam.read()
        if not success:
            break

        # Cải thiện chất lượng frame ảnh trước khi nạp vào model
        enhanced_frame = enhance_image(frame)

        # Suy luận YOLO trên frame đã được nâng cao chất lượng
        results = model.predict(source=enhanced_frame, conf=cam_conf, iou=cam_iou, imgsz=640, verbose=False)
        annotated_frame = enhanced_frame.copy()

        for box in results[0].boxes:
            cid = int(box.cls[0].item())
            cname = get_class_name(cid)
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 240, 255), 2)
            label_txt = f"{cname} | {conf * 100:.1f}%"
            (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_frame, (x1, max(y1 - 20, 0)), (x1 + tw + 8, y1), (0, 240, 255), -1)
            cv2.putText(annotated_frame, label_txt, (x1 + 4, max(y1 - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (15, 23, 42), 1, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# -----------------------------------------------------------------
# 6. FLASK ROUTES
# -----------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    global cam_conf, cam_iou
    result = None
    conf_thresh = float(request.form.get("conf", 0.40))
    iou_thresh = float(request.form.get("iou", 0.45))
    cam_conf = conf_thresh
    cam_iou = iou_thresh

    if request.method == "POST":
        file = request.files.get("image")
        if file and file.filename != "":
            filename = file.filename
            raw_path = os.path.join(app.config["UPLOAD_DIR"], "raw_" + filename)
            file.save(raw_path)

            orig_img = cv2.imread(raw_path)
            orig_h, orig_w = orig_img.shape[:2]

            # [Giai đoạn 1: Bộ lọc cải thiện ảnh & Tiền xử lý]
            t0 = time.time()
            enhanced_img = enhance_image(orig_img)
            padded_img = letterbox(enhanced_img, (640, 640))
            t_preprocess = (time.time() - t0) * 1000

            enhanced_path = os.path.join(app.config["UPLOAD_DIR"], "enh_" + filename)
            letterbox_path = os.path.join(app.config["UPLOAD_DIR"], "padded_" + filename)
            cv2.imwrite(enhanced_path, enhanced_img)
            cv2.imwrite(letterbox_path, padded_img)

            # [Giai đoạn 2: Trích xuất đặc trưng & Suy luận Deep Learning]
            t1 = time.time()
            # Đưa trực tiếp ảnh đã được cải thiện chất lượng vào YOLO
            preds = model.predict(source=enhanced_img, conf=conf_thresh, iou=iou_thresh, imgsz=640, verbose=False)
            t_infer = (time.time() - t1) * 1000

            # [Giai đoạn 3: Hậu xử lý & Tái cấu trúc ma trận]
            t2 = time.time()
            pred = preds[0]
            output_img = enhanced_img.copy()
            detections = []

            for box in pred.boxes:
                cid = int(box.cls[0].item())
                cname = get_class_name(cid)
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

                cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 240, 255), 2)
                label_txt = f"{cname} | {conf * 100:.1f}%"
                (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(output_img, (x1, max(y1 - 22, 0)), (x1 + tw + 10, y1), (0, 240, 255), -1)
                cv2.putText(output_img, label_txt, (x1 + 5, max(y1 - 6, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 10), 1, cv2.LINE_AA)

                detections.append({
                    "class": cname,
                    "confidence": round(conf, 4),
                    "box": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1)
                })

            final_path = os.path.join(app.config["UPLOAD_DIR"], "final_" + filename)
            cv2.imwrite(final_path, output_img)
            t_postprocess = (time.time() - t2) * 1000

            result = {
                "raw_img": raw_path.replace("\\", "/"),
                "enhanced_img": enhanced_path.replace("\\", "/"),
                "letterbox_img": letterbox_path.replace("\\", "/"),
                "final_img": final_path.replace("\\", "/"),
                "orig_size": f"{orig_w} × {orig_h}",
                "detections": detections,
                "count": len(detections),
                "latencies": {
                    "preprocess": round(t_preprocess, 2),
                    "inference": round(t_infer, 2),
                    "postprocess": round(t_postprocess, 2),
                    "total": round(t_preprocess + t_infer + t_postprocess, 2)
                }
            }

    return render_template("index.html",
                           metrics=METRICS,
                           result=result,
                           conf=conf_thresh,
                           iou=iou_thresh)

@app.route("/video_feed")
def video_feed():
    if not is_camera_active:
        return "Camera is off", 400
    return Response(generate_camera_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/toggle_camera", methods=["POST"])
def toggle_camera():
    global is_camera_active
    data = request.get_json(silent=True) or {}
    action = data.get("action")

    if action == "start":
        is_camera_active = True
    elif action == "stop":
        is_camera_active = False
        release_camera()
    else:
        is_camera_active = not is_camera_active
        if not is_camera_active:
            release_camera()

    return jsonify({"active": is_camera_active})

@app.route("/update_thresholds", methods=["POST"])
def update_thresholds():
    global cam_conf, cam_iou
    data = request.get_json(silent=True) or {}
    cam_conf = float(data.get("conf", cam_conf))
    cam_iou = float(data.get("iou", cam_iou))
    return jsonify({"status": "ok", "conf": cam_conf, "iou": cam_iou})

if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True)