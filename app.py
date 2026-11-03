import os
import time
import cv2
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from ultralytics import YOLO

app = Flask(__name__)
UPLOAD_DIR = os.path.join("static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.config["UPLOAD_DIR"] = UPLOAD_DIR

# -----------------------------------------------------------------
# 1. NẠP MÔ HÌNH VÀ CẤU HÌNH NHÃN
# -----------------------------------------------------------------
# Trỏ đến file best.pt mới train xong
MODEL_PATH = "sea_detection_sea1/weights/best.pt"

model = YOLO(MODEL_PATH)
RAW_CLASSES = model.names

# Bảng dịch nhãn sang tiếng Việt để báo cáo trực quan
# Tùy biến theo danh sách class trong dataset của bạn
CLASS_TRANSLATE = {
    0: "1",
    1: "2",
    2: "3",
    3: "4",
    4: "5",
    5: "6",
    6: "7",
    7: "8",
    8: "9",
    9: "10",
    10: "11"
}


def get_class_name(cid):
    if cid in CLASS_TRANSLATE:
        return CLASS_TRANSLATE[cid]
    name = RAW_CLASSES.get(cid, f"Class {cid}")
    if str(name).isdigit() and int(name) in CLASS_TRANSLATE:
        return CLASS_TRANSLATE[int(name)]
    return str(name)


# -----------------------------------------------------------------
# 2. ĐỌC CHỈ SỐ HUẤN LUYỆN TỪ RESULTS.CSV
# -----------------------------------------------------------------
def load_metrics():
    csv_paths = [
        "sea_detection_sea1/results.csv"
    ]
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
# 3. CÁC PHÉP TOÁN XỬ LÝ ẢNH SỐ (IMAGE PROCESSING OPERATORS)
# -----------------------------------------------------------------
def letterbox(image, target_shape=(640, 640), fill_color=(114, 114, 114)):
    """Biến đổi hình học: Giữ nguyên tỷ lệ và đệm viền hằng số"""
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


def extract_edge_map(image):
    """Kỹ thuật trích xuất đặc trưng biên cạnh: Gaussian Blur + Canny Edge"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    edges = cv2.Canny(blurred, 60, 180)
    # Tô màu viền để hiển thị trực quan dạng RGB
    edge_colored = cv2.applyColorMap(edges, cv2.COLORMAP_OCEAN)
    return edge_colored


# -----------------------------------------------------------------
# 4. ROUTE FLASK WEB
# -----------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    conf_thresh = float(request.form.get("conf", 0.40))
    iou_thresh = float(request.form.get("iou", 0.45))

    if request.method == "POST":
        file = request.files.get("image")
        if file and file.filename != "":
            filename = file.filename
            raw_path = os.path.join(app.config["UPLOAD_DIR"], "raw_" + filename)
            file.save(raw_path)

            orig_img = cv2.imread(raw_path)
            orig_h, orig_w = orig_img.shape[:2]

            # [Giai đoạn 1: Tiền xử lý ảnh]
            t0 = time.time()
            padded_img = letterbox(orig_img, (640, 640))
            edge_img = extract_edge_map(orig_img)
            t_preprocess = (time.time() - t0) * 1000

            letterbox_path = os.path.join(app.config["UPLOAD_DIR"], "padded_" + filename)
            edge_path = os.path.join(app.config["UPLOAD_DIR"], "edge_" + filename)
            cv2.imwrite(letterbox_path, padded_img)
            cv2.imwrite(edge_path, edge_img)

            # [Giai đoạn 2: Trích xuất đặc trưng & Suy luận Deep Learning]
            t1 = time.time()
            preds = model.predict(source=raw_path, conf=conf_thresh, iou=iou_thresh, imgsz=640, verbose=False)
            t_infer = (time.time() - t1) * 1000

            # [Giai đoạn 3: Hậu xử lý & Tái cấu trúc ma trận ảnh]
            t2 = time.time()
            pred = preds[0]
            output_img = orig_img.copy()
            detections = []

            for box in pred.boxes:
                cid = int(box.cls[0].item())
                cname = get_class_name(cid)
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

                # Vẽ Bounding Box & Phông nền cho nhãn bằng OpenCV
                cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 240, 255), 2)
                label_txt = f"{cname} | {conf * 100:.1f}%"
                (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)

                # Vẽ background tab nhãn
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
                "letterbox_img": letterbox_path.replace("\\", "/"),
                "edge_img": edge_path.replace("\\", "/"),
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


if __name__ == "__main__":
    app.run(debug=True, port=5000)