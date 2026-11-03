
import os
import cv2
import numpy as np
import pandas as pd

from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from collections import Counter

app = Flask(__name__)

app.config["SECRET_KEY"] = "sea-detection-secret"
app.config["UPLOAD_FOLDER"] = os.path.join("static", "uploads")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


# Load YOLO model
MODEL_PATH = os.path.join(
    "sea_detection_sea1",
    "weights",
    "best.pt"
)

RESULTS_PATH = os.path.join(
    "sea_detection_sea1",
    "results.csv"
)

model = YOLO(MODEL_PATH)


# Map class IDs to readable labels
# Replace these examples with the actual classes of your dataset.
CLASS_LABELS = {
    0: "Boat",
    1: "Buoy",
    2: "Person",
    3: "Floating Object",
    4: "Ship",
    5: "Other Object"
}

def summarize_detections(detections):
    """
    Thống kê số lượng vật thể theo class.
    """
    counts = Counter(
        detection["class_name"]
        for detection in detections
    )

    return [
        {
            "class_name": class_name,
            "count": count
        }
        for class_name, count in counts.items()
    ]


def load_metrics():
    """
    Read YOLO training metrics from results.csv.
    Return the row with the highest mAP50.
    """

    if not os.path.exists(RESULTS_PATH):
        return None

    try:
        df = pd.read_csv(RESULTS_PATH)

        # Remove unnecessary spaces in column names
        df.columns = df.columns.str.strip()

        # YOLO training metric columns
        metric_columns = {
            "precision": "metrics/precision(B)",
            "recall": "metrics/recall(B)",
            "map50": "metrics/mAP50(B)",
            "map50_95": "metrics/mAP50-95(B)"
        }

        # Check required columns
        missing_columns = [
            column
            for column in metric_columns.values()
            if column not in df.columns
        ]

        if missing_columns:
            print("Missing metric columns:", missing_columns)
            return None

        # Convert metric values to numeric
        for column in metric_columns.values():
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

        # Find the epoch with the highest mAP50
        best_index = df["metrics/mAP50(B)"].idxmax()

        if pd.isna(best_index):
            return None

        best_row = df.loc[best_index]

        return {
            "precision": round(
                float(best_row[metric_columns["precision"]]) * 100,
                2
            ),
            "recall": round(
                float(best_row[metric_columns["recall"]]) * 100,
                2
            ),
            "map50": round(
                float(best_row[metric_columns["map50"]]) * 100,
                2
            ),
            "map50_95": round(
                float(best_row[metric_columns["map50_95"]]) * 100,
                2
            ),
            "epochs": int(len(df)),
            "best_epoch": int(best_index) + 1
        }

    except Exception as error:
        print(f"Cannot load training metrics: {error}")
        return None



def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def get_class_name(class_id):
    return CLASS_LABELS.get(
        class_id,
        model.names.get(class_id, str(class_id))
    )


def letterbox(image, new_shape=(640, 640), color=(114, 114, 114)):
    """
    Resize image while preserving aspect ratio.
    Add padding to reach the target dimensions.
    """

    original_height, original_width = image.shape[:2]

    target_height, target_width = new_shape

    # Calculate resize ratio
    ratio = min(
        target_width / original_width,
        target_height / original_height
    )

    new_width = int(round(original_width * ratio))
    new_height = int(round(original_height * ratio))

    # Resize image
    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_LINEAR
    )

    # Calculate padding
    pad_width = target_width - new_width
    pad_height = target_height - new_height

    left = pad_width // 2
    right = pad_width - left

    top = pad_height // 2
    bottom = pad_height - top

    # Add padding
    output = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=color
    )

    return output


def extract_edge_map(image):
    """
    Convert image to grayscale,
    apply Gaussian Blur and Canny Edge Detection.
    """

    # Convert BGR image to grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Reduce image noise
    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # Detect edges
    edges = cv2.Canny(
        blurred,
        50,
        150
    )

    # Apply color map for visualization
    edge_colored = cv2.applyColorMap(
        edges,
        cv2.COLORMAP_OCEAN
    )

    return edge_colored


def save_image(image, filename):
    """Save an image and return its static URL."""

    output_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    success = cv2.imwrite(output_path, image)

    if not success:
        raise ValueError(f"Cannot save image: {filename}")

    return url_for(
        "static",
        filename=f"uploads/{filename}"
    )


def preprocess_image(image, filename):
    """
    Generate Letterbox and Canny Edge images.
    These images are for pipeline visualization.
    """

    name, extension = os.path.splitext(filename)

    # Step 1: Letterbox
    letterbox_image = letterbox(
        image,
        new_shape=(640, 640)
    )

    letterbox_url = save_image(
        letterbox_image,
        f"{name}_letterbox{extension}"
    )

    # Step 2: Grayscale -> Gaussian Blur -> Canny
    edge_image = extract_edge_map(image)

    edge_url = save_image(
        edge_image,
        f"{name}_edges{extension}"
    )

    return {
        "letterbox_url": letterbox_url,
        "edge_url": edge_url
    }


def detect_objects(image, filename, confidence_threshold, iou_threshold):
    """
    Run YOLO on the original image and draw detection results.
    """

    results = model.predict(
        source=image,
        conf=confidence_threshold,
        iou=iou_threshold,
        verbose=False
    )

    result = results[0]

    # Plot detection boxes on the image
    output_image = result.plot()

    detections = []

    for box in result.boxes:

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist()
        )

        confidence = float(box.conf[0])
        class_id = int(box.cls[0])

        class_name = get_class_name(class_id)

        detections.append({
            "class_id": class_id,
            "class_name": class_name,
            "confidence": round(confidence * 100, 2),
            "bbox": [x1, y1, x2, y2]
        })

    name, extension = os.path.splitext(filename)

    result_url = save_image(
        output_image,
        f"{name}_detected{extension}"
    )

    return detections, result_url


@app.route("/", methods=["GET", "POST"])
def index():
    metrics = load_metrics()

    if request.method == "POST":
        file = request.files.get("image")

        if not file or file.filename == "":
            return render_template(
                "index.html",
                metrics=metrics,
                error="Please select an image."
            )

        if not allowed_file(file.filename):
            return render_template(
                "index.html",
                metrics=metrics,
                error="Unsupported image format.",
            )

        filename = secure_filename(file.filename)

        image_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        file.save(image_path)

        confidence = float(request.form.get("confidence", 0.25))
        iou = float(request.form.get("iou", 0.45))

        # Preprocessing
        letterbox_url, edge_url = preprocess_image(
            image_path,
            filename
        )

        # YOLO detection
        detections, result_url = detect_objects(
            image_path,
            filename,
            confidence,
            iou
        )

        # Detection statistics
        detection_summary = summarize_detections(detections)

        return render_template(
            "index.html",
            metrics=metrics,
            original_url=url_for(
                "static",
                filename=f"uploads/{filename}"
            ),
            letterbox_url=letterbox_url,
            edge_url=edge_url,
            result_url=result_url,
            filename=filename,
            detections=detections,
            detection_summary=detection_summary,
            detection_count=len(detections),
            confidence=confidence,
            iou=iou
        )

    return render_template(
        "index.html",
        metrics=metrics
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
