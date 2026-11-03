
import os
import cv2
import numpy as np

from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from ultralytics import YOLO


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

    if request.method == "POST":

        if "image" not in request.files:
            flash("Không tìm thấy file ảnh.")
            return redirect(url_for("index"))

        file = request.files["image"]

        if file.filename == "":
            flash("Vui lòng chọn ảnh.")
            return redirect(url_for("index"))

        if not allowed_file(file.filename):
            flash("Định dạng ảnh không được hỗ trợ.")
            return redirect(url_for("index"))

        # Read thresholds
        try:
            confidence_threshold = float(
                request.form.get("confidence", 0.25)
            )

            iou_threshold = float(
                request.form.get("iou", 0.45)
            )

            if not 0 <= confidence_threshold <= 1:
                raise ValueError("Confidence must be between 0 and 1.")

            if not 0 <= iou_threshold <= 1:
                raise ValueError("IoU must be between 0 and 1.")

        except ValueError as error:
            flash(f"Ngưỡng không hợp lệ: {error}")
            return redirect(url_for("index"))

        filename = secure_filename(file.filename)

        image_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        file.save(image_path)

        # Read original image
        image = cv2.imread(image_path)

        if image is None:
            flash("Không thể đọc ảnh đã upload.")
            return redirect(url_for("index"))

        try:
            # Generate preprocessing images
            preprocessing = preprocess_image(
                image,
                filename
            )

            # YOLO inference on original image
            detections, result_url = detect_objects(
                image,
                filename,
                confidence_threshold,
                iou_threshold
            )

        except Exception as error:
            flash(f"Lỗi xử lý ảnh: {error}")
            return redirect(url_for("index"))

        return render_template(
            "index.html",
            original_url=url_for(
                "static",
                filename=f"uploads/{filename}"
            ),
            letterbox_url=preprocessing["letterbox_url"],
            edge_url=preprocessing["edge_url"],
            result_url=result_url,
            filename=filename,
            detections=detections,
            detection_count=len(detections),
            confidence=confidence_threshold,
            iou=iou_threshold
        )

    return render_template(
        "index.html",
        confidence=0.25,
        iou=0.45
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
