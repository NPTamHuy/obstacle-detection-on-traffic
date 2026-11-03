
import os
import cv2

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


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def detect_objects(image_path):
    """
    Perform object detection using YOLO.
    Save the image with bounding boxes.
    Return detection information and output image URL.
    """

    # Read uploaded image
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Cannot read uploaded image.")

    # Run YOLO inference
    results = model.predict(
        source=image,
        conf=0.25,
        iou=0.45,
        verbose=False
    )

    result = results[0]

    detections = []

    # Process bounding boxes
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        confidence = float(box.conf[0])
        class_id = int(box.cls[0])

        class_name = model.names[class_id]

        detections.append({
            "class_name": class_name,
            "confidence": round(confidence * 100, 2),
            "bbox": [x1, y1, x2, y2]
        })

        # Draw bounding box
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Draw class label and confidence
        label = f"{class_name} {confidence:.2f}"

        cv2.putText(
            image,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # Save detection result
    filename = os.path.basename(image_path)
    name, extension = os.path.splitext(filename)

    output_filename = f"{name}_detected{extension}"

    output_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        output_filename
    )

    cv2.imwrite(output_path, image)

    output_url = url_for(
        "static",
        filename=f"uploads/{output_filename}"
    )

    return detections, output_url


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

        # Save uploaded image
        filename = secure_filename(file.filename)

        image_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        file.save(image_path)

        # Run YOLO detection
        try:
            detections, result_url = detect_objects(image_path)

        except Exception as error:
            flash(f"Lỗi xử lý ảnh: {error}")
            return redirect(url_for("index"))

        return render_template(
            "index.html",
            original_url=url_for(
                "static",
                filename=f"uploads/{filename}"
            ),
            result_url=result_url,
            filename=filename,
            detections=detections,
            detection_count=len(detections)
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
