
import os
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Application configuration
app.config["SECRET_KEY"] = "sea-detection-secret"
app.config["UPLOAD_FOLDER"] = os.path.join("static", "uploads")

# Allowed image extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}

# Create upload directory if it does not exist
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


def allowed_file(filename):
    """Check whether the uploaded file has an allowed extension."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@app.route("/", methods=["GET", "POST"])
def index():
    image_url = None

    if request.method == "POST":
        # Check whether an image was submitted
        if "image" not in request.files:
            flash("Không tìm thấy file ảnh.")
            return redirect(url_for("index"))

        file = request.files["image"]

        # Check whether a file was selected
        if file.filename == "":
            flash("Vui lòng chọn một ảnh.")
            return redirect(url_for("index"))

        # Validate and save the image
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            file_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                filename
            )

            file.save(file_path)

            image_url = url_for(
                "static",
                filename=f"uploads/{filename}"
            )

            return render_template(
                "index.html",
                image_url=image_url,
                filename=filename
            )

        flash("Định dạng ảnh không được hỗ trợ.")

    return render_template("index.html", image_url=image_url)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
