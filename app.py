from flask import Flask, request, render_template
import os
import cv2

from ml.image_utils import extract_cheek
from ml.undertone import detect_undertone
from ml.recommender import recommend_top3
from ml.image_utils import extract_cheek, normalize_lighting

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":

        # 1️⃣ Get uploaded image & brand
        file = request.files["image"]
        selected_brand = request.form.get("brand")

        input_path = os.path.join(app.config["UPLOAD_FOLDER"], "input.jpg")
        cheek_path = os.path.join(app.config["UPLOAD_FOLDER"], "cheek.jpg")

        # 2️⃣ Save uploaded image
        file.save(input_path)

        # 3️⃣ Extract cheek
        cheek = extract_cheek(input_path)
        if cheek is None:
            return "Face not detected. Try another image."
        cheek = normalize_lighting(cheek)   

        cv2.imwrite(cheek_path, cheek)

        # 4️⃣ Detect undertone
        undertone, _, _ = detect_undertone(cheek_path)

        # 5️⃣ Get TOP-3 lipstick recommendations
        results = recommend_top3(cheek_path, undertone, selected_brand)

        # 6️⃣ Build HTML response
        return render_template(
    "index.html",
    result={
        "undertone": undertone,
        "recommendations": results
    }
)

    # 🟢 🔵 GET BLOCK (runs when page opens)
    return render_template("index.html",result=None)


if __name__ == "__main__":
    app.run(debug=True)