import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

# ⚠️ Order must match training! Check your Colab output: "Classes: {...}"
# Colab output: cardboard, glass, metal, paper, plastic, trash
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

IMG_SIZE = (224, 224)

# Load model once
model = None
try:
    from tensorflow.keras.models import load_model
    model = load_model("waste_classifier.h5")
    print("✅ Model loaded!")
except Exception as e:
    print(f"❌ Model error: {e}")

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded."}), 500

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    file = request.files["image"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file. Use PNG, JPG, JPEG or WEBP."}), 400

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(file.filename))
    file.save(filepath)

    try:
        img_array = preprocess(filepath)
        preds = model.predict(img_array, verbose=0)[0]

        top_idx   = int(np.argmax(preds))
        top_label = CLASS_NAMES[top_idx]
        top_conf  = float(preds[top_idx])

        all_preds = sorted(
            [{"label": CLASS_NAMES[i], "confidence": round(float(preds[i]), 4)} for i in range(len(CLASS_NAMES))],
            key=lambda x: x["confidence"], reverse=True
        )

        return jsonify({
            "predicted_label": top_label,
            "confidence": round(top_conf, 4),
            "top_predictions": all_preds
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "Image too large. Max 16MB."}), 413

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
