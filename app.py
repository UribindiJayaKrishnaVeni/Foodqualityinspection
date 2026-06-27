from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import json

app = Flask(__name__)

# ----------------------------------------------------------------------
# Load the trained model
# ----------------------------------------------------------------------
model = tf.keras.models.load_model("food_quality_model_36class.keras")

# ----------------------------------------------------------------------
# Load class names
# Prefers class_names.json (lightweight, deployment-friendly).
# Falls back to reading folder names from data_split/train if that
# file isn't present (useful for local-only runs).
# ----------------------------------------------------------------------
if os.path.exists("class_names.json"):
    with open("class_names.json") as f:
        class_names = json.load(f)
elif os.path.exists("data_split/train"):
    class_names = sorted(os.listdir("data_split/train"))
else:
    raise FileNotFoundError(
        "Could not find class_names.json or data_split/train. "
        "One of these is required to map predictions to class labels."
    )

img_size = (224, 224)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ----------------------------------------------------------------------
# Confidence threshold for "Unknown" detection.
# If the model's top prediction confidence is below this percentage,
# we report "Unknown" instead of forcing one of the 36 trained classes.
# This is a heuristic (softmax always sums to 100%, so the model
# technically always "picks" something) -- it is not a perfect
# out-of-distribution detector, but it catches clearly unrelated
# images reasonably well.
# ----------------------------------------------------------------------
CONFIDENCE_THRESHOLD = 90.0  # percent


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(image_path)

    try:
        img = image.load_img(image_path, target_size=img_size)
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array, verbose=0)
        top_index = np.argmax(preds)
        confidence_value = float(np.max(preds)) * 100  # as percentage
        predicted_class = class_names[top_index]

        # ----- Unknown image check -----
        is_unknown = confidence_value < CONFIDENCE_THRESHOLD

        if is_unknown:
            prediction_label = "Unknown"
            confidence_display = "100%"  # don't send a confidence value for Unknown
        else:
            prediction_label = predicted_class
            confidence_display = f"{confidence_value:.2f}%"

        return jsonify({
            "prediction": prediction_label,
            "confidence": confidence_display,   # will be null/None when Unknown
            "is_unknown": is_unknown,
            "raw_top_class": predicted_class,    # for your own debugging/demo explanation
            "image_path": "/" + image_path
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)