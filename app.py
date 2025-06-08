from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("char_model.h5")


# Class ID -> Symbol mapping
def class_id_to_symbol(class_id):
    symbol_map = {
        0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
        5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
        10: '+', 11: '-', 12: '*', 13: '/', 14: '=', 15: '÷'
    }
    return symbol_map.get(class_id, '?')


# Resize with proportional padding to 28x28
def resize_with_padding(img, size=28):
    h, w = img.shape
    scale = size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    padded = np.full((size, size), 0, dtype=np.uint8)
    pad_top = (size - new_h) // 2
    pad_left = (size - new_w) // 2
    padded[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    return padded


# Extract characters from image
def extract_characters(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = 255 - img
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    chars = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > 100:
            roi = thresh[y:y + h, x:x + w]
            resized = resize_with_padding(roi)
            chars.append((x, resized))

    chars = sorted(chars, key=lambda x: x[0])
    return [char for _, char in chars]


# Predict using the model
def predict_expression(model, char_images):
    expression = ""
    for img in char_images:
        img = img.reshape(1, 28, 28, 1).astype("float32") / 255.0
        pred = model.predict(img, verbose=0)
        class_id = np.argmax(pred)
        symbol = class_id_to_symbol(class_id)
        expression += symbol
    return expression


# Evaluate the expression
def evaluate_expression(expression):
    try:
        expression = expression.replace('÷', '/')  # Replace ÷ with /
        return eval(expression)
    except Exception:
        return "Invalid expression"


# Flask settings
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'


@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    expression = ""
    image_path = ""

    if request.method == "POST":
        if 'file' not in request.files:
            return render_template("index.html", error="No file uploaded.")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="No file selected.")

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Extract characters from image, predict and evaluate
        char_images = extract_characters(filepath)
        expression = predict_expression(model, char_images)
        result = evaluate_expression(expression)
        image_path = filepath

    return render_template("index.html", result=result, expression=expression, image_path=image_path)


if __name__ == "__main__":
    app.run(debug=True)
