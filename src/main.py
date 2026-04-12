import base64
import os

import cv2 as cv
import numpy as np
from flask import Flask, jsonify, render_template, request

from extract_plate import extract_plate
from perform_ocr import perform_ocr
from preprocess_plate import preprocess_plate

app = Flask(__name__)


def encode_image(image):
    # Converts an OpenCV image to a base64 string for HTML display
    _, buffer = cv.imencode(".jpg", image)
    return base64.b64encode(buffer).decode("utf-8")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    # No image uploaded
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    # Read image directly from memory
    file = request.files["image"]

    filestr = file.read()
    npimg = np.frombuffer(filestr, np.uint8)
    img = cv.imdecode(npimg, cv.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Failed to read image"}), 400

    # Temporarily save for extract_plate (since it expects a path)
    temp_path = "temp_upload.jpg"
    cv.imwrite(temp_path, img)

    # Extract Plate
    results = extract_plate(temp_path)
    if results is None:
        os.remove(temp_path)
        return jsonify({"error": "No license plate detected"}), 200

    # Preprocess and OCR
    processed = preprocess_plate(results["cropped_image"])

    if processed is None:
        os.remove(temp_path)
        return jsonify({"error": "Failed to preprocess plate"}), 400

    plate_text = perform_ocr(processed["final"])

    # Draw bounding box on original image for display
    x, y, w, h = results["coords"]
    display_img = results["original_image"].copy()
    cv.rectangle(display_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv.putText(
        display_img,
        plate_text,
        (x, y - 10),
        cv.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
    )

    # Prepare ALL intermediate steps for the frontend
    encoded_process_steps = {}
    for step_name, img_data in processed.items():
        encoded_process_steps[step_name] = encode_image(img_data)

    # Prepare data for UI
    response = {
        "original": encode_image(display_img),
        "cropped": encode_image(results["cropped_image"]),
        "steps": encoded_process_steps,
        "text": plate_text,
    }

    os.remove(temp_path)
    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
