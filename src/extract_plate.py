import cv2 as cv
import numpy as np

CONFIDENCE_THRESHOLD = 0.5
INPUT_SIZE = 416
MODEL_CONFIG = "model/config/darknet-yolov3.cfg"
MODEL_WEIGHTS = "model/weights/model.weights"
PLATE_PADDING = 2


def extract_plate(image_path) -> dict | None:
    # Load image and check it was loaded successfully
    img = cv.imread(image_path)
    if img is None:
        return None

    h, w = img.shape[:2]

    # Load YOLO model
    net = cv.dnn.readNetFromDarknet(MODEL_CONFIG, MODEL_WEIGHTS)

    # Convert image to blob for the CNN
    blob = cv.dnn.blobFromImage(
        img, 1 / 255.0, (INPUT_SIZE, INPUT_SIZE), swapRB=True, crop=False
    )
    net.setInput(blob)

    # Get output layers and run forward pass
    ln = net.getLayerNames()
    output_layers = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(output_layers)

    # Extract coordinates
    best_box = None
    max_conf = 0

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            conf = scores[0]  # Assuming class 0 is 'plate'
            if conf > CONFIDENCE_THRESHOLD and conf > max_conf:
                max_conf = conf
                # Convert YOLO relative coords to pixel coords
                cx, cy, bw, bh = detection[0:4] * np.array([w, h, w, h])
                x, y = int(cx - bw / 2), int(cy - bh / 2)
                best_box = (x, y, int(bw), int(bh))

    if not best_box:
        return None

    # Crop with Padding
    x, y, bw, bh = best_box
    pad = PLATE_PADDING

    # Ensure crop stays within image boundaries
    y1, y2 = max(0, y - pad), min(h, y + bh + pad)
    x1, x2 = max(0, x - pad), min(w, x + bw + pad)

    cropped_plate = img[y1:y2, x1:x2]

    return {
        "original_image": img,
        "cropped_image": cropped_plate,
        "coords": (x, y, bw, bh),
    }
