import os

import cv2 as cv
import numpy as np

from preprocess import preprocess_plate


class AppConfig:
    """
    Configuration class for the application.
    """

    CONFIDENCE_THRESHOLD = 0.5
    NMS_THRESHOLD = 0.4
    INPUT_SIZE = 416
    MODEL_CONFIG = "model/config/darknet-yolov3.cfg"
    MODEL_WEIGHTS = "model/weights/model.weights"
    PLATE_PADDING = 10
    OUTPUT_DIR = "output"
    FONT = cv.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.6
    FONT_THICKNESS = 2
    FONT_COLOR = (0, 0, 0)
    BG_COLOR = (0, 255, 0)


# Load YOLO model
def load_network(config_path, weights_path):
    print("Log: Loading network...")
    net = cv.dnn.readNetFromDarknet(config_path, weights_path)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    return net


# Preprocess image for cnn
def preprocess(image, config):
    blob = cv.dnn.blobFromImage(
        image,
        1 / 255.0,
        (config.INPUT_SIZE, config.INPUT_SIZE),
        (0, 0, 0),
        True,
        crop=False,
    )
    return blob


# Get output layer names (OpenCV version compatibility)
def get_outputs(net):
    layer_names = net.getLayerNames()
    output_layers = [
        layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()
    ]
    return output_layers


# Extract valid detections from network outputs
def extract_detections(outputs, w, h, config):
    detections = []

    for output in outputs:
        for detection in output:
            confidence = detection[5]
            if confidence > config.CONFIDENCE_THRESHOLD:
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                width = int(detection[2] * w)
                height = int(detection[3] * h)
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)

                detections.append(
                    {"coords": (x, y, width, height), "confidence": float(confidence)}
                )

    return detections


# Apply Non-Maximum Suppression to remove duplicates
def apply_nms(detections, config):
    if not detections:
        return []

    boxes = [d["coords"] for d in detections]
    confidences = [d["confidence"] for d in detections]

    indices = cv.dnn.NMSBoxes(
        boxes, confidences, config.CONFIDENCE_THRESHOLD, config.NMS_THRESHOLD
    )

    if hasattr(indices, "flatten"):
        indices = indices  # .flatten()

    if len(indices) > 0:
        return [detections[i] for i in indices]

    return []


# Extract license plate regions from detected areas
def extract_plate_regions(image, detections, config):
    plate_regions = []

    for i, detection in enumerate(detections):
        x, y, w, h = detection["coords"]

        # Add padding around detected region
        x_start = max(0, x - config.PLATE_PADDING)
        y_start = max(0, y - config.PLATE_PADDING)
        x_end = min(image.shape[1], x + w + config.PLATE_PADDING)
        y_end = min(image.shape[0], y + h + config.PLATE_PADDING)

        # Extract plate region
        plate_region = image[y_start:y_end, x_start:x_end]

        if plate_region.size > 0:
            plate_regions.append(
                {
                    "image": plate_region,
                    "coords": (x, y, w, h),
                    "confidence": detection["confidence"],
                    "index": i,
                }
            )

    return plate_regions


# Draw a single bounding box on the image
def draw_bounding_box(image, detection, index, config, processed_available=False):
    x, y, w, h = detection["coords"]
    conf = detection["confidence"]

    # Draw green box
    cv.rectangle(image, (x, y), (x + w, y + h), config.BG_COLOR, 2)

    # Create label
    status = "Processed" if processed_available else "Detected"
    label = f"Plate {index + 1}: {status} ({conf:.2f})"

    # Draw label background
    (text_width, text_height), _ = cv.getTextSize(
        label, config.FONT, config.FONT_SCALE, config.FONT_THICKNESS
    )
    cv.rectangle(
        image, (x, y - text_height - 10), (x + text_width, y), config.BG_COLOR, -1
    )

    # Draw label text
    cv.putText(
        image,
        label,
        (x, y - 5),
        config.FONT,
        config.FONT_SCALE,
        config.FONT_COLOR,
        config.FONT_THICKNESS,
    )


# Draw bounding boxes on image
def draw_results(image, detections, config, processed_available=False):
    result = image.copy()

    for i, detection in enumerate(detections):
        draw_bounding_box(result, detection, i, config, processed_available)

    return result


def preprocess_plates(plate_regions, show_preprocessing=False):
    processed_plates = []
    for plate_info in plate_regions:
        print(f"Log: Preprocessing plate {plate_info['index'] + 1}...")
        preprocessing_results = preprocess_plate(
            plate_info["image"], show_steps=show_preprocessing
        )
        if preprocessing_results:
            processed_plates.append(
                {
                    "original": preprocessing_results["original"],
                    "final": preprocessing_results["final"],
                    "all_steps": preprocessing_results,
                    "coords": plate_info["coords"],
                    "confidence": plate_info["confidence"],
                }
            )
            print(f"Log: Plate {plate_info['index'] + 1} preprocessed successfully")
        else:
            processed_plates.append(None)
            print(f"Error: Failed to preprocess plate {plate_info['index'] + 1}")
    return processed_plates


def run_detection(net, image, config):
    blob = preprocess(image, config)
    net.setInput(blob)

    print("Log: Running detection...")
    outputs = net.forward(get_outputs(net))

    detections = extract_detections(outputs, image.shape[1], image.shape[0], config)
    return apply_nms(detections, config)


def detect_plates(
    image_path,
    config=AppConfig(),
    show_preprocessing=False,
):
    # Load image
    print(f"+ Image path: {image_path}")

    image = cv.imread(image_path)
    if image is None:
        print("Error: Could not load image")
        return None

    # Image exists
    print(f"+ Image size: {image.shape[1]}×{image.shape[0]}")

    # Detect license plates using YOLO
    net = load_network(config.MODEL_CONFIG, config.MODEL_WEIGHTS)
    final_detections = run_detection(net, image, config)

    print(f"Log: Found {len(final_detections)} license plate(s)")

    if not final_detections:
        return {"detections": [], "processed_plates": [], "result_image": image}

    # Extract plate regions
    plate_regions = extract_plate_regions(image, final_detections, config)

    # Apply preprocessing to each detected plate
    processed_plates = preprocess_plates(plate_regions, show_preprocessing)

    # Create result image
    result_image = draw_results(
        image, final_detections, config, processed_available=bool(processed_plates)
    )

    return {
        "detections": final_detections,
        "processed_plates": processed_plates,
        "result_image": result_image,
        "original_image": image,
    }


def show_plate_comparison(i, plate_data, config):
    # Resize for consistent display
    original = cv.resize(plate_data["original"], (300, 100))

    # Convert final (grayscale) to BGR for comparison
    final = plate_data["final"]
    if len(final.shape) == 2:
        final_bgr = cv.cvtColor(final, cv.COLOR_GRAY2BGR)
    else:
        final_bgr = final

    final_display = cv.resize(final_bgr, (300, 100))

    # Create side-by-side comparison
    comparison = np.hstack([original, final_display])

    # Add labels
    cv.putText(
        comparison,
        "Original",
        (10, 20),
        config.FONT,
        config.FONT_SCALE,
        config.BG_COLOR,
        config.FONT_THICKNESS,
    )
    cv.putText(
        comparison,
        "Preprocessed",
        (310, 20),
        config.FONT,
        config.FONT_SCALE,
        config.BG_COLOR,
        config.FONT_THICKNESS,
    )

    cv.imshow(f"Plate {i + 1}: Before vs After", comparison)


def show_results(results, config=AppConfig()):
    """Display detection and preprocessing results."""
    if not results or not results["detections"]:
        print("No results to display")
        return

    # Show main result
    cv.imshow("Detection Results", results["result_image"])

    # Show individual plate comparisons
    for i, plate_data in enumerate(results["processed_plates"]):
        if plate_data:
            show_plate_comparison(i, plate_data, config)

    print("Press any key to close windows...")
    cv.waitKey(0)
    cv.destroyAllWindows()


def save_results(results, config=AppConfig()):
    """Save processing results to files."""
    if not results or not results["detections"]:
        return

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Save main result
    cv.imwrite(f"{config.OUTPUT_DIR}/detection_result.jpg", results["result_image"])

    # Save individual plates
    for i, plate_data in enumerate(results["processed_plates"]):
        if plate_data:
            cv.imwrite(
                f"{config.OUTPUT_DIR}/plate_{i + 1}_original.jpg",
                plate_data["original"],
            )
            cv.imwrite(
                f"{config.OUTPUT_DIR}/plate_{i + 1}_processed.jpg", plate_data["final"]
            )

    print(f"💾 Results saved to {config.OUTPUT_DIR}/")
