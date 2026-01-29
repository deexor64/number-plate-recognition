import os

import cv2 as cv
import numpy as np

from preprocess import preprocess_plate

# Configuration
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
INPUT_SIZE = 416


# Load YOLO model
def load_network():
    print("Log: Loading network...")
    net = cv.dnn.readNetFromDarknet(
        "model/config/darknet-yolov3.cfg", "model/weights/model.weights"
    )
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    return net


# Preprocess image for cnn
def preprocess(image):
    blob = cv.dnn.blobFromImage(
        image, 1 / 255.0, (INPUT_SIZE, INPUT_SIZE), (0, 0, 0), True, crop=False
    )
    return blob


# Get output layer names (OpenCV version compatibility)
def get_outputs(net):
    layers = net.getLayerNames()
    unconnected = net.getUnconnectedOutLayers()

    if len(unconnected.shape) > 1:
        return [layers[i[0] - 1] for i in unconnected]
    else:
        return [layers[i - 1] for i in unconnected]


# Extract valid detections from network outputs
def extract_detections(outputs, w, h):
    detections = []

    for output in outputs:
        for detection in output:
            confidence = detection[5] if len(detection) > 5 else detection[4]

            if confidence > CONFIDENCE_THRESHOLD:
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
def apply_nms(detections):
    if not detections:
        return []

    boxes = [d["coords"] for d in detections]
    confidences = [d["confidence"] for d in detections]

    indices = cv.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    if len(indices) > 0:
        if hasattr(indices[0], "__len__"):
            indices = [i[0] for i in indices]
        else:
            indices = indices.flatten() if hasattr(indices, "flatten") else indices

        return [detections[i] for i in indices]

    return []


# Extract license plate regions from detected areas
def extract_plate_regions(image, detections):
    plate_regions = []

    for i, detection in enumerate(detections):
        x, y, w, h = detection["coords"]

        # Add padding around detected region
        padding = 10
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(image.shape[1], x + w + padding)
        y_end = min(image.shape[0], y + h + padding)

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


# Draw bounding boxes on image
def draw_results(image, detections, processed_available=False):
    result = image.copy()

    for i, detection in enumerate(detections):
        x, y, w, h = detection["coords"]
        conf = detection["confidence"]

        # Draw green box
        cv.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Create label
        status = "Processed" if processed_available else "Detected"
        label = f"Plate {i + 1}: {status} ({conf:.2f})"

        # Draw label background
        (text_width, text_height), _ = cv.getTextSize(
            label, cv.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv.rectangle(
            result, (x, y - text_height - 10), (x + text_width, y), (0, 255, 0), -1
        )

        # Draw label text
        cv.putText(
            result, label, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
        )

    return result


def detect_plates(image_path, show_preprocessing=False):
    # Load image
    print(f"+ Image path: {image_path}")

    image = cv.imread(image_path)
    if image is None:
        print("Error: Could not load image")
        return None
    
    # Image exists
    print(f"+ Image size: {image.shape[1]}×{image.shape[0]}")

    # Detect license plates using YOLO
    net = load_network()
    blob = preprocess(image)
    net.setInput(blob)

    print("Log: Running detection...")
    outputs = net.forward(get_outputs(net))

    detections = extract_detections(outputs, image.shape[1], image.shape[0])
    final_detections = apply_nms(detections)

    print(f"Log: Found {len(final_detections)} license plate(s)")

    if not final_detections:
        return {
            "detections": [], 
            "processed_plates": [], 
            "result_image": image
        }

    # Extract plate regions
    plate_regions = extract_plate_regions(image, final_detections)

    # Apply preprocessing to each detected plate
    processed_plates = []
    
    # Preprocess detected plates to extract characters
    for plate_info in plate_regions:
        print(f"Log: Preprocessing plate {plate_info['index'] + 1}...")

        # Apply image preprocessing
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

    # Create result image
    result_image = draw_results(
        image, final_detections, processed_available=bool(processed_plates)
    )

    return {
        "detections": final_detections,
        "processed_plates": processed_plates,
        "result_image": result_image,
        "original_image": image,
    }


def show_results(results):
    """Display detection and preprocessing results."""
    if not results or not results["detections"]:
        print("No results to display")
        return

    # Show main result
    cv.imshow("Detection Results", results["result_image"])

    # Show individual plate comparisons
    for i, plate_data in enumerate(results["processed_plates"]):
        if plate_data:
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
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv.putText(
                comparison,
                "Preprocessed",
                (310, 20),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            cv.imshow(f"Plate {i + 1}: Before vs After", comparison)

    print("Press any key to close windows...")
    cv.waitKey(0)
    cv.destroyAllWindows()


def save_results(results, output_dir="output"):
    """Save processing results to files."""
    if not results or not results["detections"]:
        return

    os.makedirs(output_dir, exist_ok=True)

    # Save main result
    cv.imwrite(f"{output_dir}/detection_result.jpg", results["result_image"])

    # Save individual plates
    for i, plate_data in enumerate(results["processed_plates"]):
        if plate_data:
            cv.imwrite(
                f"{output_dir}/plate_{i + 1}_original.jpg", plate_data["original"]
            )
            cv.imwrite(f"{output_dir}/plate_{i + 1}_processed.jpg", plate_data["final"])

    print(f"💾 Results saved to {output_dir}/")
    