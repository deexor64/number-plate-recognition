#!/usr/bin/env python3
"""
Simple License Plate Detection - University Project
==================================================

Key concepts:
1. Image preprocessing and normalization  
2. YOLO neural network inference
3. Non-Maximum Suppression (NMS)
4. Coordinate extraction and visualization
"""

import cv2 as cv
import numpy as np
import os

# Simple configuration
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
INPUT_SIZE = 416

def load_network():
    """Load YOLO neural network."""
    print("🧠 Loading network...")
    net = cv.dnn.readNetFromDarknet(
        "model/config/darknet-yolov3.cfg",
        "model/weights/model.weights"
    )
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    return net

def get_outputs(net):
    """Get output layer names (OpenCV version compatibility)."""
    layers = net.getLayerNames()
    unconnected = net.getUnconnectedOutLayers()
    
    if len(unconnected.shape) > 1:
        return [layers[i[0] - 1] for i in unconnected]
    else:
        return [layers[i - 1] for i in unconnected]

def preprocess(image):
    """
    Preprocess image for neural network.
    Theory: Normalize pixels [0,255] → [0,1], resize to 416x416
    """
    blob = cv.dnn.blobFromImage(image, 1/255.0, (INPUT_SIZE, INPUT_SIZE), 
                                (0,0,0), True, crop=False)
    return blob

def extract_detections(outputs, w, h):
    """Extract valid detections from network outputs."""
    detections = []
    
    for output in outputs:
        for detection in output:
            confidence = detection[5] if len(detection) > 5 else detection[4]
            
            if confidence > CONFIDENCE_THRESHOLD:
                # Convert normalized coordinates to pixels
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                width = int(detection[2] * w)
                height = int(detection[3] * h)
                x = int(center_x - width/2)
                y = int(center_y - height/2)
                
                detections.append({
                    'coords': (x, y, width, height),
                    'confidence': float(confidence)
                })
    
    return detections

def apply_nms(detections):
    """Apply Non-Maximum Suppression to remove duplicates."""
    if not detections:
        return []
    
    boxes = [d['coords'] for d in detections]
    confidences = [d['confidence'] for d in detections]
    
    indices = cv.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    
    if len(indices) > 0:
        if hasattr(indices[0], '__len__'):
            indices = [i[0] for i in indices]
        else:
            indices = indices.flatten() if hasattr(indices, 'flatten') else indices
        
        return [detections[i] for i in indices]
    
    return []

def draw_results(image, detections):
    """Draw bounding boxes on image."""
    result = image.copy()
    
    for detection in detections:
        x, y, w, h = detection['coords']
        conf = detection['confidence']
        
        # Draw green box
        cv.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw label
        label = f"License Plate: {conf:.2f}"
        cv.putText(result, label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 
                  0.6, (0, 255, 0), 2)
    
    return result

def detect_license_plates(image_path):
    """
    Main detection function.
    
    Returns detected coordinates and annotated image.
    """
    print(f"🔍 Processing: {image_path}")
    
    # Load image
    image = cv.imread(image_path)
    if image is None:
        print("❌ Could not load image")
        return None
    
    print(f"📷 Image size: {image.shape[1]}×{image.shape[0]}")
    
    # Load network and process
    net = load_network()
    blob = preprocess(image)
    net.setInput(blob)
    
    print("🧠 Running detection...")
    outputs = net.forward(get_outputs(net))
    
    # Extract and filter detections
    detections = extract_detections(outputs, image.shape[1], image.shape[0])
    final_detections = apply_nms(detections)
    
    # Create result image
    result_image = draw_results(image, final_detections)
    
    return {
        'detections': final_detections,
        'image': result_image,
        'original': image
    }

def main():
    """
    Main function - specify image path here.
    """
    # TODO: Change this to your image path
    image_path = "images/car-sl.jpg"  # <-- CHANGE THIS PATH
    
    print("🚗 License Plate Detection")
    print("=" * 30)
    
    # Run detection
    results = detect_license_plates(image_path)
    
    if results is None:
        return
    
    # Print coordinates
    print(f"\n✅ Found {len(results['detections'])} license plate(s)")
    print("\n📍 Coordinates:")
    
    for i, det in enumerate(results['detections']):
        x, y, w, h = det['coords']
        conf = det['confidence']
        print(f"  Plate {i+1}: x={x}, y={y}, width={w}, height={h}, conf={conf:.3f}")
    
    # Show results
    print("\n🖼️ Showing results (press any key to close)...")
    # cv.imshow('Original', results['original'])
    cv.imshow('Detection Results', results['image'])
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    # Save result
    # os.makedirs("output", exist_ok=True)
    # cv.imwrite("output/result.jpg", results['image'])
    # print("💾 Saved to: output/result.jpg")

if __name__ == "__main__":
    main()