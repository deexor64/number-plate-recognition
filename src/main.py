import cv2
import numpy as np

# -----------------------------
# 1. Load image
# -----------------------------
# Read input image from disk
image = cv2.imread("images/car2.webp")

if image is None:
    raise ValueError("Image not found or path is incorrect")

# Keep a copy for drawing results
original = image.copy()

# -----------------------------
# 2. Convert to grayscale
# -----------------------------
# Color information is not required for edge-based detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# -----------------------------
# 3. Noise reduction
# -----------------------------
# Gaussian blur reduces small variations that cause false edges
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# -----------------------------
# 4. Edge detection
# -----------------------------
# Canny detects strong intensity changes (edges)
edges = cv2.Canny(blur, 100, 200)

# -----------------------------
# 5. Morphological processing
# -----------------------------
# Purpose: connect broken edges of characters
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# Closing = dilation followed by erosion
# This helps merge character edges into a single block
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# -----------------------------
# 6. Find contours
# -----------------------------
# Contours represent connected regions
contours, _ = cv2.findContours(
    closed,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

plate_candidates = []

# -----------------------------
# 7. Filter contours
# -----------------------------
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)

    aspect_ratio = w / float(h)
    area = w * h

    # Heuristic filters for number plates
    # These values may need tuning
    if 2.0 < aspect_ratio < 6.0 and area > 3000:
        plate_candidates.append((x, y, w, h))
        cv2.rectangle(original, (x, y), (x + w, y + h), (0, 255, 0), 2)

# -----------------------------
# 8. Crop the best plate candidate
# -----------------------------
# Assumption: largest valid rectangle is the plate
plate_image = None

if plate_candidates:
    plate_candidates = sorted(
        plate_candidates,
        key=lambda r: r[2] * r[3],
        reverse=True
    )
    x, y, w, h = plate_candidates[0]
    plate_image = gray[y:y+h, x:x+w]

# -----------------------------
# 9. Show results
# -----------------------------
cv2.imshow("Detected Plate Region", original)
print(plate_image)

if plate_image is not None:
    cv2.imshow("Cropped Plate", plate_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
