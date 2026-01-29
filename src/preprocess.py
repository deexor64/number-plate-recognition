import cv2 as cv
import numpy as np

# Resize plate to optimal width while maintaining aspect ratio.
# Larger images are generally better for character recognition.
def resize_plate(image, target_width=300):
    height, width = image.shape[:2]
    if width < target_width:
        scale = target_width / width
        new_height = int(height * scale)
        # Use INTER_CUBIC for better upscaling quality
        return cv.resize(
            image, (target_width, new_height), interpolation=cv.INTER_CUBIC
        )
    return image

# Remove noise using Gaussian blur and bilateral filtering.
# Bilateral filter reduces noise while preserving edges.
def reduce_noise(image):
    denoised = cv.bilateralFilter(image, 9, 75, 75)
    return denoised

# Improve contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
# Works better than regular histogram equalization for varying lighting.
def enhance_contrast(image):
    if len(image.shape) == 3:
        # Convert to LAB color space for better contrast enhancement
        lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
        l, a, b = cv.split(lab)

        # Apply CLAHE to the L channel
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        # Merge channels and convert back to BGR
        enhanced = cv.merge([l, a, b])
        return cv.cvtColor(enhanced, cv.COLOR_LAB2BGR)
    else:
        # Grayscale image
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)

# Sharpen the image to make character edges more defined.
# Uses a simple sharpening kernel.
def sharpen_image(image):
    # Create sharpening kernel
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

    return cv.filter2D(image, -1, kernel)

# Convert to grayscale and apply adaptive thresholding.
# This creates a binary image ideal for character recognition. 
def prepare_for_ocr(image):
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply adaptive threshold for better text segmentation
    binary = cv.adaptiveThreshold(
        gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2
    )

    return binary

# Apply complete preprocessing pipeline to a license plate image
def preprocess_plate(plate_image, show_steps=False):
    if plate_image is None or plate_image.size == 0:
        return None

    # Store results
    results = {"original": plate_image.copy()}
    current = plate_image.copy()

    # Resize for optimal processing
    current = resize_plate(current)
    results["resized"] = current.copy()
    if show_steps:
        cv.imshow("1. Resized", current)

    # Reduce noise
    current = reduce_noise(current)
    results["denoised"] = current.copy()
    if show_steps:
        cv.imshow("2. Denoised", current)

    # Enhance contrast
    current = enhance_contrast(current)
    results["contrast_enhanced"] = current.copy()
    if show_steps:
        cv.imshow("3. Contrast Enhanced", current)

    # Sharpen image
    current = sharpen_image(current)
    results["sharpened"] = current.copy()
    if show_steps:
        cv.imshow("4. Sharpened", current)

    # Prepare for OCR (grayscale + threshold)
    current = prepare_for_ocr(current)
    results["final"] = current.copy()
    if show_steps:
        cv.imshow("5. Final (OCR Ready)", current)
        cv.waitKey(0)
        cv.destroyAllWindows()

    return results
