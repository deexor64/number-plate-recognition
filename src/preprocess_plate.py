import cv2 as cv
import numpy as np


def gray_scale(image):
    if len(image.shape) == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray = image

    return gray


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


def reduce_noise(image):
    denoised = cv.bilateralFilter(image, 9, 75, 75)
    return denoised


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


def sharpen_image(image):
    # Create sharpening kernel
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

    return cv.filter2D(image, -1, kernel)


# Apply complete preprocessing pipeline to a license plate image
def preprocess_plate(plate_image):
    if plate_image is None or plate_image.size == 0:
        return None

    # Convert to grayscale
    current = gray_scale(plate_image)
    results = [("gray", current.copy())]

    # Resize for optimal processing
    current = resize_plate(current)
    results.append(("resized", current.copy()))

    # Reduce noise
    current = reduce_noise(current)
    results.append(("denoised", current.copy()))

    # Enhance contrast
    current = enhance_contrast(current)
    results.append(("enhanced", current.copy()))

    # Sharpen image
    current = sharpen_image(current)
    results.append(("sharpened", current.copy()))

    results.append(("final", current.copy()))
    return results
