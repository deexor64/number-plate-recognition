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


def blur_image(image, level=7):
    blurred = cv.GaussianBlur(image, (level, level), 0)
    return blurred


def binary_image(image):
    _, binaried = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return binaried


def erod_image(image, kernel_size=2, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded = cv.erode(image, kernel, iterations=iterations)

    return eroded


def thin_image(image, kernel_size=2, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # Dilation expands white areas, effectively thinning black text
    thinner = cv.dilate(image, kernel, iterations=iterations)
    return thinner


def adjust_brightness(image, alpha=0.7):
    darkened = cv.convertScaleAbs(image, alpha=alpha, beta=0)
    return darkened


# Apply complete preprocessing pipeline to a license plate image
def preprocess_plate(plate_image):
    if plate_image is None or plate_image.size == 0:
        return None

    results = [("cropped", plate_image)]

    current = resize_plate(plate_image)
    results.append(("resized", current.copy()))

    current = gray_scale(current)
    results.append(("gray", current.copy()))

    current = reduce_noise(current)
    results.append(("denoised", current.copy()))

    current = erod_image(current, kernel_size=3, iterations=2)
    results.append(("eroded", current.copy()))

    # current = blur_image(current, level=15)
    # results.append(("blurred", current.copy()))

    # current = adjust_brightness(current, 1.5)
    # results.append(("brighted", current.copy()))

    current = binary_image(current)
    results.append(("binary", current.copy()))

    # current = thin_image(current, kernel_size=2, iterations=1)
    # results.append(("thinned", current.copy()))

    results.append(("final", current.copy()))
    return results
