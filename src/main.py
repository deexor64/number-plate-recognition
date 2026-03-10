import cv2 as cv

from extract_plate import extract_plate
from perform_ocr import perform_ocr
from preprocess_plate import preprocess_plate


def main():
    # Run detection and preprocessing
    print("Log: Begin detection")

    # Extract the plate from the image
    results = extract_plate("images/car-6366999_1920.jpg")

    if results is None:
        return

    cv.imshow("cropped_image", results["cropped_image"])
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Preprocess the extracted plate
    processed = preprocess_plate(results["cropped_image"])

    if processed is None:
        return

    cv.imshow("processed_plate", processed["final"])
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Perform OCR on the processed image
    plate_text = perform_ocr(processed["final"])
    print(f"Plate text: {plate_text}")

    print("Log: Complete!")


if __name__ == "__main__":
    main()
