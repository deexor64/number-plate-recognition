import pytesseract


def perform_ocr(processed_image):
    # Configuration:
    # psm 7: Treat the image as a single text line (perfect for plates)
    # oem 3: Use the default OCR Engine Mode
    custom_config = r"--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"

    # Run OCR
    text = pytesseract.image_to_string(processed_image, config=custom_config)

    # Clean up text (remove whitespace and special characters)
    return text.strip()
