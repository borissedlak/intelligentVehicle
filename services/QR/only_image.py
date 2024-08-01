import os

import cv2
from pyzbar.pyzbar import decode

import utils

ROOT = os.path.dirname(__file__)


# Function to detect and highlight QR code in an image
def highlight_qr_code(image_path, output_path):
    # Read the image
    frame = cv2.resize(cv2.imread(ROOT + image_path), (1920, 1080))
    if frame is None:
        print(f"Error: Unable to open image file: {image_path}")
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    decoded_objects = decode(gray)
    combined_img = utils.highlight_qr_codes(frame, decoded_objects)

    # cv2.imshow("Detected Objects", combined_img)
    cv2.imwrite(ROOT + output_path, combined_img)


highlight_qr_code(image_path="/data/car_1_raw.jpg", output_path="/data/QR_Demo.jpg")
# highlight_qr_code("/data/car_2.jpg")
