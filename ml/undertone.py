import cv2
import numpy as np

def detect_undertone(cheek_path):
    # Read cheek image
    img = cv2.imread(cheek_path)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Calculate average HSV values
    h, s, v = cv2.split(hsv)
    avg_h = np.mean(h)
    avg_s = np.mean(s)

    # Simple undertone rules
    if avg_h < 15 or avg_h > 160:
        undertone = "Warm"
    elif 15 <= avg_h <= 35:
        undertone = "Neutral"
    else:
        undertone = "Cool"

    return undertone, round(avg_h, 2), round(avg_s, 2)