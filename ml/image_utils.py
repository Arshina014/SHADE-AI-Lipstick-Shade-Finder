import cv2
import numpy as np

def normalize_lighting(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Normalize brightness channel
    l = cv2.equalizeHist(l)

    lab = cv2.merge((l, a, b))
    normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return normalized

def extract_cheek(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]

    # approximate cheek area
    cx = x + int(0.3 * w)
    cy = y + int(0.6 * h)
    cw = int(0.25 * w)
    ch = int(0.15 * h)

    cheek = img[cy:cy+ch, cx:cx+cw]
    return cheek