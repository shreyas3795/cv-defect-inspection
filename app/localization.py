"""Classical defect localization via background subtraction + contours.

This gives a bounding box around the anomalous region without needing a
trained object detector: blur the image to approximate the "clean" local
background, take the absolute difference from the original, threshold it,
and find contours in the residual. Good enough for single, localized
defects on a fairly uniform background -- the kind of surface-inspection
scenario this dataset simulates.
"""
from __future__ import annotations

import cv2
import numpy as np


def localize_defects(img: np.ndarray, min_area: int = 15) -> list[dict]:
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    background = cv2.GaussianBlur(gray, (15, 15), 0)
    residual = cv2.absdiff(gray, background)
    _, mask = cv2.threshold(residual, 25, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        boxes.append({"x": int(x), "y": int(y), "width": int(w), "height": int(h), "area": float(area)})

    boxes.sort(key=lambda b: -b["area"])
    return boxes
