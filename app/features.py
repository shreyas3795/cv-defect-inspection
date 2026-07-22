"""Feature extraction for the defect classifier.

Combines HOG (shape/edge structure), simple intensity statistics, and Canny
edge density into a single fixed-length feature vector. This is a classical,
CPU-only computer vision pipeline -- no GPU or deep learning runtime
required to train or serve, which matters for edge/on-prem inspection
deployments.
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import hog

from app.synthetic_data import IMG_SIZE


def extract_features(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    hog_features = hog(
        img,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )

    mean_intensity = img.mean()
    std_intensity = img.std()

    edges = cv2.Canny(img, 60, 150)
    edge_density = edges.mean() / 255.0

    laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()

    stats = np.array([mean_intensity, std_intensity, edge_density, laplacian_var])
    return np.concatenate([hog_features, stats])


def extract_batch(images: list[np.ndarray]) -> np.ndarray:
    return np.stack([extract_features(img) for img in images])
