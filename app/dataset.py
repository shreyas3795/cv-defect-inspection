"""Load an image folder dataset laid out as root/<split>/<class>/*.png."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.synthetic_data import CLASSES


def load_split(root: str | Path, split: str) -> tuple[list[np.ndarray], list[str], list[str]]:
    root = Path(root) / split
    images, labels, paths = [], [], []
    for label in CLASSES:
        class_dir = root / label
        if not class_dir.exists():
            continue
        for path in sorted(class_dir.glob("*.png")):
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            images.append(img)
            labels.append(label)
            paths.append(str(path))
    return images, labels, paths
