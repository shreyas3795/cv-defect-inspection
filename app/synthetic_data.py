"""Synthetic surface-inspection image generator.

Real industrial defect datasets (e.g. MVTec AD) can't be redistributed here,
so this module procedurally generates a stand-in "brushed metal surface"
dataset with the same class structure a real inspection line would have:
a clean/OK class and several defect types. Swap `generate_dataset()` for a
loader over a real dataset directory and the rest of the pipeline
(features -> classifier -> evaluation -> serving) is unchanged.
"""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np

IMG_SIZE = 128
CLASSES = ["ok", "scratch", "spot", "crack"]


def _base_texture(rng: np.random.Generator, size: int = IMG_SIZE) -> np.ndarray:
    """Brushed-metal-like base texture: directional low-pass filtered noise."""
    noise = rng.normal(loc=160, scale=18, size=(size, size)).astype(np.float32)
    kernel = np.ones((1, 9), np.float32) / 9  # horizontal brushing
    texture = cv2.filter2D(noise, -1, kernel)
    texture = cv2.GaussianBlur(texture, (3, 3), 0)
    return np.clip(texture, 0, 255).astype(np.uint8)


def _add_scratch(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    img = img.copy()
    n = rng.integers(1, 3)
    for _ in range(n):
        x1, y1 = rng.integers(0, IMG_SIZE, size=2)
        length = rng.integers(30, 90)
        angle = rng.uniform(0, np.pi)
        x2 = int(np.clip(x1 + length * np.cos(angle), 0, IMG_SIZE - 1))
        y2 = int(np.clip(y1 + length * np.sin(angle), 0, IMG_SIZE - 1))
        intensity = int(rng.uniform(40, 90))
        thickness = int(rng.integers(1, 3))
        cv2.line(img, (int(x1), int(y1)), (x2, y2), color=intensity, thickness=thickness)
    return img


def _add_spot(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    img = img.copy()
    n = rng.integers(1, 3)
    for _ in range(n):
        cx, cy = rng.integers(15, IMG_SIZE - 15, size=2)
        radius = int(rng.integers(4, 12))
        intensity = int(rng.uniform(210, 250)) if rng.random() > 0.5 else int(rng.uniform(20, 60))
        cv2.circle(img, (int(cx), int(cy)), radius, color=intensity, thickness=-1)
        img = cv2.GaussianBlur(img, (5, 5), 0)
    return img


def _add_crack(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    img = img.copy()
    x, y = rng.integers(10, IMG_SIZE - 10, size=2)
    points = [(int(x), int(y))]
    for _ in range(rng.integers(5, 9)):
        x = int(np.clip(x + rng.integers(-15, 15), 0, IMG_SIZE - 1))
        y = int(np.clip(y + rng.integers(-15, 15), 0, IMG_SIZE - 1))
        points.append((x, y))
    intensity = int(rng.uniform(20, 60))
    cv2.polylines(img, [np.array(points)], isClosed=False, color=intensity, thickness=2)
    return img


_DEFECT_FNS = {"scratch": _add_scratch, "spot": _add_spot, "crack": _add_crack}


def generate_image(label: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = _base_texture(rng)
    if label != "ok":
        img = _DEFECT_FNS[label](img, rng)
    return img


def generate_dataset(root: str | Path, n_per_class: int = 200, seed: int = 42) -> dict:
    """Generate images under root/<split>/<class>/*.png with a 70/15/15 split."""
    root = Path(root)
    rng = random.Random(seed)
    counts = {"train": 0, "val": 0, "test": 0}

    global_seed = seed * 1000
    for label in CLASSES:
        indices = list(range(n_per_class))
        rng.shuffle(indices)
        n_train = int(0.7 * n_per_class)
        n_val = int(0.15 * n_per_class)
        splits = {
            "train": indices[:n_train],
            "val": indices[n_train : n_train + n_val],
            "test": indices[n_train + n_val :],
        }
        for split, idxs in splits.items():
            out_dir = root / split / label
            out_dir.mkdir(parents=True, exist_ok=True)
            for i in idxs:
                img = generate_image(label, seed=global_seed + hash((label, i)) % 100000)
                cv2.imwrite(str(out_dir / f"{label}_{i:04d}.png"), img)
                counts[split] += 1
    return counts


if __name__ == "__main__":
    import sys

    out = sys.argv[1] if len(sys.argv) > 1 else "data/images"
    result = generate_dataset(out)
    print(f"Generated dataset under {out}: {result}")
