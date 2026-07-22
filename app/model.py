"""Train/evaluate/persist the defect classifier."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

from app.dataset import load_split
from app.features import extract_batch
from app.synthetic_data import CLASSES

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "model.joblib"


def train(data_root: str | Path, model_path: str | Path = DEFAULT_MODEL_PATH) -> dict:
    train_imgs, train_labels, _ = load_split(data_root, "train")
    val_imgs, val_labels, _ = load_split(data_root, "val")

    X_train = extract_batch(train_imgs)
    X_val = extract_batch(val_imgs)

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)

    clf = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)
    clf.fit(X_train_s, train_labels)

    val_preds = clf.predict(X_val_s)
    report = classification_report(val_labels, val_preds, output_dict=True, zero_division=0)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler": scaler, "classifier": clf, "classes": CLASSES}, model_path)

    return {"val_accuracy": report["accuracy"], "val_report": report}


def evaluate(data_root: str | Path, model_path: str | Path = DEFAULT_MODEL_PATH, split: str = "test") -> dict:
    bundle = joblib.load(model_path)
    scaler, clf = bundle["scaler"], bundle["classifier"]

    imgs, labels, paths = load_split(data_root, split)
    X = scaler.transform(extract_batch(imgs))
    preds = clf.predict(X)

    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(labels, preds, labels=CLASSES)

    return {
        "split": split,
        "accuracy": report["accuracy"],
        "report": report,
        "confusion_matrix": cm.tolist(),
        "labels_order": CLASSES,
        "n_samples": len(labels),
    }


def load_model(model_path: str | Path = DEFAULT_MODEL_PATH):
    return joblib.load(model_path)


def predict(img: np.ndarray, bundle=None, model_path: str | Path = DEFAULT_MODEL_PATH) -> dict:
    bundle = bundle or load_model(model_path)
    scaler, clf = bundle["scaler"], bundle["classifier"]

    feats = extract_batch([img])
    feats_s = scaler.transform(feats)
    pred = clf.predict(feats_s)[0]
    proba = clf.predict_proba(feats_s)[0]
    class_probs = dict(zip(clf.classes_, proba.tolist()))

    return {"label": pred, "confidence": max(class_probs.values()), "class_probabilities": class_probs}


if __name__ == "__main__":
    import sys

    data_root = sys.argv[1] if len(sys.argv) > 1 else "data/images"
    metrics = train(data_root)
    print(f"Validation accuracy: {metrics['val_accuracy']:.4f}")

    test_metrics = evaluate(data_root)
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(json.dumps(test_metrics["report"], indent=2))
