import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.model import evaluate, predict, train
from app.synthetic_data import generate_dataset, generate_image


def test_train_and_evaluate_small_dataset(tmp_path):
    data_root = tmp_path / "images"
    model_path = tmp_path / "model.joblib"

    generate_dataset(data_root, n_per_class=40, seed=123)
    train_metrics = train(data_root, model_path=model_path)
    assert train_metrics["val_accuracy"] > 0.6  # sanity floor on a small/fast run

    test_metrics = evaluate(data_root, model_path=model_path, split="test")
    assert test_metrics["n_samples"] > 0
    assert test_metrics["accuracy"] > 0.6


def test_predict_returns_valid_schema(tmp_path):
    data_root = tmp_path / "images"
    model_path = tmp_path / "model.joblib"

    generate_dataset(data_root, n_per_class=40, seed=123)
    train(data_root, model_path=model_path)

    img = generate_image("scratch", seed=999)
    result = predict(img, model_path=model_path)

    assert result["label"] in {"ok", "scratch", "spot", "crack"}
    assert 0.0 <= result["confidence"] <= 1.0
    assert abs(sum(result["class_probabilities"].values()) - 1.0) < 1e-6
