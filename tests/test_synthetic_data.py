import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.synthetic_data import CLASSES, IMG_SIZE, generate_dataset, generate_image


def test_generate_image_shape_and_dtype():
    for label in CLASSES:
        img = generate_image(label, seed=1)
        assert img.shape == (IMG_SIZE, IMG_SIZE)
        assert img.dtype.name == "uint8"


def test_defect_images_differ_from_ok():
    ok_img = generate_image("ok", seed=7)
    for label in ["scratch", "spot", "crack"]:
        defect_img = generate_image(label, seed=7)
        assert not (ok_img == defect_img).all()


def test_generate_dataset_creates_expected_counts(tmp_path):
    counts = generate_dataset(tmp_path, n_per_class=20, seed=1)
    assert counts["train"] == 14 * len(CLASSES)  # 70% of 20
    assert counts["val"] == 3 * len(CLASSES)  # 15% of 20
    for split in ["train", "val", "test"]:
        for label in CLASSES:
            files = list((tmp_path / split / label).glob("*.png"))
            assert len(files) > 0
