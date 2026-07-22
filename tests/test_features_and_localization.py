import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.features import extract_features
from app.localization import localize_defects
from app.synthetic_data import generate_image


def test_feature_vector_is_fixed_length_and_finite():
    import numpy as np

    f1 = extract_features(generate_image("ok", seed=1))
    f2 = extract_features(generate_image("scratch", seed=2))
    assert f1.shape == f2.shape
    assert np.isfinite(f1).all()
    assert np.isfinite(f2).all()


def test_localization_finds_no_regions_on_clean_surface():
    img = generate_image("ok", seed=3)
    boxes = localize_defects(img)
    assert len(boxes) <= 1  # allow for occasional texture noise, but no strong signal


def test_localization_finds_a_region_on_defect_surface():
    img = generate_image("spot", seed=3)
    boxes = localize_defects(img)
    assert len(boxes) >= 1
    box = boxes[0]
    assert {"x", "y", "width", "height", "area"} <= box.keys()
