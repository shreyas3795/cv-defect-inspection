# Surface Defect Inspection Platform

A CPU-only computer vision pipeline for industrial surface inspection:
classify a surface image as clean or defective (scratch / spot / crack) and
draw a bounding box around the defective region, served behind a FastAPI
endpoint and packaged in a Dockerfile.

## Why classical CV instead of a CNN

Deep learning is a great fit for surface inspection when you have thousands
of labeled real images and a GPU at inference time. This project targets the
other common constraint in industrial settings: a handful of labeled
examples, CPU-only edge hardware on the factory floor, and a need for the
model to be interpretable when it's wrong. So the pipeline uses:

- **HOG (Histogram of Oriented Gradients)** — captures the edge/shape structure of scratches and cracks.
- **Canny edge density + Laplacian variance** — cheap global texture-anomaly signals.
- **Random Forest** — trains in seconds on CPU, gives per-class probabilities, no GPU required to train or serve.

The `HybridIndex`-style separation (`app/features.py` → `app/model.py`) means
swapping in a PyTorch/ONNX CNN later only touches the feature-extraction and
model-loading layers — the dataset, evaluation, API, and Docker packaging
stay the same.

## Dataset

Real industrial defect datasets (e.g. MVTec AD) aren't redistributable here,
so `app/synthetic_data.py` procedurally generates a stand-in "brushed metal"
surface dataset: a clean `ok` class plus `scratch`, `spot`, and `crack`
defects, with the same directory layout (`root/<split>/<class>/*.png`) a real
dataset would use. Point `app/dataset.py` at a real image folder and nothing
else in the pipeline changes.

## Results (this repo, reproducible with the commands below)

800 images (200/class), 70/15/15 train/val/test split, RandomForest (300
trees) on HOG + texture-statistic features:

```
Validation accuracy: 0.8917
Test accuracy:       0.8917

              precision    recall  f1-score   support
crack             0.793     0.767     0.780        30
ok                1.000     1.000     1.000        30
scratch           0.774     0.800     0.787        30
spot              1.000     1.000     1.000        30
```

`ok` and `spot` are separated perfectly; `scratch` vs `crack` is the harder
distinction (both are thin, high-contrast line features) — a realistic
result for a lightweight classical pipeline, and a natural place a CNN would
help most.

## Project layout

```
app/
  synthetic_data.py   # procedural dataset generator
  dataset.py           # image-folder loader
  features.py          # HOG + texture-statistic feature extraction
  localization.py       # background-subtraction bounding-box localization
  model.py              # train / evaluate / predict (RandomForest)
serving/
  api.py                # FastAPI: /health, /predict
tests/                  # pytest: dataset, features, localization, train/predict, API
Dockerfile
```

## Quickstart

```bash
pip install -r requirements.txt

python -m app.synthetic_data data/images     # generate the dataset
python -m app.model data/images              # train + evaluate, saves data/model.joblib

uvicorn serving.api:app --reload --port 8000
curl -X POST -F "file=@data/images/test/scratch/scratch_0001.png" http://localhost:8000/predict
```

Example response:

```json
{
  "label": "scratch",
  "confidence": 0.49,
  "class_probabilities": {"crack": 0.37, "ok": 0.06, "scratch": 0.49, "spot": 0.08},
  "defect_regions": [
    {"x": 16, "y": 83, "width": 60, "height": 22, "area": 156.0}
  ]
}
```

### Docker

```bash
docker build -t defect-inspection .
docker run -p 8000:8000 defect-inspection
```

## Tests

```bash
python -m pytest tests/ -v
# 8 passed: dataset generation, feature extraction, localization,
# train/evaluate/predict, and a small end-to-end training run
```

## Stack

Python, OpenCV, scikit-image (HOG), scikit-learn (RandomForest), FastAPI,
Docker. No GPU or deep learning runtime required.
