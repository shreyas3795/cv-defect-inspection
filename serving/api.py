"""FastAPI inference service for the surface-defect classifier.

Run locally:
    uvicorn serving.api:app --reload --port 8000

Then:
    curl -X POST -F "file=@data/images/test/scratch/scratch_0001.png" \
        http://localhost:8000/predict
"""
from __future__ import annotations

import io

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.localization import localize_defects
from app.model import DEFAULT_MODEL_PATH, load_model, predict

app = FastAPI(title="Surface Defect Inspection API", version="1.0.0")

_model_bundle = None


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    class_probabilities: dict[str, float]
    defect_regions: list[dict]


@app.on_event("startup")
def _load_model_on_startup() -> None:
    global _model_bundle
    try:
        _model_bundle = load_model(DEFAULT_MODEL_PATH)
    except FileNotFoundError:
        # Model not trained yet; /predict will report a clear error instead
        # of crashing the service on startup.
        _model_bundle = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _model_bundle is not None}


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(file: UploadFile = File(...)) -> PredictionResponse:
    if _model_bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Model not trained yet. Run `python -m app.model data/images` first.",
        )

    raw = await file.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image file.")

    result = predict(img, bundle=_model_bundle)
    regions = localize_defects(img) if result["label"] != "ok" else []

    return PredictionResponse(
        label=result["label"],
        confidence=result["confidence"],
        class_probabilities=result["class_probabilities"],
        defect_regions=regions,
    )
