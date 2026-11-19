"""Servicio de inferencia (FastAPI) con /health y /predict.

Carga el modelo una sola vez al arrancar, valida el esquema de entrada (54 valores por fila,
máximo 1 000 filas) y devuelve la clase, las probabilidades calibradas y, si existe
artifacts/conformal.json, el conjunto de predicción conforme.
"""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

ARTIFACTS = Path(os.environ.get("FOREST_ARTIFACTS", "artifacts"))
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = joblib.load(ARTIFACTS / "model.joblib")
    state["features"] = json.loads((ARTIFACTS / "feature_names.json").read_text(encoding="utf-8"))
    conf = ARTIFACTS / "conformal.json"
    state["qhat"] = json.loads(conf.read_text(encoding="utf-8")).get("model") if conf.exists() else None
    yield
    state.clear()


app = FastAPI(title="Forest cover API", version="1.0.0", lifespan=lifespan)


class Batch(BaseModel):
    instances: list[list[float]] = Field(min_length=1, max_length=1000)

    @field_validator("instances")
    @classmethod
    def check_width(cls, rows):
        if any(len(r) != 54 for r in rows):
            raise ValueError("Each instance must contain exactly 54 values")
        return rows


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(batch: Batch):
    X = pd.DataFrame(batch.instances, columns=state["features"])
    model = state["model"]
    proba = model.predict_proba(X)
    pred = model.classes_[proba.argmax(axis=1)]
    out = {
        "cover_type": pred.tolist(),
        "probabilities": proba.round(4).tolist(),
        "classes": model.classes_.tolist(),
    }
    qhat = state.get("qhat")
    if qhat is not None:
        sets = proba >= 1.0 - qhat["qhat"]
        sets[np.arange(len(proba)), proba.argmax(axis=1)] = True
        out["prediction_sets"] = [model.classes_[s].tolist() for s in sets]
        out["conformal_alpha"] = qhat["alpha"]
    return out
