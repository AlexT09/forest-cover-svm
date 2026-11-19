"""Carga de datos e ingeniería de características con sentido físico."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CONTINUOUS = [
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
]
WILDERNESS = [f"Wilderness_Area{i}" for i in range(1, 5)]
SOIL = [f"Soil_Type{i}" for i in range(1, 41)]
FEATURES = CONTINUOUS + WILDERNESS + SOIL
TARGET = "Cover_Type"

CLASS_NAMES = {
    1: "Spruce/Fir",
    2: "Lodgepole Pine",
    3: "Ponderosa Pine",
    4: "Cottonwood/Willow",
    5: "Aspen",
    6: "Douglas-fir",
    7: "Krummholz",
}

# Variables continuas tras la ingeniería: la orientación cruda se sustituye por su codificación circular.
ENGINEERED = ["Aspect_sin", "Aspect_cos", "Hydro_Euclid", "Elevation_Adj"]
SCALED = [c for c in CONTINUOUS if c != "Aspect"] + ENGINEERED

# Familias de variables (útiles para interpretar y para el monitoreo).
FAMILIES = {
    "topografía": ["Elevation", "Aspect", "Slope"],
    "hidrología": ["Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology"],
    "infraestructura": ["Horizontal_Distance_To_Roadways", "Horizontal_Distance_To_Fire_Points"],
    "iluminación": ["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"],
    "áreas silvestres": WILDERNESS,
    "suelos": SOIL,
}


def sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    """Suma SHA-256 de un archivo (versionado ligero de los datos)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def load_covtype(path: str | Path) -> pd.DataFrame:
    """Lee covtype.data(.gz) con tipos compactos (int32/int8) para ahorrar memoria."""
    dtypes: dict[str, type] = {c: np.int32 for c in CONTINUOUS}
    dtypes.update({c: np.int8 for c in WILDERNESS + SOIL})
    dtypes[TARGET] = np.int8
    return pd.read_csv(path, header=None, names=FEATURES + [TARGET], dtype=dtypes)


class PhysicalFeatures(TransformerMixin, BaseEstimator):
    """Añade variables con sentido físico. No aprende nada de los datos (sin estado).

    - Aspect_sin / Aspect_cos: la orientación es circular (0° = 360°).
    - Hydro_Euclid: distancia euclidiana a la hidrología (componentes horizontal y vertical).
    - Elevation_Adj: elevación menos la distancia vertical a la hidrología.
    """

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=FEATURES) if not isinstance(X, pd.DataFrame) else X
        out = X.drop(columns=["Aspect"]).astype(np.float32)
        rad = np.deg2rad(X["Aspect"].to_numpy(dtype=np.float32))
        out["Aspect_sin"] = np.sin(rad)
        out["Aspect_cos"] = np.cos(rad)
        h = X["Horizontal_Distance_To_Hydrology"].to_numpy(dtype=np.float32)
        v = X["Vertical_Distance_To_Hydrology"].to_numpy(dtype=np.float32)
        out["Hydro_Euclid"] = np.sqrt(h**2 + v**2)
        out["Elevation_Adj"] = X["Elevation"].to_numpy(dtype=np.float32) - v
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array([c for c in FEATURES if c != "Aspect"] + ENGINEERED, dtype=object)


def make_preprocessor() -> Pipeline:
    """Ingeniería + escalado SOLO de las continuas; las indicadoras pasan sin escalar."""
    scaler = ColumnTransformer(
        [("num", StandardScaler(), SCALED)],
        remainder="passthrough",
        verbose_feature_names_out=False,
        sparse_threshold=0.0,
    )
    return Pipeline([("physical", PhysicalFeatures()), ("scale", scaler)])
