"""Genera un modelo de juguete en una carpeta temporal para que las pruebas no dependan del modelo real.

La variable FOREST_ARTIFACTS se fija ANTES de importar la API (que la lee al importarse), así las
pruebas nunca sobrescriben artifacts/ del proyecto.
"""
import json
import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.forest.features import CONTINUOUS, FEATURES, SOIL, TARGET, WILDERNESS

_TMP = Path(tempfile.mkdtemp(prefix="forest-artifacts-"))
os.environ["FOREST_ARTIFACTS"] = str(_TMP)


def toy_frame(n=300, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.integers(0, 300, (n, len(CONTINUOUS))), columns=CONTINUOUS)
    df["Elevation"] = rng.integers(1900, 3800, n)
    df["Aspect"] = rng.integers(0, 360, n)
    for cols in (WILDERNESS, SOIL):
        hot = rng.integers(0, len(cols), n)
        for j, c in enumerate(cols):
            df[c] = (hot == j).astype(int)
    df[TARGET] = np.tile(np.arange(1, 8), n // 7 + 1)[:n]
    return df


def _build_toy_artifacts():
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression

    from src.forest.models import with_prep

    df = toy_frame()
    model = CalibratedClassifierCV(with_prep(LogisticRegression(max_iter=200)), cv=3, ensemble=False)
    model.fit(df[FEATURES], df[TARGET])
    joblib.dump(model, _TMP / "model.joblib")
    joblib.dump(model, _TMP / "model_alt.joblib")
    (_TMP / "feature_names.json").write_text(json.dumps(FEATURES))
    conf = {"qhat": 0.8, "alpha": 0.1, "n_cal": 100, "method": "sigmoid",
            "mondrian_qhat": {str(c): 0.8 for c in range(1, 8)}}
    (_TMP / "conformal.json").write_text(json.dumps({"model": {**conf, "name": "toy"}, "model_alt": {**conf, "name": "toy2"}}))
    ref = df[FEATURES].copy()
    ref["pred"] = model.predict(df[FEATURES])
    ref.to_csv(_TMP / "reference.csv.gz", index=False)
    df.head(50).to_csv(_TMP / "sample_batch.csv", index=False)
    (_TMP / "results.json").write_text(json.dumps({"champion": "toy", "alternative": "toy2"}))
    (_TMP / "model_card.md").write_text("# Tarjeta de prueba")


_build_toy_artifacts()
