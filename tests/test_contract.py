import numpy as np
from fastapi.testclient import TestClient
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.forest.api import app
from src.forest.features import FEATURES, SCALED, make_preprocessor
from tests.conftest import toy_frame


def test_scaler_uses_only_training_data():
    """Prueba de no fuga: el escalador aprende solo del entrenamiento."""
    rng = np.random.default_rng(0)
    X_tr, X_te = rng.normal(0, 1, (100, 3)), rng.normal(10, 1, (50, 3))
    y_tr = np.tile([0, 1], 50)
    pipe = Pipeline([("scaler", StandardScaler()), ("svc", SVC())]).fit(X_tr, y_tr)
    pipe.predict(X_te)
    assert np.allclose(pipe.named_steps["scaler"].mean_, X_tr.mean(axis=0))


def test_project_preprocessor_fits_only_on_training_rows():
    """La misma propiedad sobre el preprocesador real del proyecto."""
    df = toy_frame(200)
    tr, te = df.iloc[:100], df.iloc[100:].copy()
    te["Elevation"] += 5000  # un cambio enorme en la prueba no debe alterar lo aprendido
    prep = make_preprocessor().fit(tr[FEATURES])
    before = prep.named_steps["scale"].named_transformers_["num"].mean_.copy()
    prep.transform(te[FEATURES])
    after = prep.named_steps["scale"].named_transformers_["num"].mean_
    assert np.allclose(before, after)
    assert np.isclose(before[SCALED.index("Elevation")], tr["Elevation"].mean())


def test_rejects_wrong_width():
    with TestClient(app) as client:
        r = client.post("/predict", json={"instances": [[1.0, 2.0, 3.0]]})
        assert r.status_code == 422


def test_rejects_too_many_rows():
    with TestClient(app) as client:
        r = client.post("/predict", json={"instances": [[0.0] * 54] * 1001})
        assert r.status_code == 422


def test_health_and_predict_contract():
    df = toy_frame(5)
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        r = client.post("/predict", json={"instances": df[FEATURES].astype(float).values.tolist()})
        assert r.status_code == 200
        body = r.json()
        assert len(body["cover_type"]) == 5
        assert len(body["probabilities"][0]) == len(body["classes"]) == 7
        assert np.allclose(np.sum(body["probabilities"], axis=1), 1, atol=1e-3)
        assert all(p in s for p, s in zip(body["cover_type"], body["prediction_sets"], strict=False))
