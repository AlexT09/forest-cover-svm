"""Pruebas unitarias de la biblioteca (validación, modelos, calibración, deriva)."""
import numpy as np
import pandas as pd
import pytest

from src.forest import calibration as CAL
from src.forest import drift as D
from src.forest import models as M
from src.forest import validation as V
from src.forest.features import CONTINUOUS, FEATURES, PhysicalFeatures, load_covtype, sha256
from tests.conftest import toy_frame


def test_physical_features():
    df = toy_frame(20)
    out = PhysicalFeatures().fit(df[FEATURES]).transform(df[FEATURES])
    assert "Aspect" not in out and {"Aspect_sin", "Aspect_cos", "Hydro_Euclid", "Elevation_Adj"} <= set(out)
    assert np.allclose(out.Aspect_sin**2 + out.Aspect_cos**2, 1, atol=1e-5)
    assert list(PhysicalFeatures().get_feature_names_out()) == list(out.columns)
    arr = PhysicalFeatures().transform(df[FEATURES].to_numpy())
    assert arr.shape == out.shape


def test_load_and_hash(tmp_path):
    df = toy_frame(10)
    p = tmp_path / "d.csv"
    df[FEATURES + ["Cover_Type"]].to_csv(p, header=False, index=False)
    back = load_covtype(p)
    assert back.shape == (10, 55)
    assert len(sha256(p)) == 64


def test_blocks_and_splits():
    df = toy_frame(400)
    y = df.Cover_Type.to_numpy()
    g = V.make_blocks(df, ["Elevation", "Slope"], k=10, seed=0)
    assert len(np.unique(g)) == 10
    dev, test = V.holdout_by_blocks(y, g, n_splits=5, seed=0)
    assert not set(g[dev]) & set(g[test])
    sub = V.stratified_subsample(dev, y, 50, seed=0)
    assert len(sub) == 50 and set(sub) <= set(dev)
    assert len(V.stratified_subsample(dev, y, 10**6)) == len(dev)
    for scheme in ["aleatorio", "bloques"]:
        splits = V.scheme_splits(scheme, y, g, n_splits=3, n_repeats=2, seed=0)
        assert len(splits) == 6
    with pytest.raises(ValueError):
        V.scheme_splits("otro", y, g)
    tr, va, te = V.uci_split(20000)
    assert len(tr) == 11340 and len(va) == 3780 and len(te) == 20000 - 15120


def test_run_cv_and_stats():
    df = toy_frame(140)
    y = df.Cover_Type.to_numpy()
    splits = V.scheme_splits("aleatorio", y, None, n_splits=2, n_repeats=1)
    rows = V.run_cv(M.make_logreg(), df[FEATURES], y, splits, "aleatorio", "lr")
    assert len(rows) == 2 and 0 <= rows[0].macro_f1 <= 1
    summ = V.summarize_folds(pd.DataFrame([r.__dict__ for r in rows]), ["model"])
    assert "ci95_half" in summ
    t, p = V.corrected_ttest([0.02, 0.03, 0.01, 0.025], 800, 200)
    assert t > 0 and 0 <= p <= 1
    assert V.cohens_dz([0.1, 0.2, 0.3]) > 0
    r = V.mcnemar([1, 1, 2, 2], [1, 1, 2, 1], [1, 2, 1, 1])
    assert r["n_A_ok_B_fail"] == 2 and 0 <= r["p_value"] <= 1
    ci = V.bootstrap_ci(y, y, np.arange(1, 8), B=20)
    assert np.allclose(ci.estimate, 1)
    d, lo, hi = V.paired_bootstrap_diff(y, y, np.roll(y, 1), np.arange(1, 8), B=20)
    assert d > 0 and lo <= d <= hi


def test_models_build_and_fit():
    df = toy_frame(210)
    X, y = df[FEATURES], df.Cover_Type.to_numpy()
    for mdl in [M.make_svc("rbf"), M.make_svc("linear", ovr=True), M.make_linear_svc(), M.make_hgb(max_iter=5)]:
        assert len(mdl.fit(X, y).predict(X)) == len(y)
    for approx in ["nystroem", "rbf_sampler"]:
        mdl = M.make_kernel_approx(approx, n_components=20, gamma=0.1, n_epochs=2, chunk_size=64, class_weight="balanced")
        mdl.fit(X, y)
        assert mdl.decision_function(X).shape == (len(y), 7)
        assert set(mdl.predict(X)) <= set(range(1, 8))
    with pytest.raises(ValueError):
        M.ChunkedKernelSGD(approximation="x").fit(np.zeros((10, 2)), np.arange(10) % 2)
    with M.measure() as m:
        sum(range(10000))
    assert m["seconds"] >= 0 and m["peak_mb"] >= 0
    a, b, (lo, hi) = M.fit_loglog([1000, 2000, 4000, 8000], [1, 4.1, 15.8, 64.5])
    assert 1.9 < b < 2.1


def test_bootstrap_matches_sklearn_with_absent_class():
    # Una clase de `labels` ausente de y_true y de y_pred no debe contar como F1 = 0
    from sklearn.metrics import balanced_accuracy_score, f1_score
    rng = np.random.default_rng(1)
    yt = rng.choice([1, 2, 3, 5, 6, 7], 500)
    yp = np.where(rng.random(500) < 0.7, yt, rng.choice([1, 2, 3, 5, 6, 7], 500))
    ci = V.bootstrap_ci(yt, yp, np.arange(1, 8), B=50).set_index("metric")
    assert np.isclose(ci.loc["macro_f1", "estimate"], f1_score(yt, yp, average="macro"))
    assert np.isclose(ci.loc["balanced_accuracy", "estimate"], balanced_accuracy_score(yt, yp))
    assert ci.loc["macro_f1", "ci_low"] <= ci.loc["macro_f1", "estimate"] <= ci.loc["macro_f1", "ci_high"]
    assert V._macro_f1_from_cm(np.zeros((3, 3))) == 0.0


def test_hgb_survives_singleton_class():
    # Un pliegue por bloques puede dejar una cobertura rara con un solo ejemplo
    df = toy_frame(210)
    X, y = df[FEATURES], df.Cover_Type.to_numpy().copy()
    y[y == 4] = 1
    y[0] = 4
    mdl = M.make_hgb(max_iter=5).fit(X, y)
    assert mdl.named_steps["clf"].early_stopping_disabled_
    assert mdl.named_steps["clf"].early_stopping is True
    ok = M.make_hgb(max_iter=5).fit(X, df.Cover_Type.to_numpy())
    assert not ok.named_steps["clf"].early_stopping_disabled_


def test_calibration_and_conformal():
    rng = np.random.default_rng(0)
    classes = np.arange(1, 4)
    p = rng.dirichlet([2, 2, 2], 500)
    y = np.array([rng.choice(classes, p=row) for row in p])  # perfectamente calibrado por construcción
    assert CAL.ece(p, y, classes) < 0.1
    assert CAL.classwise_ece(p, y, classes).shape == (3,)
    assert 0 <= CAL.brier_multiclass(p, y, classes) <= 2
    assert {"class", "mean_pred", "frac_pos"} <= set(CAL.reliability_table(p, y, classes).columns)
    cp = CAL.SplitConformal(alpha=0.1).fit(p[:250], y[:250], classes)
    res = cp.evaluate(p[250:], y[250:])
    assert res["coverage"] > 0.8 and 1 <= res["avg_size"] <= 3
    assert cp.predict_sets(p).any(axis=1).all()


def test_drift_detectors():
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({c: rng.normal(0, 1, 2000) for c in CONTINUOUS})
    same = pd.DataFrame({c: rng.normal(0, 1, 500) for c in CONTINUOUS})
    assert D.psi(ref["Elevation"], same["Elevation"]) < 0.1
    shifted = D.shift_feature(same, "Elevation", 1.0, 1.0)
    assert D.psi(ref["Elevation"], shifted["Elevation"]) > 0.25
    assert np.all(np.diff(D.holm([0.01, 0.04, 0.03])[[0, 2, 1]]) >= 0)
    assert D.js_divergence([0.5, 0.5], [0.5, 0.5]) < 1e-9
    preds = rng.integers(1, 8, 2000)
    mon = D.DriftMonitor(ref, CONTINUOUS, preds, np.arange(1, 8))
    assert not mon.check(same, rng.integers(1, 8, 500))["psi_alert"]
    assert mon.check(shifted, rng.integers(1, 8, 500))["alert"]
    idx = D.resample_priors(preds, {c: (5 if c == 1 else 1) for c in range(1, 8)}, 300, rng)
    assert len(idx) == 300 and (preds[idx] == 1).mean() > 0.3
