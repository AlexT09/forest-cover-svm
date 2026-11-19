"""Métricas de calibración y predicción conforme dividida."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _onehot(y, classes) -> np.ndarray:
    classes = np.asarray(classes)
    return (np.asarray(y)[:, None] == classes[None, :]).astype(float)


def brier_multiclass(proba, y, classes) -> float:
    """Puntaje de Brier multiclase: media sobre filas de sum_k (p_k - 1[y=k])^2."""
    return float(np.mean(np.sum((proba - _onehot(y, classes)) ** 2, axis=1)))


def ece(proba, y, classes, n_bins: int = 15) -> float:
    """Error de calibración esperado (top-label) con n_bins intervalos de igual ancho."""
    classes = np.asarray(classes)
    conf = proba.max(axis=1)
    correct = classes[proba.argmax(axis=1)] == np.asarray(y)
    edges = np.linspace(0, 1, n_bins + 1)
    b = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    total = 0.0
    for i in range(n_bins):
        m = b == i
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def classwise_ece(proba, y, classes, n_bins: int = 15) -> np.ndarray:
    """ECE uno contra el resto para cada clase."""
    Y = _onehot(y, classes)
    edges = np.linspace(0, 1, n_bins + 1)
    out = []
    for k in range(len(classes)):
        p = proba[:, k]
        b = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
        e = 0.0
        for i in range(n_bins):
            m = b == i
            if m.any():
                e += m.mean() * abs(Y[m, k].mean() - p[m].mean())
        out.append(e)
    return np.array(out)


def reliability_table(proba, y, classes, n_bins: int = 15) -> pd.DataFrame:
    """Tabla para diagramas de fiabilidad por clase (uno contra el resto)."""
    Y = _onehot(y, classes)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for k, c in enumerate(classes):
        p = proba[:, k]
        b = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
        for i in range(n_bins):
            m = b == i
            if m.sum() >= 10:
                rows.append({"class": c, "bin": i, "mean_pred": p[m].mean(),
                             "frac_pos": Y[m, k].mean(), "count": int(m.sum())})
    return pd.DataFrame(rows)


class SplitConformal:
    """Predicción conforme dividida con puntaje s_i = 1 - p(y_i | x_i).

    q̂ es el cuantil de orden ceil((n+1)(1-alpha))/n de los puntajes de calibración y
    C(x) = {k : p(k|x) >= 1 - q̂}. Garantía (marginal): P(y ∈ C(x)) >= 1 - alpha si los
    datos de calibración y de prueba son intercambiables.
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha

    def fit(self, proba_cal, y_cal, classes):
        self.classes_ = np.asarray(classes)
        idx = np.searchsorted(self.classes_, np.asarray(y_cal))
        s = 1.0 - proba_cal[np.arange(len(idx)), idx]
        n = len(s)
        level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        self.qhat_ = float(np.quantile(s, level, method="higher"))
        self.n_cal_ = n
        return self

    def predict_sets(self, proba) -> np.ndarray:
        """Matriz booleana (n, K): True si la clase k está en el conjunto de predicción."""
        sets = proba >= 1.0 - self.qhat_
        # Nunca devolver un conjunto vacío: se incluye al menos la clase más probable.
        sets[np.arange(len(proba)), proba.argmax(axis=1)] = True
        return sets

    def evaluate(self, proba, y) -> dict:
        sets = self.predict_sets(proba)
        idx = np.searchsorted(self.classes_, np.asarray(y))
        covered = sets[np.arange(len(idx)), idx]
        per_class = {
            int(c): {
                "coverage": float(covered[idx == k].mean()) if (idx == k).any() else float("nan"),
                "avg_size": float(sets[idx == k].sum(axis=1).mean()) if (idx == k).any() else float("nan"),
                "n": int((idx == k).sum()),
            }
            for k, c in enumerate(self.classes_)
        }
        return {
            "coverage": float(covered.mean()),
            "avg_size": float(sets.sum(axis=1).mean()),
            "singleton_rate": float((sets.sum(axis=1) == 1).mean()),
            "per_class": per_class,
        }
