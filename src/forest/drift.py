"""Monitoreo de deriva: PSI, Kolmogórov-Smirnov con corrección múltiple y Jensen-Shannon."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon


def psi(reference, current, bins: int = 10, eps: float = 1e-6) -> float:
    """Índice de estabilidad poblacional con intervalos por cuantiles de la referencia.

    PSI < 0.1 estable; 0.1–0.25 cambio moderado; > 0.25 cambio significativo.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    p_ref = np.histogram(reference, bins=edges)[0] / len(reference)
    p_cur = np.histogram(current, bins=edges)[0] / len(current)
    p_ref, p_cur = np.clip(p_ref, eps, None), np.clip(p_cur, eps, None)
    return float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref)))


def holm(pvalues) -> np.ndarray:
    """Corrección de Holm-Bonferroni (controla la tasa de error por familia)."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def ks_table(reference: pd.DataFrame, current: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for c in columns:
        r = stats.ks_2samp(reference[c].to_numpy(), current[c].to_numpy())
        rows.append({"feature": c, "ks_stat": r.statistic, "p_value": r.pvalue,
                     "psi": psi(reference[c], current[c])})
    df = pd.DataFrame(rows)
    df["p_holm"] = holm(df["p_value"])
    return df


def class_distribution(labels, classes) -> np.ndarray:
    labels = np.asarray(labels)
    counts = np.array([(labels == c).sum() for c in classes], dtype=float)
    return counts / max(counts.sum(), 1.0)


def js_divergence(p, q) -> float:
    """Divergencia de Jensen-Shannon (base 2, en [0, 1]) entre dos distribuciones."""
    return float(jensenshannon(np.asarray(p) + 1e-12, np.asarray(q) + 1e-12, base=2) ** 2)


class DriftMonitor:
    """Compara cada ventana de peticiones con una referencia y decide si alertar.

    Política de alerta (definida antes de simular):
      - ALERTA si PSI > psi_threshold en alguna variable continua, o
      - si alguna prueba KS es significativa tras Holm al nivel alpha, o
      - si la JS de las clases predichas supera js_threshold.
    """

    def __init__(self, reference: pd.DataFrame, columns: list[str], ref_pred, classes,
                 psi_threshold: float = 0.2, alpha: float = 0.05, js_threshold: float = 0.05):
        self.reference = reference
        self.columns = columns
        self.classes = np.asarray(classes)
        self.ref_dist = class_distribution(ref_pred, self.classes)
        self.psi_threshold = psi_threshold
        self.alpha = alpha
        self.js_threshold = js_threshold

    def check(self, window: pd.DataFrame, pred) -> dict:
        tab = ks_table(self.reference, window, self.columns)
        js = js_divergence(self.ref_dist, class_distribution(pred, self.classes))
        psi_alert = bool((tab["psi"] > self.psi_threshold).any())
        ks_alert = bool((tab["p_holm"] < self.alpha).any())
        js_alert = js > self.js_threshold
        return {
            "max_psi": float(tab["psi"].max()),
            "psi_feature": str(tab["feature"].iloc[int(np.argmax(tab["psi"].to_numpy()))]),
            "min_p_holm": float(tab["p_holm"].min()),
            "js": js,
            "psi_alert": psi_alert,
            "ks_alert": ks_alert,
            "js_alert": js_alert,
            "alert": psi_alert or ks_alert or js_alert,
            "table": tab,
        }


def shift_feature(df: pd.DataFrame, column: str, delta_sd: float, sd: float) -> pd.DataFrame:
    """Desplaza una variable en delta_sd desviaciones estándar (deriva de covariables)."""
    out = df.copy()
    out[column] = out[column] + delta_sd * sd
    return out


def resample_priors(y, weights: dict, n: int, rng) -> np.ndarray:
    """Índices remuestreados con nuevas proporciones de clase (deriva de priors)."""
    y = np.asarray(y)
    classes = np.array(sorted(weights))
    w = np.array([weights[c] for c in classes], dtype=float)
    w = w / w.sum()
    counts = rng.multinomial(n, w)
    idx = [rng.choice(np.flatnonzero(y == c), size=k, replace=True) for c, k in zip(classes, counts, strict=False) if k > 0]
    return rng.permutation(np.concatenate(idx))
