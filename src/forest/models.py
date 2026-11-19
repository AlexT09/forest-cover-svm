"""Pipelines de modelos, aproximaciones de kernel y medición de costo computacional."""
from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils.multiclass import unique_labels

from .features import make_preprocessor


def with_prep(estimator) -> Pipeline:
    """Todo paso que aprende de los datos vive dentro del Pipeline."""
    return Pipeline([("prep", make_preprocessor()), ("clf", estimator)])


def make_svc(kernel="rbf", C=1.0, gamma="scale", degree=3, class_weight=None, ovr=False, seed=42):
    svc = SVC(
        kernel=kernel, C=C, gamma=gamma, degree=degree, class_weight=class_weight,
        cache_size=500, random_state=seed,
    )
    return with_prep(OneVsRestClassifier(svc) if ovr else svc)


def make_linear_svc(C=1.0, class_weight=None, seed=42):
    return with_prep(LinearSVC(C=C, class_weight=class_weight, dual="auto", max_iter=5000, random_state=seed))


def make_logreg(C=1.0, class_weight=None):
    return with_prep(LogisticRegression(C=C, class_weight=class_weight, max_iter=500))


class SafeHGB(HistGradientBoostingClassifier):
    """HistGradientBoosting cuya parada temprana no falla con clases muy raras.

    La parada temprana separa una validación interna estratificada (``validation_fraction``),
    que exige al menos 2 ejemplos por clase y tantas filas de validación como clases. En la
    validación por bloques una cobertura rara (p. ej. Cottonwood/Willow) puede quedar con
    1 ejemplo en el pliegue de entrenamiento; en ese ajuste se desactiva la parada temprana
    (se entrena ``max_iter`` iteraciones) y queda registrado en ``early_stopping_disabled_``.
    """

    def fit(self, X, y, sample_weight=None):
        _, counts = np.unique(np.asarray(y), return_counts=True)
        frac = self.validation_fraction or 0.0
        n_val = int(np.ceil(frac * len(y)))
        infeasible = counts.min() < 2 or n_val < len(counts) or len(y) - n_val < len(counts)
        original = self.early_stopping
        self.early_stopping_disabled_ = bool(original is True and infeasible)
        if self.early_stopping_disabled_:
            self.early_stopping = False
        try:
            return super().fit(X, y, sample_weight=sample_weight)
        finally:
            self.early_stopping = original


def make_hgb(seed=42, **params):
    defaults = dict(
        max_iter=300, learning_rate=0.1, max_leaf_nodes=31, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=seed,
    )
    defaults.update(params)
    return with_prep(SafeHGB(**defaults))


class ChunkedKernelSGD(ClassifierMixin, BaseEstimator):
    """Aproximación de kernel RBF + clasificador lineal entrenado por bloques (partial_fit).

    Permite usar TODOS los datos de desarrollo sin materializar la matriz n x m de
    características aproximadas (que con m=1000 y n≈4·10^5 ocuparía ~3 GB): cada bloque se
    transforma y se pasa a SGDClassifier(loss="hinge"), es decir, una SVM lineal en el
    espacio de características aproximado.
    """

    def __init__(self, approximation="nystroem", n_components=300, gamma=0.05, alpha=1e-5,
                 n_epochs=5, chunk_size=20000, class_weight=None, random_state=42):
        self.approximation = approximation
        self.n_components = n_components
        self.gamma = gamma
        self.alpha = alpha
        self.n_epochs = n_epochs
        self.chunk_size = chunk_size
        self.class_weight = class_weight
        self.random_state = random_state

    def _make_map(self):
        if self.approximation == "nystroem":
            return Nystroem(kernel="rbf", gamma=self.gamma, n_components=self.n_components,
                            random_state=self.random_state)
        if self.approximation == "rbf_sampler":
            return RBFSampler(gamma=self.gamma, n_components=self.n_components,
                              random_state=self.random_state)
        raise ValueError(self.approximation)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        rng = np.random.default_rng(self.random_state)
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]
        # Nystroem elige m puntos de referencia: basta ajustarlo sobre una submuestra.
        fit_idx = rng.choice(len(X), size=min(len(X), max(5 * self.n_components, 5000)), replace=False)
        self.map_ = self._make_map().fit(X[fit_idx])
        cw = None
        if self.class_weight == "balanced":
            w = compute_class_weight("balanced", classes=self.classes_, y=y)
            cw = dict(zip(self.classes_, w, strict=False))
        self.sgd_ = SGDClassifier(loss="hinge", alpha=self.alpha, class_weight=cw,
                                  learning_rate="optimal", random_state=self.random_state)
        n = len(X)
        for _ in range(self.n_epochs):
            order = rng.permutation(n)
            for start in range(0, n, self.chunk_size):
                idx = order[start:start + self.chunk_size]
                self.sgd_.partial_fit(self.map_.transform(X[idx]), y[idx], classes=self.classes_)
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=np.float32)
        parts = [self.sgd_.decision_function(self.map_.transform(X[s:s + self.chunk_size]))
                 for s in range(0, len(X), self.chunk_size)]
        return np.vstack(parts)

    def predict(self, X):
        return self.classes_[self.decision_function(X).argmax(axis=1)]


def make_kernel_approx(approximation="nystroem", n_components=300, gamma=0.05, alpha=1e-5,
                       n_epochs=5, chunk_size=20000, class_weight=None, seed=42):
    return with_prep(ChunkedKernelSGD(approximation, n_components, gamma, alpha, n_epochs,
                                      chunk_size, class_weight, seed))


# --------------------------------------------------------------------------- medición
def _rss_mb() -> float:
    import psutil
    return psutil.Process(os.getpid()).memory_info().rss / 2**20


@contextmanager
def measure():
    """Mide tiempo de pared y pico de memoria residente (MB) por encima de la línea base.

    La memoria se muestrea cada 20 ms en un hilo aparte, porque libsvm reserva memoria
    en C y no es visible para tracemalloc.
    """
    res = {"seconds": 0.0, "peak_mb": 0.0}
    base = _rss_mb()
    peak = [base]
    stop = threading.Event()

    def poll():
        while not stop.is_set():
            peak[0] = max(peak[0], _rss_mb())
            time.sleep(0.02)

    th = threading.Thread(target=poll, daemon=True)
    th.start()
    t0 = time.perf_counter()
    try:
        yield res
    finally:
        res["seconds"] = time.perf_counter() - t0
        stop.set()
        th.join()
        res["peak_mb"] = max(0.0, peak[0] - base)


def fit_loglog(n, t):
    """Ajusta log t = a + b log n por MCO y devuelve (a, b, IC95 de b)."""
    import statsmodels.api as sm
    x = np.log(np.asarray(n, dtype=float))
    y = np.log(np.asarray(t, dtype=float))
    res = sm.OLS(y, sm.add_constant(x)).fit()
    a, b = res.params
    lo, hi = res.conf_int(0.05)[1] if len(x) > 2 else (np.nan, np.nan)
    return float(a), float(b), (float(lo), float(hi))
