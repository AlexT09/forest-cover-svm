"""Esquemas de validación, bloques pseudo-espaciales y pruebas estadísticas."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import (
    GroupKFold,
    RepeatedStratifiedKFold,
    StratifiedGroupKFold,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler

UCI_TRAIN, UCI_VAL = 11_340, 3_780


def make_blocks(df: pd.DataFrame, columns: list[str], k: int = 20, seed: int = 42) -> np.ndarray:
    """Bloques pseudo-espaciales: KMeans (k) sobre variables estandarizadas.

    El conjunto no trae coordenadas, así que los bloques aproximan la vecindad espacial
    agrupando parcelas con elevación y distancias a vías/incendios parecidas. Los bloques
    son una definición de la estructura de validación, no una entrada del modelo.
    """
    Z = StandardScaler().fit_transform(df[columns].to_numpy(dtype=np.float32))
    km = KMeans(n_clusters=k, n_init=4, random_state=seed)
    return km.fit_predict(Z).astype(np.int16)


def holdout_by_blocks(y: np.ndarray, groups: np.ndarray, n_splits: int = 5, seed: int = 42):
    """Reserva bloques completos como prueba (1 de n_splits pliegues de StratifiedGroupKFold)."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    dev_idx, test_idx = next(sgkf.split(np.zeros(len(y)), y, groups))
    return dev_idx, test_idx


def stratified_subsample(idx: np.ndarray, y: np.ndarray, n: int, seed: int = 42) -> np.ndarray:
    """Submuestra estratificada de tamaño n tomada de los índices idx."""
    if n >= len(idx):
        return np.sort(idx)
    sub, _ = train_test_split(idx, train_size=n, stratify=y[idx], random_state=seed)
    return np.sort(sub)


def uci_split(n_total: int):
    """Partición original de la UCI: 11 340 entrenamiento, 3 780 validación, resto prueba."""
    tr = np.arange(UCI_TRAIN)
    va = np.arange(UCI_TRAIN, UCI_TRAIN + UCI_VAL)
    te = np.arange(UCI_TRAIN + UCI_VAL, n_total)
    return tr, va, te


def scores(y_true, y_pred) -> dict:
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
    }


@dataclass
class FoldResult:
    scheme: str
    model: str
    fold: int
    n_train: int
    n_test: int
    macro_f1: float
    balanced_accuracy: float
    mcc: float
    accuracy: float
    fit_time: float


def run_cv(model, X, y, splits, scheme: str, name: str) -> list[FoldResult]:
    """Evalúa un modelo (pipeline completo, sin fuga) sobre una lista de particiones."""
    rows = []
    for i, (tr, te) in enumerate(splits):
        est = clone(model)
        t0 = time.perf_counter()
        est.fit(X.iloc[tr], y[tr])
        ft = time.perf_counter() - t0
        s = scores(y[te], est.predict(X.iloc[te]))
        rows.append(FoldResult(scheme, name, i, len(tr), len(te), fit_time=ft, **s))
    return rows


def scheme_splits(scheme: str, y, groups, n_splits=5, n_repeats=1, seed=42):
    """Genera particiones para 'aleatorio' (estratificado) o 'bloques' (GroupKFold)."""
    X0 = np.zeros(len(y))
    if scheme == "aleatorio":
        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
        return list(cv.split(X0, y))
    if scheme == "bloques":
        out = []
        for r in range(n_repeats):
            # GroupKFold no baraja: se permutan las etiquetas de bloque para cada repetición.
            rng = np.random.default_rng(seed + r)
            perm = rng.permutation(np.unique(groups))
            remap = dict(zip(np.unique(groups), perm, strict=False))
            g = np.vectorize(remap.get)(groups)
            out += list(GroupKFold(n_splits=n_splits).split(X0, y, g))
        return out
    raise ValueError(scheme)


def corrected_ttest(diffs, n_train: int, n_test: int):
    """t de Nadeau y Bengio (2003) corregida para validación cruzada repetida.

    t = mean(d) / sqrt((1/k + n_test/n_train) * var(d)), con k-1 grados de libertad.
    """
    d = np.asarray(diffs, dtype=float)
    k = len(d)
    var = d.var(ddof=1)
    denom = np.sqrt((1 / k + n_test / n_train) * var)
    t = d.mean() / denom if denom > 0 else np.inf * np.sign(d.mean())
    p = 2 * stats.t.sf(np.abs(t), df=k - 1)
    return float(t), float(p)


def cohens_dz(diffs) -> float:
    """Tamaño del efecto para diferencias pareadas (d_z = media / desviación)."""
    d = np.asarray(diffs, dtype=float)
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else float("inf")


def mcnemar(y_true, pred_a, pred_b):
    """Prueba de McNemar con corrección de continuidad + razón de odds de discordancia."""
    a_ok = np.asarray(pred_a) == np.asarray(y_true)
    b_ok = np.asarray(pred_b) == np.asarray(y_true)
    n01 = int(np.sum(a_ok & ~b_ok))  # A acierta, B falla
    n10 = int(np.sum(~a_ok & b_ok))  # A falla, B acierta
    chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10) if (n01 + n10) > 0 else 0.0
    p = float(stats.chi2.sf(chi2, df=1))
    odds = (n01 + 0.5) / (n10 + 0.5)
    return {"n_A_ok_B_fail": n01, "n_A_fail_B_ok": n10, "chi2": chi2, "p_value": p, "odds_ratio": odds}


def _macro_f1_from_cm(cm: np.ndarray) -> float:
    """Macro-F1 sobre las clases presentes (en y_true o en y_pred), igual que sklearn.f1_score.

    Una cobertura ausente del conjunto evaluado y nunca predicha (p. ej. Cottonwood/Willow en los
    bloques de TEST) no cuenta como F1 = 0: así la estimación coincide con f1_score(average="macro").
    """
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    present = denom > 0
    if not present.any():
        return 0.0
    return float((2 * tp[present] / denom[present]).mean())


def _bal_acc_from_cm(cm: np.ndarray) -> float:
    support = cm.sum(axis=1)
    rec = np.divide(np.diag(cm), support, out=np.zeros(len(cm)), where=support > 0)
    return float(rec[support > 0].mean())


def bootstrap_ci(y_true, y_pred, labels, B: int = 1000, level: float = 0.95, seed: int = 42) -> pd.DataFrame:
    """IC bootstrap (percentil) para macro-F1, balanced accuracy y exactitud.

    Vectorizado vía matrices de confusión: cada réplica es un bincount, no una llamada a sklearn.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    lab_idx = {c: i for i, c in enumerate(labels)}
    yt = np.vectorize(lab_idx.get)(y_true)
    yp = np.vectorize(lab_idx.get)(y_pred)
    K = len(labels)
    code = yt * K + yp
    n = len(code)
    rng = np.random.default_rng(seed)
    out: dict[str, list[float]] = {"macro_f1": [], "balanced_accuracy": [], "accuracy": []}
    for _ in range(B):
        idx = rng.integers(0, n, n)
        cm = np.bincount(code[idx], minlength=K * K).reshape(K, K)
        out["macro_f1"].append(_macro_f1_from_cm(cm))
        out["balanced_accuracy"].append(_bal_acc_from_cm(cm))
        out["accuracy"].append(np.trace(cm) / n)
    cm_full = np.bincount(code, minlength=K * K).reshape(K, K)
    point = {
        "macro_f1": _macro_f1_from_cm(cm_full),
        "balanced_accuracy": _bal_acc_from_cm(cm_full),
        "accuracy": np.trace(cm_full) / n,
    }
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    rows = [
        {"metric": m, "estimate": point[m], "ci_low": np.quantile(v, lo), "ci_high": np.quantile(v, hi)}
        for m, v in out.items()
    ]
    return pd.DataFrame(rows)


def paired_bootstrap_diff(y_true, pred_a, pred_b, labels, B: int = 1000, level: float = 0.95, seed: int = 42):
    """IC bootstrap pareado de la diferencia de macro-F1 (A − B) remuestreando las mismas filas."""
    lab_idx = {c: i for i, c in enumerate(labels)}
    K = len(labels)
    to = np.vectorize(lab_idx.get)
    yt, pa, pb = to(np.asarray(y_true)), to(np.asarray(pred_a)), to(np.asarray(pred_b))
    ca, cb = yt * K + pa, yt * K + pb
    n = len(yt)
    rng = np.random.default_rng(seed)
    diffs = np.empty(B)
    for b in range(B):
        i = rng.integers(0, n, n)
        diffs[b] = (_macro_f1_from_cm(np.bincount(ca[i], minlength=K * K).reshape(K, K))
                    - _macro_f1_from_cm(np.bincount(cb[i], minlength=K * K).reshape(K, K)))
    point = (_macro_f1_from_cm(np.bincount(ca, minlength=K * K).reshape(K, K))
             - _macro_f1_from_cm(np.bincount(cb, minlength=K * K).reshape(K, K)))
    lo, hi = np.quantile(diffs, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(point), float(lo), float(hi)


def summarize_folds(df: pd.DataFrame, by: list[str], metric: str = "macro_f1") -> pd.DataFrame:
    g = df.groupby(by)[metric]
    res = g.agg(["mean", "std", "count"]).reset_index()
    res["ci95_half"] = stats.t.ppf(0.975, res["count"] - 1) * res["std"] / np.sqrt(res["count"])
    return res
