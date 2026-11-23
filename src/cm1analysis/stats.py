"""Descriptive statistics and effect sizes for comparing two samples.

Used by :mod:`cm1analysis.plots.statistics` to compare a run's wind swath or
cold pool with a reference run's, but plain functions of numpy arrays so they
can be tested and reused on their own.

Sign convention: every effect of ``a`` against ``b`` is positive when ``a``
is larger. For a cold pool (theta' < 0) a negative effect therefore means a
colder, stronger pool than the reference.

Grid points of a swath are strongly autocorrelated in space, so the effective
sample size is far below the number of points, and p-values from t- or
Kolmogorov-Smirnov tests are not meaningful. Only descriptive measures are
provided: Cohen's d, Cliff's delta, quantile differences and the K-S statistic
D itself (no p-value).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

MAD_TO_SIGMA = 1.4826  # scales the median absolute deviation to sigma for normal data

# |Cliff's delta| thresholds of Romano et al. (2006)
CLIFFS_THRESHOLDS = ((0.147, "negligible"), (0.33, "small"), (0.474, "medium"))
# |Cohen's d| thresholds of Cohen (1988)
COHENS_THRESHOLDS = ((0.2, "negligible"), (0.5, "small"), (0.8, "medium"))


def _finite(x) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    return x[np.isfinite(x)]


def robust_scale(x) -> float:
    """MAD estimate of the standard deviation: 1.4826 * median(|x - median(x)|)."""
    x = _finite(x)
    if x.size == 0:
        return float("nan")
    return float(MAD_TO_SIGMA * np.median(np.abs(x - np.median(x))))


def pooled_std(a, b) -> float:
    """Pooled sample standard deviation of two samples (ddof = 1 in each)."""
    a, b = _finite(a), _finite(b)
    na, nb = a.size, b.size
    if na < 2 or nb < 2:
        return float("nan")
    var = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    return float(np.sqrt(var))


def cohens_d(a, b) -> float:
    """Cohen's d of a against b: (mean(a) - mean(b)) / pooled SD; NaN if the SD is zero."""
    s = pooled_std(a, b)
    if not np.isfinite(s) or s == 0:
        return float("nan")
    return float((_finite(a).mean() - _finite(b).mean()) / s)


def cliffs_delta(a, b) -> float:
    """Cliff's (1993) delta of a against b, in [-1, 1].

    delta = [#(a_i > b_j) - #(a_i < b_j)] / (n_a n_b), computed in
    O((n_a + n_b) log n_b) by sorting b. Positive means a tends to exceed b.
    """
    a, b = _finite(a), np.sort(_finite(b))
    if a.size == 0 or b.size == 0:
        return float("nan")
    below = np.searchsorted(b, a, side="left")  # b_j < a_i
    above = b.size - np.searchsorted(b, a, side="right")  # b_j > a_i
    return float(np.sum(below - above, dtype=np.int64) / (a.size * b.size))


def ks_statistic(a, b) -> float:
    """Two-sample Kolmogorov-Smirnov statistic D = max |F_a - F_b| (no p-value, see module notes)."""
    a, b = np.sort(_finite(a)), np.sort(_finite(b))
    if a.size == 0 or b.size == 0:
        return float("nan")
    grid = np.concatenate([a, b])
    fa = np.searchsorted(a, grid, side="right") / a.size
    fb = np.searchsorted(b, grid, side="right") / b.size
    return float(np.max(np.abs(fa - fb)))


def _label(value: float, thresholds) -> str:
    if not np.isfinite(value):
        return ""
    for limit, name in thresholds:
        if abs(value) < limit:
            return name
    return "large"


def cliffs_label(delta: float) -> str:
    """negligible / small / medium / large for |Cliff's delta| (Romano et al. 2006)."""
    return _label(delta, CLIFFS_THRESHOLDS)


def cohens_label(d: float) -> str:
    """negligible / small / medium / large for |Cohen's d| (Cohen 1988)."""
    return _label(d, COHENS_THRESHOLDS)


@dataclass(frozen=True)
class QuantileEffect:
    """One percentile p of a run against a reference.

    ``diff`` = Q_p(run) - Q_p(ref); ``sqd`` is diff over the reference's robust
    scale (the standardized quantile difference); ``ratio`` = Q_p(run) / Q_p(ref).
    """

    percentile: float
    q_run: float
    q_ref: float
    diff: float
    sqd: float
    ratio: float


def quantile_effects(run, ref, percentiles: Iterable[float]) -> dict[float, QuantileEffect]:
    """Quantile differences, standardized differences and ratios at each percentile."""
    run, ref = _finite(run), _finite(ref)
    percentiles = [float(p) for p in percentiles]
    if run.size == 0 or ref.size == 0:
        nan = float("nan")
        return {p: QuantileEffect(p, nan, nan, nan, nan, nan) for p in percentiles}
    scale = robust_scale(ref)
    q_run = np.percentile(run, percentiles)
    q_ref = np.percentile(ref, percentiles)
    out = {}
    for p, qr, qc in zip(percentiles, q_run, q_ref, strict=True):
        diff = float(qr - qc)
        out[p] = QuantileEffect(
            p,
            float(qr),
            float(qc),
            diff,
            diff / scale if scale > 0 else float("nan"),
            float(qr / qc) if qc != 0 else float("nan"),
        )
    return out


def describe(x) -> dict[str, float]:
    """n, mean, median, std (ddof = 1), IQR, robust scale and the 5/10/90/95th percentiles, min, max."""
    x = _finite(x)
    keys = ("n", "mean", "median", "std", "iqr", "robust_scale", "p05", "p10", "p90", "p95", "min", "max")
    if x.size == 0:
        return dict.fromkeys(keys, float("nan")) | {"n": 0}
    p05, p10, p25, p75, p90, p95 = np.percentile(x, [5, 10, 25, 75, 90, 95])
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "std": float(x.std(ddof=1)) if x.size > 1 else float("nan"),
        "iqr": float(p75 - p25),
        "robust_scale": robust_scale(x),
        "p05": float(p05),
        "p10": float(p10),
        "p90": float(p90),
        "p95": float(p95),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def pearson_r(a, b) -> float:
    """Pearson correlation over points where both are finite; NaN if either is constant."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    a, b = a[ok] - a[ok].mean(), b[ok] - b[ok].mean()
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return float(np.sum(a * b) / denom) if denom > 0 else float("nan")
