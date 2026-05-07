"""
Phase 8 (module) — Metric definitions.

Importable by any downstream script that needs to evaluate model predictions
on SC³ splits.  Every metric has a consistent signature:

    metric(y_true, y_pred, *, sigma=None, solvent=None) -> float

where `sigma` and `solvent` are optional per-row arrays; metrics that need
them raise ValueError if unprovided.

Metrics:
  rmse       standard (count-weighted) RMSE  — for baseline comparability
  mae        standard MAE
  medae      median absolute error — robust to heavy-tailed residuals
  ps_rmse    per-solvent mean of RMSE — strips between-solvent inflation
             AND count-weighting bias simultaneously
  z_rmse     (error / sigma)-RMSE on rows with finite sigma — error in units
             of the aleatoric floor; Z = 1 means "model matches the noise"
  mape       mean absolute percentage error — REPORTED AS DIAGNOSTIC ONLY;
             diverges when |y_true| → 0, which happens for logS near zero

Design notes:
  - All metrics return float (not tuple) for composability.
  - `sample_count` / `coverage` wrappers around a metric return (value, n_used).
"""
from __future__ import annotations
import numpy as np


def _as_arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def rmse(y_true, y_pred, **_) -> float:
    yt, yp = _as_arr(y_true), _as_arr(y_pred)
    return float(np.sqrt(np.mean((yp - yt) ** 2)))


def mae(y_true, y_pred, **_) -> float:
    yt, yp = _as_arr(y_true), _as_arr(y_pred)
    return float(np.mean(np.abs(yp - yt)))


def medae(y_true, y_pred, **_) -> float:
    yt, yp = _as_arr(y_true), _as_arr(y_pred)
    return float(np.median(np.abs(yp - yt)))


def ps_rmse(y_true, y_pred, *, solvent, **_) -> float:
    if solvent is None:
        raise ValueError("ps_rmse requires per-row `solvent`")
    yt, yp = _as_arr(y_true), _as_arr(y_pred)
    sv = np.asarray(solvent)
    per: list[float] = []
    for s in np.unique(sv):
        m = sv == s
        if m.sum() == 0:
            continue
        per.append(float(np.sqrt(np.mean((yp[m] - yt[m]) ** 2))))
    return float(np.mean(per))


def z_rmse(y_true, y_pred, *, sigma, **_) -> tuple[float, int]:
    if sigma is None:
        raise ValueError("z_rmse requires per-row `sigma`")
    yt, yp = _as_arr(y_true), _as_arr(y_pred)
    sg = _as_arr(sigma)
    ok = np.isfinite(sg) & (sg > 0)
    if ok.sum() == 0:
        return float("nan"), 0
    z = (yp[ok] - yt[ok]) / sg[ok]
    return float(np.sqrt(np.mean(z ** 2))), int(ok.sum())


def mape(y_true, y_pred, *, logs_eps: float = 1e-6, **_) -> tuple[float, int]:
    """Mean absolute percentage error — DIAGNOSTIC ONLY.
    Excludes rows with |y_true| < logs_eps (divergent)."""
    yt, yp = _as_arr(y_true), _as_arr(y_pred)
    ok = np.abs(yt) >= logs_eps
    if ok.sum() == 0:
        return float("nan"), 0
    return float(np.mean(np.abs((yp[ok] - yt[ok]) / yt[ok])) * 100), int(ok.sum())


# ─── convenience dispatcher ────────────────────────────────────────────────
def evaluate(y_true, y_pred, *, sigma=None, solvent=None,
             include: list[str] | None = None) -> dict[str, float | tuple]:
    """Compute all metrics where inputs allow it.

    include: optional whitelist of metric names.  Defaults to everything
             computable given the inputs provided.
    """
    out: dict = {}
    to_do = include if include is not None else \
        ["rmse", "mae", "medae", "ps_rmse", "z_rmse", "mape"]
    if "rmse"  in to_do: out["rmse"]  = rmse(y_true, y_pred)
    if "mae"   in to_do: out["mae"]   = mae(y_true, y_pred)
    if "medae" in to_do: out["medae"] = medae(y_true, y_pred)
    if "ps_rmse" in to_do and solvent is not None:
        out["ps_rmse"] = ps_rmse(y_true, y_pred, solvent=solvent)
    if "z_rmse"  in to_do and sigma is not None:
        out["z_rmse"] = z_rmse(y_true, y_pred, sigma=sigma)
    if "mape"    in to_do:
        out["mape"] = mape(y_true, y_pred)
    return out
