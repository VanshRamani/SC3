#!/usr/bin/env python3
"""
Run the four analytical / Abraham-style baselines on the v2 splits and
write SDK-style summary.json + per-seed JSONs into results/<method>/.

Methods covered (kept lightweight: no GPU, all CPU):
    abraham_lfer  – per-solvent Abraham 5-feature linear regression
                    (E, S, A, B, V proxies fit per solvent on training rows;
                    falls back to a global model for unseen solvents).
    esol          – Delaney ESOL formula evaluated per solute, then a
                    per-solvent affine (slope+intercept) calibration is
                    learned on the training rows so the prediction is in
                    the v2 LogS = log10(mole_fraction) frame.
    gse           – General Solubility Equation (-logP, simplified, since
                    Tm is not in the dataset) with the same per-solvent
                    affine calibration.
    abraham_ml    – LightGBM on the 16-d Abraham-only feature cache
                    (feature_cache/abraham_only.npz).

WHY A PER-SOLVENT AFFINE CALIBRATION IS NEEDED
----------------------------------------------
The SC3 dataset stores LogS = log10(mole_fraction); the published
analytical formulas predict log10(S in mol/L) for water (ESOL/GSE) or a
log-partition coefficient (Abraham LFER).  Treating either as a direct
predictor of the SC3 target leaves a fixed ~1.5-log offset per solvent
(the log10(rho/MW * 1000) term) PLUS solvent-dependent activity terms.
The fix here: keep the analytical formula as the predictor but absorb
the unit conversion + per-solvent activity term into a per-solvent
affine map fit on the training split.

USAGE
-----
    python scripts/run_analyticals.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Cap thread counts BEFORE importing numpy / lightgbm to play nicely on a shared host.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
SDK_ROOT = HERE.parent
sys.path.insert(0, str(SDK_ROOT))

from sc3_bench.data import load_all_splits, load_cached_features  # noqa: E402
from sc3_bench.evaluate import compute_metrics  # noqa: E402
from sc3_bench.registry import EVAL_SPLITS, DEFAULT_SEEDS  # noqa: E402

RESULTS_DIR = SDK_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Solute descriptor caches (computed once, shared across methods)
# ---------------------------------------------------------------------------

def _abraham_proxy_descriptors(smiles_list: list[str]) -> np.ndarray:
    """Return [N, 5] array of (E, S, A, B, V) Abraham-proxy descriptors.

    Uses the same RDKit-based proxies as sc3-benchmark/abraham.py.  Note
    these are *proxies*, not real Abraham descriptors: E uses MolMR/10
    which is roughly 5-10x larger than the literature E.  Because we
    *fit* the LFER coefficients per solvent below, the absolute scale
    does not matter — the regression absorbs it.
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    _atom_volumes = {
        "C": 16.35, "N": 14.39, "O": 12.43, "F": 10.48, "Cl": 20.95,
        "Br": 26.21, "I": 34.53, "S": 22.91, "P": 24.87, "H": 8.71,
        "Se": 25.10, "B": 18.32, "Si": 26.83,
    }
    _hetero = Chem.MolFromSmarts("[O,N,S,F,Cl,Br,I,P]")
    _arom = Chem.MolFromSmarts("a1aaaaa1")
    _acids = [Chem.MolFromSmarts(s) for s in ["[OH]c", "C(=O)[OH]", "[SH]"]]
    _bases = [Chem.MolFromSmarts(s) for s in ["[NH2,NH1,NH0]", "n", "[CX3]=[OX1]"]]

    cache: dict[str, tuple] = {}
    out = np.zeros((len(smiles_list), 5), dtype=np.float64)
    for i, smi in enumerate(smiles_list):
        if smi in cache:
            out[i] = cache[smi]
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            cache[smi] = (0.0,) * 5
            continue
        E = Descriptors.MolMR(mol) / 10.0
        S = (len(mol.GetSubstructMatches(_hetero)) * 0.2
             + len(mol.GetSubstructMatches(_arom)) * 0.3)
        A = (Descriptors.NumHDonors(mol) * 0.1
             + sum(len(mol.GetSubstructMatches(p)) for p in _acids) * 0.4)
        B = (Descriptors.NumHAcceptors(mol) * 0.1
             + sum(len(mol.GetSubstructMatches(p)) for p in _bases) * 0.3)
        V = sum(_atom_volumes.get(a.GetSymbol(), 15.0) for a in mol.GetAtoms())
        V += sum(a.GetTotalNumHs() for a in mol.GetAtoms()) * 8.71
        V -= 6.56 * mol.GetNumBonds()
        V /= 100.0
        cache[smi] = (E, S, A, B, V)
        out[i] = cache[smi]
    return out


def _solute_logp_mw(smiles_list: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (logP, MW, RotBonds, AromProportion) per solute (ESOL features)."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski

    cache: dict[str, tuple] = {}
    n = len(smiles_list)
    logp = np.zeros(n); mw = np.zeros(n); rb = np.zeros(n); ap = np.zeros(n)
    for i, smi in enumerate(smiles_list):
        if smi not in cache:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                cache[smi] = (0.0, 0.0, 0.0, 0.0)
            else:
                lp = Descriptors.MolLogP(mol)
                m = Descriptors.MolWt(mol)
                r = Lipinski.NumRotatableBonds(mol)
                aromatic = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
                heavy = mol.GetNumHeavyAtoms()
                a = aromatic / heavy if heavy > 0 else 0.0
                cache[smi] = (lp, m, r, a)
        logp[i], mw[i], rb[i], ap[i] = cache[smi]
    return logp, mw, rb, ap


# ---------------------------------------------------------------------------
# Per-solvent affine / linear regression with global-model fallback
# ---------------------------------------------------------------------------

def _per_solvent_linreg(X_train: np.ndarray, y_train: np.ndarray,
                        solv_train: np.ndarray, ridge: float = 1e-2,
                        min_per_solvent: int = 30) -> dict:
    """Fit a (closed-form) ridge regression per solvent name.

    Returns a dict with one entry per solvent
    ``{name: (w, b)}`` where ``preds = X @ w + b``.  Solvents with too
    few training rows fall through to the global model stored under
    the key ``"_global"``.
    """
    out: dict[str, tuple[np.ndarray, float]] = {}
    # Global model first (fallback for unseen / undersampled solvents)
    out["_global"] = _ridge_fit(X_train, y_train, ridge)

    for s in np.unique(solv_train):
        mask = solv_train == s
        if mask.sum() < min_per_solvent:
            continue
        out[s] = _ridge_fit(X_train[mask], y_train[mask], ridge)
    return out


def _ridge_fit(X: np.ndarray, y: np.ndarray, ridge: float) -> tuple[np.ndarray, float]:
    """Closed-form ridge regression with intercept absorbed."""
    n, d = X.shape
    Xc = np.hstack([X, np.ones((n, 1))])  # add intercept column
    A = Xc.T @ Xc
    A[:d, :d] += ridge * np.eye(d)  # don't regularize the intercept
    b = Xc.T @ y
    coef = np.linalg.solve(A, b)
    return coef[:d], float(coef[d])


def _per_solvent_predict(X: np.ndarray, solv: np.ndarray,
                         models: dict) -> np.ndarray:
    """Predict using per-solvent models with global fallback."""
    w_g, b_g = models["_global"]
    out = X @ w_g + b_g
    for s, (w, b) in models.items():
        if s == "_global":
            continue
        m = solv == s
        if m.any():
            out[m] = X[m] @ w + b
    return out


# ---------------------------------------------------------------------------
# Result writers (mirror the SDK's train.py format)
# ---------------------------------------------------------------------------

def _save_method_results(method_key: str, display: str, family: str,
                         seeds: list[int], params: dict,
                         all_seed_metrics: dict[int, dict]) -> None:
    out_dir = RESULTS_DIR / method_key
    out_dir.mkdir(parents=True, exist_ok=True)

    for s in seeds:
        with open(out_dir / f"seed_{s}.json", "w") as f:
            json.dump(all_seed_metrics[s], f, indent=2)

    agg: dict = {}
    for sn in EVAL_SPLITS:
        agg[sn] = {}
        if sn not in all_seed_metrics[seeds[0]]:
            continue
        for mk in all_seed_metrics[seeds[0]][sn]:
            vals = [all_seed_metrics[s][sn][mk] for s in seeds
                    if isinstance(all_seed_metrics[s][sn].get(mk), (int, float))
                    and not (isinstance(all_seed_metrics[s][sn][mk], float)
                             and np.isnan(all_seed_metrics[s][sn][mk]))]
            if vals:
                agg[sn][f"{mk}_mean"] = float(np.mean(vals))
                agg[sn][f"{mk}_std"] = float(np.std(vals))

    summary = {
        "method": method_key, "display": display, "family": family,
        "seeds": seeds, "params": params, "aggregated": agg,
        "per_seed": {str(s): v for s, v in all_seed_metrics.items()},
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  saved -> {out_dir}/summary.json")


# ---------------------------------------------------------------------------
# 1) Abraham LFER (5-param per-solvent linear regression on Abraham proxies)
# ---------------------------------------------------------------------------

def run_abraham_lfer(splits: dict, seeds: list[int]) -> None:
    print("\n" + "=" * 70)
    print("  abraham_lfer")
    print("=" * 70)
    t_total = time.time()

    print("  computing Abraham (E,S,A,B,V) proxies for all unique solutes...")
    t0 = time.time()
    all_solutes = pd.concat([df["Solute"] for df in splits.values()]).unique()
    desc_table = {smi: tuple(d) for smi, d
                  in zip(all_solutes, _abraham_proxy_descriptors(list(all_solutes)))}
    print(f"  done ({time.time()-t0:.1f}s, {len(desc_table)} unique solutes)")

    def _X(df: pd.DataFrame) -> np.ndarray:
        return np.array([desc_table[s] for s in df["Solute"].values], dtype=np.float64)

    train_df = splits["train"]
    X_tr = _X(train_df)
    y_tr = train_df["LogS"].values
    s_tr = train_df["Solvent_Name"].values

    # Deterministic fit; do all seeds in one pass and stamp wall-clock per seed.
    t_fit = time.time()
    models = _per_solvent_linreg(X_tr, y_tr, s_tr,
                                 ridge=1e-2, min_per_solvent=50)
    fit_time = time.time() - t_fit
    print(f"  fit done in {fit_time:.2f}s "
          f"({len(models)-1} per-solvent models + 1 global)")

    all_seed_metrics: dict[int, dict] = {}
    for seed in seeds:
        m_seed: dict = {}
        for sn in EVAL_SPLITS:
            df = splits[sn]
            X = _X(df)
            preds = _per_solvent_predict(
                X, df["Solvent_Name"].values, models)
            metrics = compute_metrics(
                df["LogS"].values, preds,
                df["Solvent_Name"].values if "Solvent_Name" in df.columns else None,
                df["Uncertainty"].values if "Uncertainty" in df.columns else None,
            )
            metrics["train_time_s"] = fit_time
            m_seed[sn] = metrics
        all_seed_metrics[seed] = m_seed

    for sn in EVAL_SPLITS:
        m = all_seed_metrics[seeds[0]][sn]
        ps = m.get("PS_RMSE", float("nan"))
        z = m.get("Z_RMSE", float("nan"))
        print(f"    {sn:11s}  RMSE={m['RMSE']:.4f}  R2={m['R2']:.4f}  "
              f"PS_RMSE={ps:.4f}  Z_RMSE={z:.2f}")
    print(f"  total {time.time()-t_total:.1f}s")

    _save_method_results("abraham_lfer", "Abraham LFER", "Physics",
                         seeds, {"ridge": 1e-2, "min_per_solvent": 50},
                         all_seed_metrics)


# ---------------------------------------------------------------------------
# 2) ESOL (Delaney 1-feature: ESOL_pred -> per-solvent affine)
# ---------------------------------------------------------------------------

def _esol_score(logp: np.ndarray, mw: np.ndarray,
                rb: np.ndarray, ap: np.ndarray) -> np.ndarray:
    """Delaney ESOL log10(S in mol/L for water) — used as a *feature*, not the
    final prediction.  The per-solvent affine layer below corrects it."""
    return 0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * rb - 0.74 * ap


def run_esol(splits: dict, seeds: list[int]) -> None:
    print("\n" + "=" * 70)
    print("  esol")
    print("=" * 70)
    t_total = time.time()

    all_solutes = pd.concat([df["Solute"] for df in splits.values()]).unique()
    print("  computing logP, MW, RB, AP for all unique solutes...")
    t0 = time.time()
    lp, mw, rb, ap = _solute_logp_mw(list(all_solutes))
    esol_pred = _esol_score(lp, mw, rb, ap)
    desc_table = {smi: float(p) for smi, p in zip(all_solutes, esol_pred)}
    print(f"  done ({time.time()-t0:.1f}s)")

    def _X(df: pd.DataFrame) -> np.ndarray:
        return np.array([[desc_table[s]] for s in df["Solute"].values],
                        dtype=np.float64)

    train_df = splits["train"]
    X_tr = _X(train_df)
    y_tr = train_df["LogS"].values
    s_tr = train_df["Solvent_Name"].values

    t_fit = time.time()
    models = _per_solvent_linreg(X_tr, y_tr, s_tr,
                                 ridge=1e-3, min_per_solvent=30)
    fit_time = time.time() - t_fit
    print(f"  fit done in {fit_time:.3f}s "
          f"({len(models)-1} per-solvent models + 1 global)")

    all_seed_metrics: dict[int, dict] = {}
    for seed in seeds:
        m_seed: dict = {}
        for sn in EVAL_SPLITS:
            df = splits[sn]
            preds = _per_solvent_predict(_X(df), df["Solvent_Name"].values, models)
            metrics = compute_metrics(
                df["LogS"].values, preds,
                df["Solvent_Name"].values if "Solvent_Name" in df.columns else None,
                df["Uncertainty"].values if "Uncertainty" in df.columns else None,
            )
            metrics["train_time_s"] = fit_time
            m_seed[sn] = metrics
        all_seed_metrics[seed] = m_seed

    for sn in EVAL_SPLITS:
        m = all_seed_metrics[seeds[0]][sn]
        ps = m.get("PS_RMSE", float("nan"))
        z = m.get("Z_RMSE", float("nan"))
        print(f"    {sn:11s}  RMSE={m['RMSE']:.4f}  R2={m['R2']:.4f}  "
              f"PS_RMSE={ps:.4f}  Z_RMSE={z:.2f}")
    print(f"  total {time.time()-t_total:.1f}s")

    _save_method_results("esol", "ESOL", "Physics", seeds,
                         {"ridge": 1e-3, "min_per_solvent": 30},
                         all_seed_metrics)


# ---------------------------------------------------------------------------
# 3) GSE (logP-only, simplified — no Tm in dataset; per-solvent affine)
# ---------------------------------------------------------------------------

def run_gse(splits: dict, seeds: list[int]) -> None:
    """General Solubility Equation, simplified: log10(S_aq) ≈ 0.5 - logP.

    The full GSE is 0.5 - logP - 0.01*(Tm - 25), but melting point is not
    in the dataset.  We use just -logP as the feature and let the
    per-solvent affine absorb the constant offset and any unit conversion.
    """
    print("\n" + "=" * 70)
    print("  gse")
    print("=" * 70)
    t_total = time.time()

    all_solutes = pd.concat([df["Solute"] for df in splits.values()]).unique()
    print("  computing logP for all unique solutes...")
    t0 = time.time()
    lp, _mw, _rb, _ap = _solute_logp_mw(list(all_solutes))
    gse_pred = 0.5 - lp
    desc_table = {smi: float(p) for smi, p in zip(all_solutes, gse_pred)}
    print(f"  done ({time.time()-t0:.1f}s)")

    def _X(df: pd.DataFrame) -> np.ndarray:
        return np.array([[desc_table[s]] for s in df["Solute"].values],
                        dtype=np.float64)

    train_df = splits["train"]
    X_tr = _X(train_df)
    y_tr = train_df["LogS"].values
    s_tr = train_df["Solvent_Name"].values

    t_fit = time.time()
    models = _per_solvent_linreg(X_tr, y_tr, s_tr,
                                 ridge=1e-3, min_per_solvent=30)
    fit_time = time.time() - t_fit
    print(f"  fit done in {fit_time:.3f}s "
          f"({len(models)-1} per-solvent models + 1 global)")

    all_seed_metrics: dict[int, dict] = {}
    for seed in seeds:
        m_seed: dict = {}
        for sn in EVAL_SPLITS:
            df = splits[sn]
            preds = _per_solvent_predict(_X(df), df["Solvent_Name"].values, models)
            metrics = compute_metrics(
                df["LogS"].values, preds,
                df["Solvent_Name"].values if "Solvent_Name" in df.columns else None,
                df["Uncertainty"].values if "Uncertainty" in df.columns else None,
            )
            metrics["train_time_s"] = fit_time
            m_seed[sn] = metrics
        all_seed_metrics[seed] = m_seed

    for sn in EVAL_SPLITS:
        m = all_seed_metrics[seeds[0]][sn]
        ps = m.get("PS_RMSE", float("nan"))
        z = m.get("Z_RMSE", float("nan"))
        print(f"    {sn:11s}  RMSE={m['RMSE']:.4f}  R2={m['R2']:.4f}  "
              f"PS_RMSE={ps:.4f}  Z_RMSE={z:.2f}")
    print(f"  total {time.time()-t_total:.1f}s")

    _save_method_results("gse", "GSE", "Physics", seeds,
                         {"ridge": 1e-3, "min_per_solvent": 30},
                         all_seed_metrics)


# ---------------------------------------------------------------------------
# 4) Abraham ML — LightGBM on the 16-d abraham_only feature cache
# ---------------------------------------------------------------------------

def run_abraham_ml(splits: dict, seeds: list[int]) -> None:
    print("\n" + "=" * 70)
    print("  abraham_ml  (LightGBM on abraham_only cache)")
    print("=" * 70)
    t_total = time.time()

    cached = load_cached_features("abraham_only")
    if cached is None:
        raise FileNotFoundError(
            "feature_cache/abraham_only.npz not found. "
            "Run `python sc3 cache --featurizers abraham_only` first.")

    X_tr, y_tr = cached["X_train"], cached["y_train"]
    X_ev, y_ev = cached["X_eval"], cached["y_eval"]
    print(f"  abraham_only cache: X_train={X_tr.shape}  X_eval={X_ev.shape}")

    import lightgbm as lgb
    from lightgbm import LGBMRegressor

    # Use the same default tree config as lgb_rdkit/lgb_dissolvr in the SDK,
    # but slightly trimmed (only 16 features so don't over-grow trees).
    params = dict(
        n_estimators=3000, learning_rate=0.05, num_leaves=63,
        min_child_samples=20, feature_fraction=1.0, bagging_fraction=0.8,
        bagging_freq=5, reg_alpha=0.1, reg_lambda=5,
    )

    all_seed_metrics: dict[int, dict] = {}
    for seed in seeds:
        print(f"  --- seed {seed} ---")
        t0 = time.time()
        model = LGBMRegressor(random_state=seed, n_jobs=8, verbose=-1, **params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_ev, y_ev)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        fit_time = time.time() - t0
        bi = getattr(model, "best_iteration_", None)
        print(f"    trained in {fit_time:.1f}s  best_iter={bi}")

        m_seed: dict = {}
        for sn in EVAL_SPLITS:
            X = cached[f"X_{sn}"]
            preds = model.predict(X)
            df = splits[sn]
            metrics = compute_metrics(
                df["LogS"].values, preds,
                df["Solvent_Name"].values if "Solvent_Name" in df.columns else None,
                df["Uncertainty"].values if "Uncertainty" in df.columns else None,
            )
            metrics["train_time_s"] = fit_time
            m_seed[sn] = metrics
            print(f"      {sn:11s}  RMSE={metrics['RMSE']:.4f}  "
                  f"PS_RMSE={metrics.get('PS_RMSE', float('nan')):.4f}")
        all_seed_metrics[seed] = m_seed

    print(f"  total {time.time()-t_total:.1f}s")
    _save_method_results("abraham_ml", "Abraham ML", "Domain",
                         seeds, params, all_seed_metrics)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+",
                        default=["abraham_lfer", "esol", "gse", "abraham_ml"],
                        choices=["abraham_lfer", "esol", "gse", "abraham_ml"])
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    args = parser.parse_args()

    print("Loading v2 splits...")
    splits = load_all_splits(verbose=True)

    runners = {
        "abraham_lfer": run_abraham_lfer,
        "esol": run_esol,
        "gse": run_gse,
        "abraham_ml": run_abraham_ml,
    }
    grand_t0 = time.time()
    for m in args.methods:
        runners[m](splits, args.seeds)
    print(f"\nAll done in {(time.time()-grand_t0):.1f}s.")


if __name__ == "__main__":
    main()
