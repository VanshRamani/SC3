"""
Phase 3B sensitivity — sweep the copycat threshold θ_B and measure how
  (a) ε_A primary (HoS-excluded, interpolated),
  (b) P90 of pair MAE,
  (c) number of multi-source pairs surviving,
  (d) number of Hall-of-Shame groups,
depend on θ_B.

The method is identical to the main pipeline but with θ_B parameterised.
We reconstruct Stage-A preliminary groups on the fly (bit-exact merging
is threshold-free), fit Apelblat/van't-Hoff per preliminary group, then
for every θ_B in the sweep grid:

  1. Union-find on preliminary groups using interpolated weighted MAE < θ_B.
  2. Compute interpolated Stage C' reliability.
  3. Flag Hall of Shame at the fixed policy threshold 0.6 log S.
  4. Compute ε_A primary (mean of per-pair MAE, HoS-excluded).

Output:
  reports/55_threshold_sweep.csv   one row per θ_B
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parent.parent
INTERIM = ROOT / "data/interim"
REPORTS = ROOT / "reports"

DT_GRID = 1.0
DT_MIN = 5.0
BAD_FIT_R2 = 0.80
BAD_FIT_RMSE = 0.30
HOS_THRESHOLD = 0.6

THETA_GRID = [0.0, 0.001, 0.002, 0.003, 0.005, 0.007,
              0.010, 0.015, 0.020, 0.030, 0.050, 0.080, 0.120, 0.200]


class UF:
    def __init__(self):
        self.p: dict = {}

    def find(self, a):
        self.p.setdefault(a, a)
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def _apelblat(T, A, B, C):
    return A + B / T + C * np.log(T)


def _vhf(T, A, B):
    return A + B / T


def _fit_one(T: np.ndarray, y: np.ndarray) -> tuple[str, tuple, float, float] | None:
    n = len(T)
    try:
        if n >= 3:
            popt, _ = curve_fit(_apelblat, T, y, p0=[y.mean(), 0, 0], maxfev=5000)
            yhat = _apelblat(T, *popt)
            ss_res = float(np.sum((y - yhat) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
            rmse = float(np.sqrt(ss_res / n))
            return "apelblat", popt, r2, rmse
        if n == 2:
            popt, _ = curve_fit(_vhf, T, y, p0=[y.mean(), 0], maxfev=5000)
            yhat = _vhf(T, *popt)
            ss_res = float(np.sum((y - yhat) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
            rmse = float(np.sqrt(ss_res / n))
            return "vanthoff", popt, r2, rmse
    except Exception:
        return None
    return None


def _eval_fit(model: str, coefs: tuple, T: np.ndarray):
    if model == "apelblat":
        return _apelblat(T, *coefs)
    return _vhf(T, *coefs)


# ─────────────────────────────────────────────────────────────────────────
def build_prelim_groups(df: pd.DataFrame) -> dict[str, int]:
    """Stage A: bit-exact at (solute, solvent, round(T,2), round(logS,4))."""
    sub = df.copy()
    sub["T_r"] = sub["Temperature_K"].round(2)
    sub["logS_r"] = sub["LogS"].round(4)
    key = ["Solute_Canon", "Solvent_Canon", "T_r", "logS_r"]
    grp = sub.groupby(key)["Source"].agg(lambda x: sorted(set(x))).reset_index(name="dois")
    grp["n_dois"] = grp["dois"].map(len)
    hits = grp[grp["n_dois"] >= 2]
    uf = UF()
    for d in sub["Source"].unique():
        uf.find(d)
    for dois in hits["dois"]:
        root = dois[0]
        for d in dois[1:]:
            uf.union(root, d)
    roots = {uf.find(d) for d in sub["Source"].unique()}
    root_to_id = {r: i for i, r in enumerate(sorted(roots))}
    return {d: root_to_id[uf.find(d)] for d in sub["Source"].unique()}


def fit_per_prelim_triple(df: pd.DataFrame, doi_to_prelim: dict):
    """Return dict[(solute, solvent, pgid)] = (model, coefs, R2, RMSE, Tmin, Tmax)."""
    df = df.copy()
    df["pgid"] = df["Source"].map(doi_to_prelim)
    fits: dict = {}
    for (solute, solvent, pg), sub in df.groupby(["Solute_Canon", "Solvent_Canon", "pgid"]):
        # average logS at equal T
        g = sub.groupby("Temperature_K")["LogS"].mean().reset_index()
        T = g["Temperature_K"].to_numpy(dtype=float)
        y = g["LogS"].to_numpy(dtype=float)
        order = np.argsort(T)
        T, y = T[order], y[order]
        res = _fit_one(T, y)
        if res is None:
            continue
        model, coefs, r2, rmse = res
        if r2 < BAD_FIT_R2 or rmse > BAD_FIT_RMSE:
            continue
        fits[(solute, solvent, int(pg))] = (model, coefs, r2, rmse, float(T.min()), float(T.max()))
    return fits


def precompute_prelim_pair_data(fits: dict):
    """For every prelim-group pair sharing a (solute, solvent), compute
    their interpolated atom-level |Δ| and weighted MAE."""
    by_ss: dict[tuple, list] = defaultdict(list)
    for (solute, solvent, pg), val in fits.items():
        by_ss[(solute, solvent)].append((pg, val))

    # pair_key → {"weighted_mae_sum": float, "grid_count": int,
    #             "per_pair_deltas": list of {"solute","solvent","deltas":array}}
    pair_entries = defaultdict(lambda: {"n_shared_pairs": 0, "sum_delta": 0.0, "n_grid": 0,
                                         "ss_deltas": []})
    for (solute, solvent), items in by_ss.items():
        if len(items) < 2:
            continue
        for (pi, vi), (pj, vj) in combinations(sorted(items, key=lambda x: x[0]), 2):
            Tmin = max(vi[4], vj[4])
            Tmax = min(vi[5], vj[5])
            if Tmax - Tmin < DT_MIN:
                continue
            n_pts = int(np.floor((Tmax - Tmin) / DT_GRID)) + 1
            T = Tmin + np.arange(n_pts) * DT_GRID
            y_i = _eval_fit(vi[0], vi[1], T)
            y_j = _eval_fit(vj[0], vj[1], T)
            deltas = np.abs(y_i - y_j)
            key = (pi, pj) if pi < pj else (pj, pi)
            e = pair_entries[key]
            e["n_shared_pairs"] += 1
            e["sum_delta"] += float(deltas.sum())
            e["n_grid"] += int(len(deltas))
            e["ss_deltas"].append((solute, solvent, deltas))
    return pair_entries


def union_for_theta(prelim_ids: set, pair_entries: dict, theta: float):
    uf = UF()
    for p in prelim_ids:
        uf.find(p)
    for (pi, pj), e in pair_entries.items():
        if e["n_grid"] == 0:
            continue
        wmae = e["sum_delta"] / e["n_grid"]
        if wmae < theta:
            uf.union(pi, pj)
    roots = {uf.find(p): None for p in prelim_ids}
    root_to_fgid = {r: i for i, r in enumerate(sorted(roots))}
    prelim_to_final = {p: root_to_fgid[uf.find(p)] for p in prelim_ids}
    return prelim_to_final


def compute_metrics(df: pd.DataFrame, doi_to_prelim: dict, fits: dict,
                    pair_entries: dict, prelim_to_final: dict) -> dict:
    """Given final groups, compute ε_A primary + P90 + counts."""
    # Build per-final representative fit per (solute, solvent, final group)
    # (pick best R² among prelim members)
    best_rep: dict = {}
    for (solute, solvent, pg), val in fits.items():
        fg = prelim_to_final[pg]
        key = (solute, solvent, fg)
        if key not in best_rep or val[2] > best_rep[key][2]:
            best_rep[key] = val

    # Per-(solute, solvent), list of (fg, val) with distinct fg
    ss_to_fgs: dict = defaultdict(list)
    for (solute, solvent, fg), val in best_rep.items():
        ss_to_fgs[(solute, solvent)].append((fg, val))

    # Compute per-final-group reliability (deviation from consensus of others)
    fg_devs: dict = defaultdict(list)
    for (solute, solvent), items in ss_to_fgs.items():
        if len(items) < 2:
            continue
        Tmin = max(v[4] for _, v in items)
        Tmax = min(v[5] for _, v in items)
        if Tmax - Tmin < DT_MIN:
            continue
        T = Tmin + np.arange(int((Tmax - Tmin) / DT_GRID) + 1) * DT_GRID
        ys = {fg: _eval_fit(v[0], v[1], T) for fg, v in items}
        for fg, _ in items:
            others = [ys[o] for o in ys if o != fg]
            if not others:
                continue
            cons = np.mean(np.stack(others, axis=0), axis=0)
            fg_devs[fg].extend(np.abs(ys[fg] - cons).tolist())

    hos_fgs = set()
    for fg, devs in fg_devs.items():
        if len(devs) == 0:
            continue
        if np.mean(devs) >= HOS_THRESHOLD:
            hos_fgs.add(fg)

    # Pair-level aleatoric on non-HoS pairs
    pair_maes = []
    for (solute, solvent), items in ss_to_fgs.items():
        usable = [(fg, v) for fg, v in items if fg not in hos_fgs]
        if len(usable) < 2:
            continue
        gp_maes = []
        for (fi_fg, fi), (fj_fg, fj) in combinations(usable, 2):
            Tmin = max(fi[4], fj[4])
            Tmax = min(fi[5], fj[5])
            if Tmax - Tmin < DT_MIN:
                continue
            T = Tmin + np.arange(int((Tmax - Tmin) / DT_GRID) + 1) * DT_GRID
            y_i = _eval_fit(fi[0], fi[1], T)
            y_j = _eval_fit(fj[0], fj[1], T)
            gp_maes.append(float(np.mean(np.abs(y_i - y_j))))
        if gp_maes:
            pair_maes.append(float(np.mean(gp_maes)))
    arr = np.asarray(pair_maes)
    if arr.size == 0:
        return {"n_pairs": 0}
    return {
        "n_pairs": int(arr.size),
        "eps_A": float(np.mean(arr)),
        "P90": float(np.percentile(arr, 90)),
        "median": float(np.median(arr)),
        "RMSE": float(np.sqrt(np.mean(arr ** 2))),
        "n_final_groups": int(len(set(prelim_to_final.values()))),
        "n_hos": int(len(hos_fgs)),
    }


def main():
    print("Loading data…")
    df = pd.read_csv(INTERIM / "02_cleaned.csv")
    print(f"  rows: {len(df):,}  DOIs: {df['Source'].nunique()}")

    print("Stage A (prelim groups)…")
    doi_to_prelim = build_prelim_groups(df)
    print(f"  prelim groups: {len(set(doi_to_prelim.values()))}")

    print("Fitting Apelblat / van't Hoff per prelim-group triple…")
    fits = fit_per_prelim_triple(df, doi_to_prelim)
    print(f"  usable fits: {len(fits):,}")

    print("Precomputing prelim-group pair interpolated deltas…")
    pair_entries = precompute_prelim_pair_data(fits)
    print(f"  prelim-group pairs with ≥1 shared (solute, solvent): {len(pair_entries)}")

    prelim_ids = set(doi_to_prelim.values())
    results = []
    for theta in THETA_GRID:
        prelim_to_final = union_for_theta(prelim_ids, pair_entries, theta)
        metrics = compute_metrics(df, doi_to_prelim, fits, pair_entries, prelim_to_final)
        metrics["theta"] = theta
        results.append(metrics)
        print(f"  θ = {theta:.3f}  → ε_A = {metrics.get('eps_A', float('nan')):.4f}  "
              f"P90 = {metrics.get('P90', float('nan')):.4f}  "
              f"n_pairs = {metrics['n_pairs']}  n_hos = {metrics.get('n_hos', 0)}")

    out = pd.DataFrame(results)
    out.to_csv(REPORTS / "55_threshold_sweep.csv", index=False)
    print(f"Saved → reports/55_threshold_sweep.csv")


if __name__ == "__main__":
    main()
