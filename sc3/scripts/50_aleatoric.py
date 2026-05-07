"""
Phase 5 — Aleatoric limit (pure-interpolation, corrected data).

Definition (D-12 / D-14, updated 2026-04-17):

  For each (solute, solvent) pair with ≥2 independent source groups, each
  with a usable Apelblat or van't Hoff fit (R² ≥ 0.80, RMSE ≤ 0.30):
    For each pair of groups (i, j):
      T_min_ij = max(fit_i.T_min, fit_j.T_min)
      T_max_ij = min(fit_i.T_max, fit_j.T_max)
      Skip if overlap range < DT_MIN (= 5 K).
      Reference grid = arange(T_min_ij, T_max_ij + ε, DT_GRID)  (DT_GRID = 1 K).
      At every reference T, evaluate BOTH groups' fits (never measured values).
      |Δ logS| = |fit_i(T) − fit_j(T)| at every reference T.
    Group-pair MAE = mean of |Δ|.
    Pair MAE = mean of group-pair MAE within the (solute, solvent) pair.

  ε_A = mean over all multi-source pairs of Pair MAE.

Single-point groups (n = 1 measurement, no fit) and bad-fit groups are
excluded from the aleatoric computation (they cannot be interpolated).

Two headline numbers are reported:
  primary   — excluding Hall-of-Shame independence groups (mean_abs_dev_from_
              consensus ≥ 0.6 log S from Phase 3 output).  This is the
              "community-consensus-agreeing" floor used for tier construction.
  inclusive — all independence groups.

Inputs:
  data/interim/02_cleaned.csv
  data/interim/03_doi_groups.csv
  data/interim/04_fits.csv
  reports/30_doi_reliability.csv

Outputs:
  data/interim/05_pair_mae.csv
  data/interim/05_atom_deltas.csv
  reports/50_aleatoric.json
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import gamma, kstest

ROOT = Path(__file__).resolve().parent.parent
INTERIM = ROOT / "data/interim"
REPORTS = ROOT / "reports"

# ── policy constants ──────────────────────────────────────────────────────
DT_GRID = 1.0     # K — uniform reference grid spacing
DT_MIN = 5.0      # K — minimum overlap range to include a group pair
BAD_FIT_R2 = 0.80
BAD_FIT_RMSE = 0.30
HOS_THRESHOLD = 0.6   # Hall-of-Shame: mean_abs_dev_from_consensus ≥ this
BOOT_SEED = 42
BOOT_N = 5000
BOOT_ALPHA = 0.05


# ── model evaluators ──────────────────────────────────────────────────────
def _eval(row: pd.Series, T: float | np.ndarray):
    m = row["model"]
    if m == "apelblat":
        return row["A"] + row["B"] / T + row["C"] * np.log(T)
    if m == "vanthoff":
        return row["A"] + row["B"] / T
    raise ValueError(f"Cannot evaluate model='{m}'")


def main():
    df = pd.read_csv(INTERIM / "02_cleaned.csv")
    gmap = pd.read_csv(INTERIM / "03_doi_groups.csv").set_index("doi")["group_id"].to_dict()
    df["gid"] = df["Source"].map(gmap)
    fits = pd.read_csv(INTERIM / "04_fits.csv")
    rel = pd.read_csv(REPORTS / "30_doi_reliability.csv")

    # Fit-quality mask: usable Apelblat or van't Hoff with R² and RMSE thresholds
    fits["fit_ok"] = (
        (fits["model"].isin(["apelblat", "vanthoff"]))
        & (fits["R2"].fillna(0) >= BAD_FIT_R2)
        & (fits["RMSE"].fillna(1) <= BAD_FIT_RMSE)
    )
    fit_by_key = {
        (r["Solute_Canon"], r["Solvent_Canon"], int(r["gid"])): r
        for _, r in fits.iterrows() if r["fit_ok"]
    }
    print(f"Loaded {len(df):,} rows, {len(fit_by_key):,} usable fits "
          f"(of {len(fits):,} triples)")

    # Hall-of-Shame groups (per Phase 3 reliability ranking)
    hos_dois = set(rel[rel["mean_abs_dev_from_consensus"] >= HOS_THRESHOLD]["doi"])
    hos_gids = set(df[df["Source"].isin(hos_dois)]["gid"].unique())
    print(f"Hall-of-Shame DOIs:   {len(hos_dois)}")
    print(f"Hall-of-Shame groups: {len(hos_gids)}")

    # Identify multi-source pairs with ≥2 usable-fit groups
    pair_gid = (df.groupby(["Solute_Canon", "Solvent_Canon"])["gid"]
                  .nunique()
                  .reset_index(name="n_groups"))
    multi_pairs = pair_gid[pair_gid["n_groups"] >= 2]
    print(f"Multi-source pairs (≥2 groups, any kind): {len(multi_pairs):,}")

    atom_rows: list[dict] = []
    pair_rows: list[dict] = []
    skipped = defaultdict(int)

    for _, prow in multi_pairs.iterrows():
        solute = prow["Solute_Canon"]
        solvent = prow["Solvent_Canon"]
        gids = sorted(df[(df["Solute_Canon"] == solute)
                         & (df["Solvent_Canon"] == solvent)]["gid"].unique())
        # keep only groups with usable fit for this triple
        usable_gids = [g for g in gids if (solute, solvent, int(g)) in fit_by_key]
        if len(usable_gids) < 2:
            skipped["<2_usable_fits"] += 1
            continue

        group_pair_maes: list[float] = []
        for gi, gj in combinations(usable_gids, 2):
            fi = fit_by_key[(solute, solvent, int(gi))]
            fj = fit_by_key[(solute, solvent, int(gj))]
            T_min = max(float(fi["T_min"]), float(fj["T_min"]))
            T_max = min(float(fi["T_max"]), float(fj["T_max"]))
            if T_max - T_min < DT_MIN:
                skipped["T_overlap_too_small"] += 1
                continue
            # Uniform reference grid within [T_min, T_max]
            n_pts = int(np.floor((T_max - T_min) / DT_GRID)) + 1
            ref_T = T_min + np.arange(n_pts) * DT_GRID
            y_i = _eval(fi, ref_T)
            y_j = _eval(fj, ref_T)
            deltas = np.abs(np.asarray(y_i) - np.asarray(y_j))
            for T, d in zip(ref_T, deltas):
                atom_rows.append({
                    "Solute_Canon": solute,
                    "Solvent_Canon": solvent,
                    "gi": int(gi), "gj": int(gj),
                    "T": float(T), "delta": float(d),
                    "gi_is_hos": int(gi) in hos_gids,
                    "gj_is_hos": int(gj) in hos_gids,
                })
            group_pair_maes.append(float(np.mean(deltas)))

        if len(group_pair_maes) == 0:
            skipped["no_valid_group_pairs"] += 1
            continue

        pair_rows.append({
            "Solute_Canon": solute,
            "Solvent_Canon": solvent,
            "n_groups_total": int(len(gids)),
            "n_groups_usable": int(len(usable_gids)),
            "n_group_pairs": int(len(group_pair_maes)),
            "pair_MAE": float(np.mean(group_pair_maes)),
            "pair_MAE_max": float(np.max(group_pair_maes)),
            "any_hos": any(gid in hos_gids for gid in usable_gids),
        })

    pair_df = pd.DataFrame(pair_rows)
    atom_df = pd.DataFrame(atom_rows)
    pair_df.to_csv(INTERIM / "05_pair_mae.csv", index=False)
    atom_df.to_csv(INTERIM / "05_atom_deltas.csv", index=False)

    # ═══════════════════════════════════════════════════════════════════
    # Headline stats — primary (HoS excluded) and inclusive
    # ═══════════════════════════════════════════════════════════════════
    # Primary: only pairs where no group is in HoS
    pair_primary = pair_df[~pair_df["any_hos"]]
    atom_primary = atom_df[~(atom_df["gi_is_hos"] | atom_df["gj_is_hos"])]

    rng = np.random.default_rng(BOOT_SEED)

    def _stats(arr: np.ndarray) -> dict:
        if arr.size == 0:
            return {"n": 0}
        return {
            "n": int(arr.size),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "P75": float(np.percentile(arr, 75)),
            "P90": float(np.percentile(arr, 90)),
            "P95": float(np.percentile(arr, 95)),
            "P99": float(np.percentile(arr, 99)),
            "RMSE": float(np.sqrt(np.mean(arr ** 2))),
            "max": float(np.max(arr)),
        }

    def _ci(arr: np.ndarray) -> list[float]:
        if arr.size < 2:
            return [float("nan"), float("nan")]
        s = rng.choice(arr, size=(BOOT_N, arr.size), replace=True).mean(axis=1)
        return [float(np.percentile(s, 100 * BOOT_ALPHA / 2)),
                float(np.percentile(s, 100 * (1 - BOOT_ALPHA / 2)))]

    stats_pair_primary = _stats(pair_primary["pair_MAE"].to_numpy())
    stats_pair_inclusive = _stats(pair_df["pair_MAE"].to_numpy())
    stats_atom_primary = _stats(atom_primary["delta"].to_numpy())
    stats_atom_inclusive = _stats(atom_df["delta"].to_numpy())
    stats_pair_primary["mean_95CI"] = _ci(pair_primary["pair_MAE"].to_numpy())
    stats_pair_inclusive["mean_95CI"] = _ci(pair_df["pair_MAE"].to_numpy())

    # Gamma fit (atom-level, primary — for paper figure)
    pos = atom_primary["delta"].to_numpy()
    pos = pos[pos > 1e-12]
    gamma_fit: dict = {}
    if pos.size > 100:
        shape, loc, scale = gamma.fit(pos, floc=0)
        ks = float(kstest(pos, "gamma", args=(shape, loc, scale)).statistic)
        gamma_fit = {"shape": float(shape), "scale": float(scale),
                     "loc": float(loc), "ks_stat": ks}

    # Per-solvent decomposition (primary only, ≥10 pairs)
    per_solv = (pair_primary.groupby("Solvent_Canon")
                .agg(n_pairs=("pair_MAE", "size"),
                     mean_pair_MAE=("pair_MAE", "mean"),
                     median_pair_MAE=("pair_MAE", "median"),
                     P90_pair_MAE=("pair_MAE", lambda x: np.percentile(x, 90)))
                .reset_index()
                .sort_values("n_pairs", ascending=False))
    per_solv.to_csv(REPORTS / "50_per_solvent_aleatoric.csv", index=False)

    # ═══════════════════════════════════════════════════════════════════
    # Print
    # ═══════════════════════════════════════════════════════════════════
    print(f"\nSkipped reasons: {dict(skipped)}")
    print(f"\n{'═'*72}\nε_A — primary (HoS-excluded)\n{'═'*72}")
    print(f"  ε_A = {stats_pair_primary['mean']:.4f} log S  "
          f"(95% CI [{stats_pair_primary['mean_95CI'][0]:.4f}, "
          f"{stats_pair_primary['mean_95CI'][1]:.4f}])")
    print(f"  n pairs: {stats_pair_primary['n']},  "
          f"n atom |Δ|: {stats_atom_primary['n']}")
    print(f"  distribution: median {stats_pair_primary['median']:.4f}, "
          f"P90 {stats_pair_primary['P90']:.4f}, P95 {stats_pair_primary['P95']:.4f}, "
          f"RMSE {stats_pair_primary['RMSE']:.4f}, max {stats_pair_primary['max']:.4f}")

    print(f"\n{'═'*72}\nε_A — inclusive\n{'═'*72}")
    print(f"  ε_A = {stats_pair_inclusive['mean']:.4f} log S  "
          f"(95% CI [{stats_pair_inclusive['mean_95CI'][0]:.4f}, "
          f"{stats_pair_inclusive['mean_95CI'][1]:.4f}])")
    print(f"  n pairs: {stats_pair_inclusive['n']}")
    print(f"  distribution: median {stats_pair_inclusive['median']:.4f}, "
          f"P90 {stats_pair_inclusive['P90']:.4f}, P95 {stats_pair_inclusive['P95']:.4f}, "
          f"RMSE {stats_pair_inclusive['RMSE']:.4f}")

    if gamma_fit:
        print(f"\nGamma fit (atom-level, primary): shape={gamma_fit['shape']:.3f}, "
              f"scale={gamma_fit['scale']:.3f}, KS={gamma_fit['ks_stat']:.3f}")

    print(f"\nTop 10 solvents by n_pairs (primary):")
    print(per_solv[per_solv["n_pairs"] >= 10].head(10).to_string(index=False))

    report = {
        "policy": {
            "DT_GRID_K": DT_GRID,
            "DT_MIN_overlap_K": DT_MIN,
            "BAD_FIT_R2_min": BAD_FIT_R2,
            "BAD_FIT_RMSE_max": BAD_FIT_RMSE,
            "HOS_threshold_logS": HOS_THRESHOLD,
            "interpolation_only": True,
            "measured_values_used": False,
        },
        "primary": {
            "epsA_mean": stats_pair_primary["mean"],
            "epsA_mean_95CI": stats_pair_primary["mean_95CI"],
            "pair_distribution": stats_pair_primary,
            "atom_distribution": stats_atom_primary,
            "n_pairs_excluded_as_hos": int(len(pair_df) - len(pair_primary)),
        },
        "inclusive": {
            "epsA_mean": stats_pair_inclusive["mean"],
            "epsA_mean_95CI": stats_pair_inclusive["mean_95CI"],
            "pair_distribution": stats_pair_inclusive,
            "atom_distribution": stats_atom_inclusive,
        },
        "gamma_fit_atom_level_primary": gamma_fit,
        "skipped_reasons": dict(skipped),
        "hos": {
            "n_dois": len(hos_dois),
            "n_groups": len(hos_gids),
        },
    }
    with open(REPORTS / "50_aleatoric.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote: reports/50_aleatoric.json")


if __name__ == "__main__":
    main()
