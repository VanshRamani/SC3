"""
Phase 6 — SC³ tier construction.

Policy:
  • Eligibility.  Every (solute, solvent) multi-source pair from Phase 5's
    HoS-excluded pool (n = 481 pairs).  Pairs whose groups include any
    Hall-of-Shame independence group are NOT tier-eligible (they are still
    available in the training pool).

  • Thresholds (pair-level MAE from Phase 5 pure-interpolation).
        Gold   — pair_MAE ≤ 0.1 log S
        Silver — pair_MAE ≤ 0.2 log S
        Bronze — pair_MAE ≤ 0.5 log S
    Nested: Gold ⊂ Silver ⊂ Bronze.

  • Consensus.  For each (solute, solvent) pair, at every reference
    temperature measured by at least one contributing non-HoS group:
        contributing = {groups with a usable Apelblat/vanthoff fit whose
                        [T_min, T_max] covers T}
        consensus_logS = mean of contributing fits at T
        σ              = std of contributing fits at T  (NaN if n<2;
                         floored at 0.012 log S — median Apelblat RMSE)
    No extrapolation: only groups whose fit range strictly covers T are
    used.

  • Output rows are at MEASURED temperatures only (no synthetic points),
    so every benchmark row corresponds to an actual physical measurement
    reported by some non-HoS group.

Inputs:
  data/interim/02_cleaned.csv
  data/interim/03_doi_groups.csv         DOI → final_group_id
  data/interim/04_fits.csv               per (solute, solvent, final_gid)
  data/interim/05_pair_mae.csv           per multi-source pair, with pair_MAE + any_hos flag
  reports/30_doi_reliability.csv         final_gid → reliability (HoS list)

Outputs:
  data/sc3/gold.csv                     (tightest GT, ⊂ silver ⊂ bronze)
  data/sc3/silver.csv                   (silver ⊃ gold)
  data/sc3/bronze.csv                   (bronze ⊃ silver ⊃ gold)
  data/sc3/tier_pairs.csv               one row per multi-source pair, tier flags
  data/sc3/tier_summary.json            row / pair / solute / solvent counts, σ coverage
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INTERIM = ROOT / "data/interim"
REPORTS = ROOT / "reports"
SC3 = ROOT / "data/sc3"
SC3.mkdir(parents=True, exist_ok=True)

GOLD_T   = 0.10
SILVER_T = 0.20
BRONZE_T = 0.50

SIGMA_FLOOR = 0.012               # floor, ≈ median Apelblat fit RMSE
BAD_FIT_R2   = 0.80
BAD_FIT_RMSE = 0.30
HOS_THRESHOLD = 0.6               # mean_abs_dev_from_consensus (log S)


def _eval(row: pd.Series, T: float) -> float:
    m = row["model"]
    if m == "apelblat":
        return row["A"] + row["B"] / T + row["C"] * np.log(T)
    if m == "vanthoff":
        return row["A"] + row["B"] / T
    return float("nan")


def main():
    # ── Load everything ──
    df = pd.read_csv(INTERIM / "02_cleaned.csv")
    gmap = pd.read_csv(INTERIM / "03_doi_groups.csv").set_index("doi")["group_id"].to_dict()
    df["gid"] = df["Source"].map(gmap)
    fits = pd.read_csv(INTERIM / "04_fits.csv")
    rel = pd.read_csv(REPORTS / "30_doi_reliability.csv")
    pair_mae = pd.read_csv(INTERIM / "05_pair_mae.csv")

    # Usable fits
    fits["fit_ok"] = (
        (fits["model"].isin(["apelblat", "vanthoff"]))
        & (fits["R2"].fillna(0) >= BAD_FIT_R2)
        & (fits["RMSE"].fillna(1) <= BAD_FIT_RMSE)
    )
    fit_by_key = {
        (r["Solute_Canon"], r["Solvent_Canon"], int(r["gid"])): r
        for _, r in fits.iterrows() if r["fit_ok"]
    }

    # Hall-of-Shame independence groups (final_gid level)
    hos_groups = set(rel[rel["mean_abs_dev_from_consensus"] >= HOS_THRESHOLD]["final_gid"])
    print(f"Hall-of-Shame final groups: {len(hos_groups)}")

    # Tier-eligible pairs: from Phase 5 primary pool (any_hos == False)
    eligible = pair_mae[~pair_mae["any_hos"]].copy()
    print(f"Tier-eligible pairs (HoS-excluded): {len(eligible)}")
    print(f"  distribution of pair_MAE: min {eligible['pair_MAE'].min():.4f}, "
          f"median {eligible['pair_MAE'].median():.4f}, "
          f"P90 {eligible['pair_MAE'].quantile(0.9):.4f}, "
          f"max {eligible['pair_MAE'].max():.4f}")

    # Precompute measured T values per (solute, solvent, gid)
    meas_T = (df.groupby(["Solute_Canon", "Solvent_Canon", "gid"])
                ["Temperature_K"]
                .agg(lambda x: sorted(set(round(float(t), 2) for t in x))))

    # ═══════════════════════════════════════════════════════════════════
    # Build tier rows
    # ═══════════════════════════════════════════════════════════════════
    tier_rows: list[dict] = []
    pair_info: list[dict] = []
    n_rows_covered_by_n_contributing: dict[int, int] = defaultdict(int)

    for _, pr in eligible.iterrows():
        solute = pr["Solute_Canon"]
        solvent = pr["Solvent_Canon"]
        pair_mae_val = float(pr["pair_MAE"])

        # Assign tiers (nested)
        tiers: list[str] = []
        if pair_mae_val <= GOLD_T:
            tiers.append("gold")
        if pair_mae_val <= SILVER_T:
            tiers.append("silver")
        if pair_mae_val <= BRONZE_T:
            tiers.append("bronze")
        if not tiers:
            continue  # pair_MAE > 0.5 — not tier-eligible

        # Find all contributing groups (non-HoS with usable fit for this pair)
        pair_gids = df[(df["Solute_Canon"] == solute)
                       & (df["Solvent_Canon"] == solvent)]["gid"].unique()
        contrib_gids = [
            int(g) for g in pair_gids
            if int(g) not in hos_groups
            and (solute, solvent, int(g)) in fit_by_key
        ]
        if len(contrib_gids) < 2:
            continue  # shouldn't happen since pair_mae had >=2 usable; but guard

        # Union of measured Ts from contributing groups
        ref_Ts = sorted({
            t for g in contrib_gids
            for t in meas_T.get((solute, solvent, g), [])
        })

        n_rows_for_pair = 0
        n_sigma_defined = 0
        sigmas_for_pair: list[float] = []
        for T in ref_Ts:
            contributing_vals: list[float] = []
            contributing_g: list[int] = []
            for g in contrib_gids:
                fit = fit_by_key[(solute, solvent, g)]
                if fit["T_min"] <= T <= fit["T_max"]:
                    y = _eval(fit, T)
                    if np.isfinite(y):
                        contributing_vals.append(float(y))
                        contributing_g.append(g)
            n_contrib = len(contributing_vals)
            n_rows_covered_by_n_contributing[n_contrib] += 1
            if n_contrib < 1:
                continue  # should be rare — T outside all fits
            consensus = float(np.mean(contributing_vals))
            if n_contrib >= 2:
                raw_sigma = float(np.std(contributing_vals, ddof=1))
                sigma = max(raw_sigma, SIGMA_FLOOR)
                sigmas_for_pair.append(sigma)
                n_sigma_defined += 1
            else:
                sigma = float("nan")
            row = {
                "Solute_Canon": solute,
                "Solvent_Canon": solvent,
                "Temperature_K": T,
                "LogS_consensus": consensus,
                "sigma": sigma,
                "n_contributing_groups": n_contrib,
                "pair_MAE": pair_mae_val,
                "tier_gold":   "gold" in tiers,
                "tier_silver": "silver" in tiers,
                "tier_bronze": "bronze" in tiers,
            }
            tier_rows.append(row)
            n_rows_for_pair += 1

        pair_info.append({
            "Solute_Canon": solute,
            "Solvent_Canon": solvent,
            "pair_MAE": pair_mae_val,
            "n_contributing_groups": len(contrib_gids),
            "n_rows": n_rows_for_pair,
            "n_rows_sigma_defined": n_sigma_defined,
            "median_sigma": float(np.median(sigmas_for_pair)) if sigmas_for_pair else None,
            "tier_gold":   "gold" in tiers,
            "tier_silver": "silver" in tiers,
            "tier_bronze": "bronze" in tiers,
        })

    master = pd.DataFrame(tier_rows)
    pairs_df = pd.DataFrame(pair_info)

    # ═══════════════════════════════════════════════════════════════════
    # Write tier CSVs (nested: Gold ⊂ Silver ⊂ Bronze)
    # ═══════════════════════════════════════════════════════════════════
    gold = master[master["tier_gold"]].copy().drop(columns=["tier_gold", "tier_silver", "tier_bronze"])
    silver = master[master["tier_silver"]].copy().drop(columns=["tier_gold", "tier_silver", "tier_bronze"])
    bronze = master[master["tier_bronze"]].copy().drop(columns=["tier_gold", "tier_silver", "tier_bronze"])
    gold.to_csv(SC3 / "gold.csv", index=False)
    silver.to_csv(SC3 / "silver.csv", index=False)
    bronze.to_csv(SC3 / "bronze.csv", index=False)
    pairs_df.to_csv(SC3 / "tier_pairs.csv", index=False)

    # ═══════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════
    def _counts(sub: pd.DataFrame) -> dict:
        n_rows = int(len(sub))
        n_pairs = int(sub.groupby(["Solute_Canon", "Solvent_Canon"]).ngroups)
        n_solutes = int(sub["Solute_Canon"].nunique())
        n_solvents = int(sub["Solvent_Canon"].nunique())
        n_sigma_defined = int(sub["sigma"].notna().sum())
        coverage = n_sigma_defined / n_rows if n_rows > 0 else 0.0
        if n_sigma_defined > 0:
            σ = sub["sigma"].dropna().to_numpy()
            σ_stats = {
                "median": float(np.median(σ)),
                "mean": float(np.mean(σ)),
                "P90": float(np.percentile(σ, 90)),
                "P95": float(np.percentile(σ, 95)),
            }
        else:
            σ_stats = None
        return {
            "n_rows": n_rows,
            "n_pairs": n_pairs,
            "n_solutes": n_solutes,
            "n_solvents": n_solvents,
            "n_rows_sigma_defined": n_sigma_defined,
            "sigma_coverage_frac": float(coverage),
            "sigma_stats": σ_stats,
        }

    summary = {
        "policy": {
            "thresholds_pair_MAE": {"gold": GOLD_T, "silver": SILVER_T, "bronze": BRONZE_T},
            "sigma_floor_logS": SIGMA_FLOOR,
            "HoS_threshold_logS": HOS_THRESHOLD,
            "BAD_FIT_R2_min": BAD_FIT_R2,
            "BAD_FIT_RMSE_max": BAD_FIT_RMSE,
            "extrapolation_allowed": False,
            "consensus_from": "non-HoS groups' Apelblat/vanthoff fits at measured reference Ts",
            "sigma_from": "std across fit evaluations at each T (NaN if n<2, floored to 0.012)",
            "nesting": "Gold ⊂ Silver ⊂ Bronze",
        },
        "tier_eligible_pairs": int(len(eligible)),
        "per_tier": {
            "gold":   _counts(gold),
            "silver": _counts(silver),
            "bronze": _counts(bronze),
        },
        "n_contributing_groups_per_row_hist": {
            int(k): int(v) for k, v in sorted(n_rows_covered_by_n_contributing.items())
        },
    }
    with open(SC3 / "tier_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Stdout report
    print(f"\n{'═'*72}")
    print(f"{'Tier':<8}{'pairs':>10}{'rows':>10}{'solutes':>10}{'solvents':>10}"
          f"{'σ-cov':>10}{'median σ':>12}")
    print(f"{'─'*72}")
    for name in ("gold", "silver", "bronze"):
        s = summary["per_tier"][name]
        σmed = s["sigma_stats"]["median"] if s["sigma_stats"] else float("nan")
        print(f"{name.capitalize():<8}{s['n_pairs']:>10}{s['n_rows']:>10}"
              f"{s['n_solutes']:>10}{s['n_solvents']:>10}"
              f"{s['sigma_coverage_frac']*100:>9.1f}%{σmed:>12.4f}")
    print(f"{'═'*72}")

    print(f"\nPer-row contributor histogram:")
    for k, v in sorted(n_rows_covered_by_n_contributing.items()):
        pct = 100 * v / max(1, sum(n_rows_covered_by_n_contributing.values()))
        print(f"  n_contrib = {k}:  {v:>6}  ({pct:.1f}%)")

    print(f"\nWrote: data/sc3/gold.csv, silver.csv, bronze.csv, tier_pairs.csv, tier_summary.json")


if __name__ == "__main__":
    main()
