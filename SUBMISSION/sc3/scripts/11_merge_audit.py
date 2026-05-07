"""
Phase 1.5 — Empirical audit of every Option-C merge group.

For each merge group (canon_C shared by ≥2 raw SMILES):
  1. Inside the group, find (solvent, T to 0.1 K) cells where ≥2 distinct raw
     SMILES have measurements.  Compute |Δ logS| per overlap cell.
  2. Aggregate per group: median and max |Δ logS|, and the count of overlap
     cells.
  3. Also report for each raw member: n_rows, n_dois, compound_name_mode.

Output:
  reports/11_merge_audit_empirical.csv   (one row per (merge_group, raw_member))
  reports/11_merge_audit_summary.csv     (one row per merge_group, aggregate)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INTERIM = ROOT / "data/interim"
REPORTS = ROOT / "reports"


def main():
    df = pd.read_csv(INTERIM / "01_canonical.csv")
    # Merge groups = canon_C with ≥2 distinct raw SMILES
    g = df.groupby("Solute_Canon")["SMILES_Solute"].nunique()
    merge_canons = g[g > 1].index.tolist()
    print(f"Merge groups: {len(merge_canons)}")

    per_member_rows = []
    per_group_rows = []

    for canon in merge_canons:
        sub = df[df["Solute_Canon"] == canon].copy()
        sub["T_r"] = sub["Temperature_K"].round(1)
        sub = sub.dropna(subset=["LogS(mol/L)"])
        raw_members = sub["SMILES_Solute"].unique().tolist()

        # per-member descriptive stats
        for r in raw_members:
            ss = sub[sub["SMILES_Solute"] == r]
            per_member_rows.append({
                "canon_C": canon,
                "raw": r,
                "n_rows": int(len(ss)),
                "n_dois": int(ss["Source"].nunique()),
                "n_solvents": int(ss["Solvent"].nunique()),
                "compound_name_mode": ss["Compound_Name"].mode().iat[0]
                    if not ss["Compound_Name"].mode().empty
                    else (ss["Compound_Name"].iloc[0] if len(ss) else None),
                "cas_mode": ss["CAS"].mode().iat[0]
                    if not ss["CAS"].mode().empty
                    else (ss["CAS"].iloc[0] if len(ss) else None),
                "dois": sorted(set(ss["Source"])),
            })

        # Overlap cells: (Solvent, T_r) where ≥2 raw members report
        cell_g = (sub.groupby(["Solvent", "T_r"])
                    .agg(n_raw=("SMILES_Solute", "nunique"),
                         logS_values=("LogS(mol/L)", list),
                         raw_list=("SMILES_Solute", list))
                    .reset_index())
        overlap = cell_g[cell_g["n_raw"] >= 2]

        deltas = []
        for _, row in overlap.iterrows():
            # For each cell, pair up logS values from DIFFERENT raw members and
            # record |Δ|.
            pairs = list(zip(row["raw_list"], row["logS_values"]))
            for i in range(len(pairs)):
                for j in range(i + 1, len(pairs)):
                    if pairs[i][0] != pairs[j][0]:
                        deltas.append(abs(pairs[i][1] - pairs[j][1]))
        deltas = np.array(deltas) if deltas else np.array([])

        per_group_rows.append({
            "canon_C": canon,
            "n_raw_members": len(raw_members),
            "group_total_rows": int(len(sub)),
            "overlap_cells": int(len(overlap)),
            "overlap_pair_count": int(deltas.size),
            "median_abs_delta_logS": float(np.median(deltas)) if deltas.size else None,
            "max_abs_delta_logS": float(np.max(deltas)) if deltas.size else None,
            "p90_abs_delta_logS": float(np.percentile(deltas, 90)) if deltas.size else None,
        })

    pm = pd.DataFrame(per_member_rows)
    pg = pd.DataFrame(per_group_rows)
    pg = pg.sort_values(["group_total_rows"], ascending=False).reset_index(drop=True)

    pm.to_csv(REPORTS / "11_merge_audit_empirical.csv", index=False)
    pg.to_csv(REPORTS / "11_merge_audit_summary.csv", index=False)

    # Summary to stdout
    print("\nPer-group summary (sorted by total rows):")
    print(pg.to_string(index=False))

    # The rows that need HUMAN attention: overlap cells AND max |Δ| > 0.2
    sus = pg[(pg["overlap_pair_count"] > 0) & (pg["max_abs_delta_logS"] > 0.2)]
    print(f"\nMerge groups with overlap AND max |Δ logS| > 0.2 (suspicious): "
          f"{len(sus)}")
    if len(sus):
        print(sus.to_string(index=False))

    # Also: merge groups with NO overlap cells — silent merges (cannot be
    # empirically checked from this data)
    silent = pg[pg["overlap_pair_count"] == 0]
    print(f"\nMerge groups with NO overlap cells (silent merges): {len(silent)}")
    if len(silent):
        print(silent[["canon_C", "n_raw_members", "group_total_rows"]].to_string(index=False))


if __name__ == "__main__":
    main()
