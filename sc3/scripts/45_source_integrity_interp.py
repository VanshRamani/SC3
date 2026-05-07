"""
Phase 3B — Source integrity with Apelblat/van't Hoff interpolation.

Runs AFTER scripts/40_apelblat.py has fit per-preliminary-group curves.
Redoes Stage B (gray-zone copycat detection) and Stage C (DOI reliability)
using interpolated fits on a uniform 1 K reference grid, matching the Phase 5
aleatoric methodology (D-13).

  Stage B' (interpolated gray-zone).  For every pair of preliminary-groups
    that share ≥ 1 (solute, solvent), evaluate both fits on a uniform 1 K
    grid inside their fit-range intersection (≥ 5 K required).  If the
    interpolation-MAE averaged across all shared (solute, solvent) pairs
    (weighted by number of grid points) is < θ_B = 0.01, union the two
    preliminary groups into the same final group.  Catches copycats that
    don't share exact measured temperatures.

  Stage C' (interpolated reliability).  For each preliminary group, find
    every (solute, solvent) where it overlaps in-fit-range with at least
    one OTHER final group.  At each reference T on that pair's overlap
    grid, compute the consensus = mean of the other groups' fit values.
    Mean absolute deviation from consensus across all grid points is the
    group's reliability score.

The Hall of Shame is the set of final groups whose reliability-score ≥ 0.6
log S.  Since Stage B' merges more copycat pairs than shared-T Stage B,
some of v2's originally-flagged HoS DOIs may now be absorbed into larger
final groups (where they agree with their copies and disagree jointly
with the rest of the field).

Inputs:
  data/interim/02_cleaned.csv
  data/interim/03_doi_groups.csv   (preliminary groups from Phase 3 Stage A)
  data/interim/04_fits.csv          (per-preliminary-group Apelblat/vanthoff fits)

Outputs (overwrite Phase 3 originals — these become the authoritative ones):
  data/interim/03_doi_groups.csv   (final groups, DOI → final_group_id)
  reports/30_doi_pair_mae.csv
  reports/30_doi_reliability.csv
  reports/30_source_integrity.json
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict, Counter
from itertools import combinations
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INTERIM = ROOT / "data/interim"
REPORTS = ROOT / "reports"

DT_GRID = 1.0                 # K, must match Phase 5
DT_MIN = 5.0                  # K minimum overlap
BAD_FIT_R2 = 0.80
BAD_FIT_RMSE = 0.30
THETA_B = 0.01                # copycat gray-zone threshold
HOS_THRESHOLD = 0.6


# ── union-find ────────────────────────────────────────────────────────────
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


def _eval(row: pd.Series, T: np.ndarray) -> np.ndarray:
    m = row["model"]
    if m == "apelblat":
        return row["A"] + row["B"] / T + row["C"] * np.log(T)
    if m == "vanthoff":
        return row["A"] + row["B"] / T
    return None


def main():
    df = pd.read_csv(INTERIM / "02_cleaned.csv")
    prelim = pd.read_csv(INTERIM / "03_doi_groups.csv")
    fits = pd.read_csv(INTERIM / "04_fits.csv")

    # Start UF at current preliminary groups (carry over Stage-A bit-exact
    # unions).  Each DOI is a node; we already know its prelim group.
    doi_to_prelim = prelim.set_index("doi")["group_id"].to_dict()
    df["prelim_gid"] = df["Source"].map(doi_to_prelim)

    # UF operates on PRELIM-GROUP ids (not individual DOIs) because within
    # a prelim group all DOIs are already known-equivalent.
    uf = UF()
    for pg in prelim["group_id"].unique():
        uf.find(int(pg))

    # Valid-fit filter
    fits["fit_ok"] = (
        (fits["model"].isin(["apelblat", "vanthoff"]))
        & (fits["R2"].fillna(0) >= BAD_FIT_R2)
        & (fits["RMSE"].fillna(1) <= BAD_FIT_RMSE)
    )
    fit_by_key = {
        (r["Solute_Canon"], r["Solvent_Canon"], int(r["gid"])): r
        for _, r in fits.iterrows() if r["fit_ok"]
    }
    print(f"Loaded {len(df):,} rows, {prelim['group_id'].nunique()} prelim groups, "
          f"{len(fit_by_key):,} usable fits")

    # ═══════════════════════════════════════════════════════════════════
    # Stage B' — interpolated gray-zone detection
    # ═══════════════════════════════════════════════════════════════════
    # For every prelim-group pair that shares a (solute, solvent) pair with
    # usable fits, compute interpolated MAE.
    pair_to_prelim_gids: dict[tuple, set[int]] = defaultdict(set)
    for (solute, solvent, pg) in fit_by_key:
        pair_to_prelim_gids[(solute, solvent)].add(int(pg))

    # Accumulate grid-weighted MAE per (pg_i, pg_j) across all shared pairs
    pg_pair_mae_sum: dict[tuple[int, int], float] = defaultdict(float)
    pg_pair_n_grid: dict[tuple[int, int], int] = defaultdict(int)

    for (solute, solvent), pgs in pair_to_prelim_gids.items():
        if len(pgs) < 2:
            continue
        pgs_sorted = sorted(pgs)
        for pi, pj in combinations(pgs_sorted, 2):
            fi = fit_by_key[(solute, solvent, pi)]
            fj = fit_by_key[(solute, solvent, pj)]
            Tmin = max(float(fi["T_min"]), float(fj["T_min"]))
            Tmax = min(float(fi["T_max"]), float(fj["T_max"]))
            if Tmax - Tmin < DT_MIN:
                continue
            n_pts = int(np.floor((Tmax - Tmin) / DT_GRID)) + 1
            ref_T = Tmin + np.arange(n_pts) * DT_GRID
            y_i = _eval(fi, ref_T)
            y_j = _eval(fj, ref_T)
            deltas = np.abs(y_i - y_j)
            key = (pi, pj)
            pg_pair_mae_sum[key] += float(deltas.sum())
            pg_pair_n_grid[key] += int(len(deltas))

    # Compute weighted MAE per prelim-group pair; apply Stage-B' threshold
    b_pair_rows = []
    new_unions = 0
    for (pi, pj), sumd in pg_pair_mae_sum.items():
        n = pg_pair_n_grid[(pi, pj)]
        if n == 0:
            continue
        wmae = sumd / n
        b_pair_rows.append({"pg_i": pi, "pg_j": pj,
                            "n_grid_pts": n, "weighted_MAE": wmae})
        if wmae < THETA_B:
            if uf.find(pi) != uf.find(pj):
                new_unions += 1
                uf.union(pi, pj)
    b_pair_df = (pd.DataFrame(b_pair_rows)
                 .sort_values("weighted_MAE")
                 .reset_index(drop=True))
    b_pair_df.to_csv(REPORTS / "30_doi_pair_mae.csv", index=False)
    print(f"Stage B' (interpolated MAE < {THETA_B}): {new_unions} new unions "
          f"on top of {prelim['group_id'].nunique()} prelim groups")

    # ═══════════════════════════════════════════════════════════════════
    # Re-assemble final groups (post Stage A + B')
    # ═══════════════════════════════════════════════════════════════════
    prelim_to_final: dict[int, int] = {}
    final_roots: dict[int, list[int]] = defaultdict(list)
    for pg in prelim["group_id"].unique():
        root = uf.find(int(pg))
        final_roots[root].append(int(pg))
    # Assign stable final_gid by descending total DOI count for reproducibility
    prelim_size = prelim.groupby("group_id")["doi"].count().to_dict()
    root_order = sorted(final_roots, key=lambda r: (
        -sum(prelim_size.get(p, 0) for p in final_roots[r]), r))
    for fid, root in enumerate(root_order):
        for pg in final_roots[root]:
            prelim_to_final[pg] = fid

    # DOI → final gid
    df["final_gid"] = df["prelim_gid"].map(prelim_to_final)
    doi_to_final = {}
    for _, r in prelim.iterrows():
        doi_to_final[r["doi"]] = prelim_to_final[int(r["group_id"])]

    n_final = int(df["final_gid"].nunique())
    dois_per_final = df.groupby("final_gid")["Source"].nunique()
    size_hist = dois_per_final.value_counts().sort_index()
    print(f"Final independence groups: {n_final}")
    print(f"  group-size distribution (# of DOIs per final group): {size_hist.to_dict()}")

    # ═══════════════════════════════════════════════════════════════════
    # Stage C' — interpolated reliability
    # ═══════════════════════════════════════════════════════════════════
    # For each (solute, solvent), evaluate each final group's fit on a shared
    # reference grid and compare to consensus of OTHER final groups.
    # Representative fit per final group for (solute, solvent) = the best-R²
    # prelim-group fit (or any if tied).
    best_fit_for_final: dict[tuple, pd.Series] = {}
    for (solute, solvent, pg), r in fit_by_key.items():
        fg = prelim_to_final[pg]
        key = (solute, solvent, fg)
        cur = best_fit_for_final.get(key)
        if cur is None or r["R2"] > cur["R2"]:
            best_fit_for_final[key] = r

    # Build per-(solute, solvent) list of (final_gid, fit) tuples
    ss_to_fgs: dict[tuple, list[tuple[int, pd.Series]]] = defaultdict(list)
    for (solute, solvent, fg), r in best_fit_for_final.items():
        ss_to_fgs[(solute, solvent)].append((fg, r))

    # Per final-group: list of deviations across (solute, solvent, T) points
    fg_devs: dict[int, list[float]] = defaultdict(list)

    for (solute, solvent), fgs in ss_to_fgs.items():
        if len(fgs) < 2:
            continue
        # Common reference range across ALL fgs (intersection of all fit ranges)
        Tmin = max(float(r["T_min"]) for _, r in fgs)
        Tmax = min(float(r["T_max"]) for _, r in fgs)
        if Tmax - Tmin < DT_MIN:
            continue
        n_pts = int(np.floor((Tmax - Tmin) / DT_GRID)) + 1
        ref_T = Tmin + np.arange(n_pts) * DT_GRID
        # Evaluate every fg at every T
        ys = {fg: _eval(r, ref_T) for fg, r in fgs}
        for fg, r in fgs:
            others = [other for other in ys if other != fg]
            if not others:
                continue
            cons = np.mean(np.stack([ys[o] for o in others], axis=0), axis=0)
            devs = np.abs(ys[fg] - cons)
            fg_devs[fg].extend(devs.tolist())

    rel_rows = []
    all_fgs = sorted(set(df["final_gid"].unique()))
    fg_to_n_dois = dois_per_final.to_dict()
    for fg in all_fgs:
        n_dois = int(fg_to_n_dois.get(fg, 0))
        devs = fg_devs.get(fg, [])
        if len(devs) == 0:
            rel_rows.append({
                "final_gid": fg, "n_grid_points": 0,
                "mean_abs_dev_from_consensus": None,
                "median_abs_dev_from_consensus": None,
                "max_abs_dev_from_consensus": None,
                "n_dois_in_group": n_dois,
            })
        else:
            arr = np.asarray(devs)
            rel_rows.append({
                "final_gid": fg,
                "n_grid_points": int(arr.size),
                "mean_abs_dev_from_consensus": float(arr.mean()),
                "median_abs_dev_from_consensus": float(np.median(arr)),
                "max_abs_dev_from_consensus": float(arr.max()),
                "n_dois_in_group": n_dois,
            })
    rel_df = pd.DataFrame(rel_rows)

    # Expand to DOI level for convenience
    doi_rows = []
    for _, r in prelim.iterrows():
        fg = prelim_to_final[int(r["group_id"])]
        gr = rel_df[rel_df["final_gid"] == fg].iloc[0]
        doi_rows.append({
            "doi": r["doi"],
            "final_gid": fg,
            "n_grid_points": int(gr["n_grid_points"]),
            "mean_abs_dev_from_consensus": gr["mean_abs_dev_from_consensus"],
            "median_abs_dev_from_consensus": gr["median_abs_dev_from_consensus"],
            "max_abs_dev_from_consensus": gr["max_abs_dev_from_consensus"],
            "group_n_dois": int(gr["n_dois_in_group"]),
        })
    doi_rel = pd.DataFrame(doi_rows).sort_values(
        "mean_abs_dev_from_consensus", ascending=False, na_position="last"
    )
    doi_rel.to_csv(REPORTS / "30_doi_reliability.csv", index=False)

    # Save final DOI groups (overwrite 03_doi_groups.csv with FINAL groups)
    doi_map = pd.DataFrame([
        {"doi": doi, "group_id": doi_to_final[doi]}
        for doi in sorted(doi_to_final)
    ])
    row_per_doi = df["Source"].value_counts()
    doi_map["n_rows"] = doi_map["doi"].map(row_per_doi).fillna(0).astype(int)
    gsize = df.groupby("final_gid")["Source"].nunique().to_dict()
    doi_map["group_n_dois"] = doi_map["group_id"].map(gsize)
    doi_map.to_csv(INTERIM / "03_doi_groups.csv", index=False)

    tested = doi_rel.dropna(subset=["mean_abs_dev_from_consensus"])
    hof = int((tested["mean_abs_dev_from_consensus"] <= 0.2).sum())
    hos = int((tested["mean_abs_dev_from_consensus"] >= HOS_THRESHOLD).sum())
    print(f"\nStage C' (interpolated reliability):")
    print(f"  DOIs tested: {len(tested):,} / {len(doi_rel):,} "
          f"(untested = groups with no ≥2-group overlap)")
    print(f"  Hall of Fame (≤0.2):  {hof:,} ({100*hof/len(tested):.1f}% of tested)")
    print(f"  Hall of Shame (≥0.6): {hos:,} ({100*hos/len(tested):.1f}% of tested)")

    # summary JSON
    summary = {
        "policy": {
            "DT_GRID_K": DT_GRID,
            "DT_MIN_overlap_K": DT_MIN,
            "theta_B_logS": THETA_B,
            "HOS_threshold_logS": HOS_THRESHOLD,
        },
        "prelim_groups": int(prelim["group_id"].nunique()),
        "stage_B_interpolated_unions": int(new_unions),
        "final_groups": n_final,
        "reliability": {
            "tested_DOIs": int(len(tested)),
            "untested_DOIs": int(len(doi_rel) - len(tested)),
            "hall_of_fame_leq_0p2": hof,
            "hall_of_shame_geq_0p6": hos,
            "median_dev_tested": float(tested["mean_abs_dev_from_consensus"].median())
                if len(tested) else None,
            "mean_dev_tested": float(tested["mean_abs_dev_from_consensus"].mean())
                if len(tested) else None,
        },
    }
    with open(REPORTS / "30_source_integrity.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote final DOI groups → data/interim/03_doi_groups.csv")
    print(f"Wrote reliability → reports/30_doi_reliability.csv")
    print(f"Wrote summary → reports/30_source_integrity.json")


if __name__ == "__main__":
    main()
