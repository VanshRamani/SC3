"""
Phase 3 — Source integrity, Stage A only (bit-exact copycat detection).

Stages B (gray zone copycats) and C (DOI reliability ranking) are done AFTER
Apelblat fitting, with interpolation, in scripts/45_source_integrity_interp.py.

Input:  data/interim/02_cleaned.csv
Output: data/interim/03_doi_groups.csv   (preliminary DOI → group_id)
        reports/30_stageA_summary.json

Rationale.  Stage A uses bit-exact match at
  (Solute_Canon, Solvent_Canon, round(T, 2), round(LogS, 4))
which requires no interpolation.  Stages B + C need interpolation to catch
copycat DOIs that don't share exact temperatures (and to rank reliability
fairly against any peer group, not only peers that happen to overlap in T).

Interpolation is required because many duplicate/copycat sources report the
same solute-solvent pair at nearby but non-identical temperatures.
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

THETA_GRAY = 0.01  # Stage B threshold


# ── union-find ─────────────────────────────────────────────────────────────
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

    def groups(self) -> dict[str, list[str]]:
        g: dict[str, list[str]] = defaultdict(list)
        for k in list(self.p):
            g[self.find(k)].append(k)
        return {r: sorted(v) for r, v in g.items()}


def main():
    df = pd.read_csv(INTERIM / "02_cleaned.csv")
    print(f"Loaded {len(df):,} rows ({df['Source'].nunique()} DOIs)")

    # Ensure typed columns
    df["T_r"] = df["Temperature_K"].round(2)
    df["logS_r"] = df["LogS"].round(4)

    all_dois = sorted(df["Source"].unique())
    uf = UF()
    for d in all_dois:
        uf.find(d)

    # ═══════════════════════════════════════════════════════════════════
    # Stage A — bit-exact copycats
    # ═══════════════════════════════════════════════════════════════════
    key = ["Solute_Canon", "Solvent_Canon", "T_r", "logS_r"]
    grp_A = (df.groupby(key)["Source"]
               .agg(lambda s: sorted(set(s)))
               .reset_index(name="dois"))
    grp_A["n_dois"] = grp_A["dois"].map(len)
    hits_A = grp_A[grp_A["n_dois"] >= 2]
    a_unions = 0
    for dois in hits_A["dois"]:
        root = dois[0]
        for d in dois[1:]:
            if uf.find(root) != uf.find(d):
                a_unions += 1
            uf.union(root, d)
    print(f"Stage A  bit-exact hits: {len(hits_A)} keys; unions: {a_unions}")
    print(f"         rows touched:   {int((df.set_index(key).index.isin(hits_A.set_index(key).index)).sum())}")

    # Note.  Stages B (gray-zone merging) and C (DOI reliability) are done
    # AFTER Apelblat fits, using interpolation, in scripts/45_source_integrity_interp.py.

    # ═══════════════════════════════════════════════════════════════════
    # Assemble preliminary DOI independence groups (Stage-A merges only)
    # ═══════════════════════════════════════════════════════════════════
    groups = uf.groups()
    # Re-key groups by a stable integer id in descending-size order.
    groups_list = sorted(groups.values(), key=lambda g: (-len(g), g[0]))
    doi_to_gid: dict[str, int] = {}
    for gid, g in enumerate(groups_list):
        for d in g:
            doi_to_gid[d] = gid
    size_hist = pd.Series([len(g) for g in groups_list]).value_counts().sort_index()
    print(f"Total independence groups: {len(groups_list)}")
    print(f"  group-size distribution: {size_hist.to_dict()}")
    # biggest groups
    print("  top-10 biggest independence groups (size / first DOI):")
    for g in groups_list[:10]:
        print(f"    {len(g):>3}  {g[0]}")

    # Save DOI group map
    doi_map = pd.DataFrame({
        "doi": sorted(doi_to_gid),
        "group_id": [doi_to_gid[d] for d in sorted(doi_to_gid)],
    })
    # attach group size and per-DOI row count
    row_per_doi = df["Source"].value_counts()
    doi_map["n_rows"] = doi_map["doi"].map(row_per_doi).fillna(0).astype(int)
    gsize = {gid: sum(row_per_doi.get(d, 0) for d in g) for gid, g in enumerate(groups_list)}
    doi_map["group_n_rows"] = doi_map["group_id"].map(gsize)
    doi_map["group_n_dois"] = doi_map["group_id"].map(
        {gid: len(g) for gid, g in enumerate(groups_list)}
    )
    doi_map.to_csv(INTERIM / "03_doi_groups.csv", index=False)
    print(f"Wrote DOI group map → data/interim/03_doi_groups.csv")

    # ═══════════════════════════════════════════════════════════════════
    # Summary JSON — Stage A only
    # ═══════════════════════════════════════════════════════════════════
    report = {
        "stage": "A_bit_exact_only",
        "input": {"rows": int(len(df)), "dois": int(len(all_dois))},
        "stage_A_bit_exact": {
            "hit_keys": int(len(hits_A)),
            "doi_unions": int(a_unions),
        },
        "preliminary_groups": {
            "count": int(len(groups_list)),
            "size_histogram": {int(k): int(v) for k, v in size_hist.items()},
            "top_10_sizes": [len(g) for g in groups_list[:10]],
        },
        "note": ("Stages B and C run in scripts/45_source_integrity_interp.py "
                 "AFTER Apelblat fits, using interpolation."),
    }
    with open(REPORTS / "30_stageA_summary.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote Stage-A summary → reports/30_stageA_summary.json")


if __name__ == "__main__":
    main()
