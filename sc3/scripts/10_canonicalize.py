"""
Phase 1 — Canonicalization (Option D).

Rules (per decision D-01 on 2026-04-17):
  Solute:  plain RDKit canonical with isomericSmiles=True.  NO tautomer
           enumeration.  NO chirality stripping.  NO geometric-isomer
           stripping.  Every stereoisomer in BigSolDB is kept distinct.
  Solvent: plain canonical (isomericSmiles=True).

Rationale.  Option C's empirical audit (scripts/11_merge_audit.py) showed
10 of 15 Option-C merge groups disagreeing by 0.3–1.3 log S at matched
(solvent, T) — 5–20× the aleatoric floor — so chirality stripping destroys
real variance we cannot recover.  See DECISIONS.md §D-01.

Implementation: just MolToSmiles(mol, isomericSmiles=True) on both sides.

Artifacts:
  - data/interim/01_canonical.csv: raw rows with added Solute_Canon / Solvent_Canon
  - reports/10_canonicalization.json: cardinalities, sanity checks
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw/bigsoldb_v2.1"
INTERIM = ROOT / "data/interim"
REPORTS = ROOT / "reports"
INTERIM.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
def canonicalize_solute(smi: str) -> tuple[str | None, bool, bool]:
    """
    Option D canonicalization: plain canonical with all stereo preserved.
    Returns (canonical_smiles, had_chirality, had_geometry) for auditing.
    """
    if not isinstance(smi, str) or not smi.strip():
        return None, False, False
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None, False, False
    had_chi = any(a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for a in m.GetAtoms())
    had_geo = any(b.GetStereo() in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ,
                                    Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS)
                  for b in m.GetBonds())
    canon = Chem.MolToSmiles(m, isomericSmiles=True)
    return canon, had_chi, had_geo


def canonicalize_solvent(smi: str) -> str | None:
    if not isinstance(smi, str) or not smi.strip():
        return None
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return Chem.MolToSmiles(m, isomericSmiles=True)


# ---------------------------------------------------------------------------
def main():
    df = pd.read_csv(RAW / "BigSolDBv2.1.csv")
    raw_solutes = df["SMILES_Solute"].dropna().unique()
    raw_solvents = df["SMILES_Solvent"].dropna().unique()
    print(f"Raw solute  SMILES: {len(raw_solutes)}")
    print(f"Raw solvent SMILES: {len(raw_solvents)}")

    # Solute canonicalization table
    solute_tbl = []
    for s in raw_solutes:
        canon, had_chi, had_geo = canonicalize_solute(s)
        solute_tbl.append({"raw": s, "canon": canon,
                           "had_chirality": had_chi, "had_geometry": had_geo})
    sdf = pd.DataFrame(solute_tbl)
    sdf["parse_failed"] = sdf["canon"].isna()

    n_parse_fail = int(sdf["parse_failed"].sum())
    n_canon = int(sdf["canon"].nunique())
    print(f"Unique Option-D canonical solutes: {n_canon} (parse fails: {n_parse_fail})")

    # Under Option D we expect zero merges (no policy collapses distinct SMILES).
    merges = (sdf.dropna(subset=["canon"])
                 .groupby("canon")["raw"].nunique())
    n_merges = int((merges > 1).sum())
    print(f"Merge groups (canon with >1 raw): {n_merges}  (expected: 0)")

    # Solvent canonicalization
    solv_tbl = []
    for s in raw_solvents:
        solv_tbl.append({"raw": s, "canon": canonicalize_solvent(s)})
    vdf = pd.DataFrame(solv_tbl)
    n_solv_fail = int(vdf["canon"].isna().sum())
    n_solv_canon = int(vdf["canon"].nunique())
    print(f"Unique solvent canonical SMILES: {n_solv_canon} (parse fails: {n_solv_fail})")

    # Apply to full dataframe and save
    canon_solute_map = dict(zip(sdf["raw"], sdf["canon"]))
    canon_solvent_map = dict(zip(vdf["raw"], vdf["canon"]))
    df["Solute_Canon"] = df["SMILES_Solute"].map(canon_solute_map)
    df["Solvent_Canon"] = df["SMILES_Solvent"].map(canon_solvent_map)

    out_path = INTERIM / "01_canonical.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote canonicalized raw → {out_path}")

    # Sanity checks: distinct molecules stay distinct; same-molecule stays same.
    checks = {}
    samples = {
        "fumaric (trans)": "O=C(O)/C=C/C(=O)O",
        "maleic (cis)":    "O=C(O)/C=C\\C(=O)O",
        "L-tryptophan":    "N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O",
        "D-tryptophan":    "N[C@H](Cc1c[nH]c2ccccc12)C(=O)O",
        "tolfenamic acid": "Cc1c(Cl)cccc1Nc1ccccc1C(=O)O",
    }
    for name, smi in samples.items():
        canon, had_chi, had_geo = canonicalize_solute(smi)
        checks[name] = {"raw": smi, "canon": canon,
                        "had_chirality": had_chi, "had_geometry": had_geo}
    print("\nSanity checks (Option D):")
    for name, c in checks.items():
        print(f"  {name:<20s} → {c['canon']}")
    fm_distinct = checks["fumaric (trans)"]["canon"] != checks["maleic (cis)"]["canon"]
    ld_distinct = checks["L-tryptophan"]["canon"] != checks["D-tryptophan"]["canon"]
    print(f"\n  fumaric ≠ maleic:     {fm_distinct}")
    print(f"  L-Trp   ≠ D-Trp:      {ld_distinct}")

    # Report
    report = {
        "policy": ("Option D: plain RDKit canonical with isomericSmiles=True. "
                   "No tautomer enumeration, no chirality stripping, no "
                   "geometric-isomer stripping."),
        "raw_unique_solutes": int(len(raw_solutes)),
        "canon_unique_solutes": n_canon,
        "solute_parse_failures": n_parse_fail,
        "merge_groups_count": n_merges,
        "raw_unique_solvents": int(len(raw_solvents)),
        "canon_unique_solvents": n_solv_canon,
        "solvent_parse_failures": n_solv_fail,
        "sanity_checks": checks,
        "fumaric_vs_maleic_distinct": bool(fm_distinct),
        "L_vs_D_tryptophan_distinct": bool(ld_distinct),
    }
    with open(REPORTS / "10_canonicalization.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nWrote summary → reports/10_canonicalization.json")


if __name__ == "__main__":
    main()
