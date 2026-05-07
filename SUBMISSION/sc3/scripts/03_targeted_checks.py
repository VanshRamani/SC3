"""
Phase 0.75 — Targeted follow-ups to specific findings from 01/02.

T1  Full footprint of 10.1016/j.molliq.2017.02.075 (suspicious-residual DOI):
    How many total rows, solvents, solutes? Are only 8 rows problematic or
    does the DOI have a consistent systematic error pattern?

T2  Does v1's canonicalization (TautomerEnumerator + isomericSmiles=False)
    CREATE collisions that were not there in the raw data? I.e., do different
    raw SMILES collapse to the same tautomer-canonical SMILES? This is where
    the (L/D) amino acid merges came from.

T3  How does v1's canonicalization change solute cardinality vs simple
    non-tautomer canonicalization? Quantify difference.

T4  Stereochemistry count: how many raw solutes carry stereo descriptors
    (@, /, \\)? Without canonicalization, this informs stripping decision.

T5  What fraction of data is covered by the 88 cross-DOI exact-match rows?
    i.e. at the row level (not key level), how many raw rows are involved
    across those 88 keys?
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import Descriptors

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw/bigsoldb_v2.1"
OUT = ROOT / "reports/03_targeted.json"

_tenum = rdMolStandardize.TautomerEnumerator()


def canon_plain(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    return Chem.MolToSmiles(m) if m is not None else None


def canon_plain_no_stereo(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    return Chem.MolToSmiles(m, isomericSmiles=False) if m is not None else None


def canon_v1(smi: str) -> str | None:
    """Tautomer-enumerated canonical, no stereo. Matches v1's canonicalize_smiles."""
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    if m is None:
        return None
    try:
        cm = _tenum.Canonicalize(m)
        return Chem.MolToSmiles(cm, isomericSmiles=False)
    except Exception:
        return None


def t1_suspect_doi(df: pd.DataFrame) -> dict:
    doi = "10.1016/j.molliq.2017.02.075"
    sub = df[df["Source"] == doi]
    solutes = sub["SMILES_Solute"].unique().tolist()
    solvents = sub["Solvent"].unique().tolist()
    by_pair = (sub.groupby(["SMILES_Solute", "Solvent"])
                  .agg(n=("Temperature_K", "size"),
                       T_min=("Temperature_K", "min"),
                       T_max=("Temperature_K", "max"),
                       logS_min=("LogS(mol/L)", "min"),
                       logS_max=("LogS(mol/L)", "max"))
                  .reset_index())
    return {
        "n_rows": int(len(sub)),
        "n_solutes": int(sub["SMILES_Solute"].nunique()),
        "n_solvents": int(sub["Solvent"].nunique()),
        "solutes": solutes,
        "solvents": solvents,
        "per_pair": by_pair.to_dict("records"),
    }


def t2_canon_collisions(df: pd.DataFrame) -> dict:
    smi_unique = df["SMILES_Solute"].dropna().unique()
    rows = []
    for s in smi_unique:
        rows.append({"raw": s,
                     "canon": canon_plain(s),
                     "canon_no_stereo": canon_plain_no_stereo(s),
                     "canon_v1_tautomer": canon_v1(s)})
    out = pd.DataFrame(rows)
    n_raw = out["raw"].nunique()
    n_plain = out["canon"].nunique()
    n_nostereo = out["canon_no_stereo"].nunique()
    n_v1 = out["canon_v1_tautomer"].nunique()

    # Collisions introduced ONLY by stereo stripping (i.e., raw differed but
    # no-stereo canon matches):
    gstereo = out.groupby("canon_no_stereo")["raw"].nunique()
    stereo_merges = gstereo[gstereo > 1].index.tolist()
    stereo_merge_examples = out[out["canon_no_stereo"].isin(stereo_merges)] \
        .sort_values("canon_no_stereo").head(40).to_dict("records")

    # Collisions introduced by v1's tautomer step (beyond no-stereo):
    gtaut = out.groupby("canon_v1_tautomer")["canon_no_stereo"].nunique()
    taut_merges = gtaut[gtaut > 1].index.tolist()
    taut_merge_examples = out[out["canon_v1_tautomer"].isin(taut_merges)] \
        .sort_values("canon_v1_tautomer").head(40).to_dict("records")

    # Ultimately, how many unique molecules does v1 see vs plain canonical?
    return {
        "unique_raw": int(n_raw),
        "unique_plain_canon": int(n_plain),
        "unique_no_stereo": int(n_nostereo),
        "unique_v1_tautomer_no_stereo": int(n_v1),
        "merges_from_stereo_stripping": int(n_plain - n_nostereo),
        "merges_from_tautomer_on_top": int(n_nostereo - n_v1),
        "stereo_merge_groups_count": int(len(stereo_merges)),
        "tautomer_merge_groups_count": int(len(taut_merges)),
        "stereo_merge_examples": stereo_merge_examples,
        "tautomer_merge_examples": taut_merge_examples,
    }


def t4_stereo_count(df: pd.DataFrame) -> dict:
    smi = df["SMILES_Solute"].dropna().unique()
    n = 0
    n_at = 0
    n_slash = 0
    for s in smi:
        has_at = "@" in s
        has_slash = ("/" in s) or ("\\" in s)
        if has_at or has_slash:
            n += 1
        if has_at:
            n_at += 1
        if has_slash:
            n_slash += 1
    return {
        "unique_solutes": int(len(smi)),
        "with_any_stereo": int(n),
        "with_@_chirality": int(n_at),
        "with_/\\_geometry": int(n_slash),
        "frac_with_stereo": float(n / len(smi)),
    }


def t5_crossdoi_row_coverage(df: pd.DataFrame) -> dict:
    sub = df.dropna(subset=["LogS(mol/L)"]).copy()
    sub["T_r"] = sub["Temperature_K"].round(1)
    sub["logS_r"] = sub["LogS(mol/L)"].round(4)
    key = ["SMILES_Solute", "SMILES_Solvent", "T_r", "logS_r"]
    grp = sub.groupby(key)["Source"].nunique()
    multi_keys = grp[grp > 1].reset_index()
    # How many raw rows lie at those keys?
    mask = sub.set_index(key).index.isin(multi_keys.set_index(key).index)
    n_rows = int(mask.sum())
    rows_by_doi = sub[mask]["Source"].value_counts().head(20).to_dict()
    return {
        "n_keys_with_cross_doi_match": int(len(multi_keys)),
        "n_rows_involved_total": n_rows,
        "frac_of_all_rows": float(n_rows / len(sub)),
        "top_20_dois_by_copycat_rowcount": rows_by_doi,
    }


def main():
    df = pd.read_csv(RAW / "BigSolDBv2.1.csv")
    out: dict = {}
    print("T1 suspect DOI footprint …")
    out["T1_suspect_doi"] = t1_suspect_doi(df)
    print("T2 canonicalization collisions (compare plain / no-stereo / v1) …")
    out["T2_canon_collisions"] = t2_canon_collisions(df)
    print("T4 stereo prevalence …")
    out["T4_stereo"] = t4_stereo_count(df)
    print("T5 cross-DOI exact-match row coverage …")
    out["T5_crossdoi_rowcount"] = t5_crossdoi_row_coverage(df)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Saved → {OUT}")


if __name__ == "__main__":
    main()
