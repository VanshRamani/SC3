"""
Phase 0 — Raw audit of BigSolDB v2.1.

Answers a specific list of questions about the raw data BEFORE any cleaning.
Saves a JSON of findings to reports/01_raw_audit.json and prints a human-
readable summary to stdout.

Questions:
  Q1  Shape, schema, dtypes, NaN counts.
  Q2  Are there exact-duplicate rows?
  Q3  Are there near-duplicate rows within the same DOI (data entry dupes)?
  Q4  Are there cross-DOI exact matches at (solute, solvent, T, logS) — raw
      copycat evidence BEFORE any analysis?
  Q5  SMILES validity (solute, solvent) via RDKit.
  Q6  Temperature range, clustering (is it 5 K lattice + standard-T dominance?).
  Q7  Mole-fraction range validity (0 < x ≤ 1).
  Q8  Consistency: BigSolDB's own logS vs back-computed from mole fraction using
      Coeffs.csv (linear density) — per-row residual distribution.
  Q9  Per-DOI inventory: rows/DOI, solvents/DOI, solutes/DOI.
  Q10 Precision signature: decimals of logS per DOI.
  Q11 Solvent name canonicalization quality: distinct names → same canonical
      SMILES collisions.
  Q12 Solute SMILES collisions: distinct raw SMILES mapping to same canonical
      SMILES (pre-tautomer).
"""
from __future__ import annotations
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw/bigsoldb_v2.1"
OUT = ROOT / "reports/01_raw_audit.json"
OUT.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
def load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(RAW / "BigSolDBv2.1.csv")
    dens = pd.read_csv(RAW / "BigSolDBv2.1_densities.csv")
    coef = pd.read_csv(RAW / "Coeffs.csv")
    return df, dens, coef


def canonical_smiles(smi: str) -> str | None:
    """Non-tautomer canonical SMILES, stereo kept — for canonicalization-quality check."""
    if not isinstance(smi, str) or not smi.strip():
        return None
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return Chem.MolToSmiles(m)


def n_decimals(x: float | str) -> int:
    """Decimal count of a float-valued string — informative about reporting precision."""
    s = str(x)
    if "." not in s:
        return 0
    frac = s.split(".", 1)[1]
    frac = frac.rstrip("0")
    return len(frac)


# ---------------------------------------------------------------------------
def q1_schema(df: pd.DataFrame) -> dict:
    nan = df.isna().sum().to_dict()
    return {
        "n_rows": int(len(df)),
        "n_cols": int(len(df.columns)),
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "nan_counts": {k: int(v) for k, v in nan.items()},
    }


def q2_exact_duplicates(df: pd.DataFrame) -> dict:
    full_mask = df.duplicated(keep=False)
    n_full = int(full_mask.sum())
    key_cols = ["SMILES_Solute", "SMILES_Solvent", "Temperature_K",
                "LogS(mol/L)", "Source"]
    key_mask = df.duplicated(subset=key_cols, keep=False)
    n_key = int(key_mask.sum())
    return {
        "full_row_duplicates_count": n_full,
        "key_tuple_duplicates_count": n_key,  # (solute,solvent,T,logS,DOI)
        "example_key_dupes": df.loc[key_mask].head(4)[key_cols].to_dict("records"),
    }


def q3_intra_doi_near_dupes(df: pd.DataFrame) -> dict:
    """Within the same DOI, same (solute, solvent), T within 0.1 K, logS within 0.001."""
    g = df.sort_values(["Source", "SMILES_Solute", "SMILES_Solvent", "Temperature_K"])
    cnt = 0
    sample = []
    prev = None
    for _, row in g.iterrows():
        if prev is not None:
            same = (row["Source"] == prev["Source"]
                    and row["SMILES_Solute"] == prev["SMILES_Solute"]
                    and row["SMILES_Solvent"] == prev["SMILES_Solvent"])
            if same:
                dT = abs(row["Temperature_K"] - prev["Temperature_K"])
                dL = abs(row["LogS(mol/L)"] - prev["LogS(mol/L)"]) \
                    if pd.notna(row["LogS(mol/L)"]) and pd.notna(prev["LogS(mol/L)"]) else np.inf
                if dT < 0.1 and dL < 0.001:
                    cnt += 1
                    if len(sample) < 5:
                        sample.append({
                            "source": row["Source"],
                            "solute": row["SMILES_Solute"],
                            "solvent": row["Solvent"],
                            "T1": float(prev["Temperature_K"]),
                            "T2": float(row["Temperature_K"]),
                            "logS1": float(prev["LogS(mol/L)"]) if pd.notna(prev["LogS(mol/L)"]) else None,
                            "logS2": float(row["LogS(mol/L)"]) if pd.notna(row["LogS(mol/L)"]) else None,
                        })
        prev = row
    return {"intra_doi_near_dupe_count": cnt, "examples": sample}


def q4_cross_doi_exact_matches(df: pd.DataFrame) -> dict:
    """Cross-DOI: same (solute, solvent, T) reported identically in logS by two DOIs.
    Direct evidence of copying at the raw level — no statistics needed."""
    sub = df.dropna(subset=["LogS(mol/L)"]).copy()
    # Round T to 0.1 K to match "same temperature" loosely.
    sub["T_round"] = sub["Temperature_K"].round(1)
    sub["logS_round"] = sub["LogS(mol/L)"].round(4)
    key = ["SMILES_Solute", "SMILES_Solvent", "T_round", "logS_round"]
    # Group by the key; rows spanning >1 DOI for the same key are exact cross-DOI matches.
    grp = sub.groupby(key)["Source"].nunique()
    multi = grp[grp > 1]
    n_pairs = int((multi - 1).sum())  # upper bound on "cross-DOI copy rows"
    keys_affected = int(len(multi))
    return {
        "cross_doi_exact_match_keys": keys_affected,
        "cross_doi_redundant_rows_upper_bound": n_pairs,
    }


def q5_smiles_validity(df: pd.DataFrame) -> dict:
    solu = df["SMILES_Solute"].dropna().unique()
    solv = df["SMILES_Solvent"].dropna().unique()
    n_bad_solu = sum(1 for s in solu if Chem.MolFromSmiles(s) is None)
    n_bad_solv = sum(1 for s in solv if Chem.MolFromSmiles(s) is None)
    bad_solu = [s for s in solu if Chem.MolFromSmiles(s) is None][:10]
    bad_solv = [s for s in solv if Chem.MolFromSmiles(s) is None][:10]
    return {
        "unique_solutes": int(len(solu)),
        "unique_solvents": int(len(solv)),
        "invalid_solute_smiles": int(n_bad_solu),
        "invalid_solvent_smiles": int(n_bad_solv),
        "examples_invalid_solute": bad_solu,
        "examples_invalid_solvent": bad_solv,
    }


def q6_temperature(df: pd.DataFrame) -> dict:
    T = df["Temperature_K"].dropna()
    # detect 5 K lattice: fraction where T mod 5 is within 0.2 of either 0 or 0.15
    T_mod5 = T % 5
    near5 = ((T_mod5 < 0.2) | (T_mod5 > 4.8)).sum()
    near_015 = ((T_mod5 > 0.10) & (T_mod5 < 0.20)).sum()
    top_T = T.round(2).value_counts().head(10).to_dict()
    return {
        "T_min": float(T.min()),
        "T_max": float(T.max()),
        "T_median": float(T.median()),
        "frac_at_5K_lattice": float(near5 / len(T)),
        "frac_at_xK_plus_0p15": float(near_015 / len(T)),
        "top_10_temperatures": {str(k): int(v) for k, v in top_T.items()},
    }


def q7_mole_fraction(df: pd.DataFrame) -> dict:
    x = df["Solubility(mole_fraction)"].dropna()
    return {
        "n": int(len(x)),
        "x_min": float(x.min()),
        "x_max": float(x.max()),
        "x_leq_0": int((x <= 0).sum()),
        "x_gt_1": int((x > 1).sum()),
        "x_eq_1": int((x == 1).sum()),
        "x_gt_0p5": int((x > 0.5).sum()),
        "x_gt_0p1": int((x > 0.1).sum()),
    }


def q8_logs_consistency(df: pd.DataFrame, coef: pd.DataFrame) -> dict:
    """Back-compute logS from mole fraction using linear density fit a*T + b for
    the solvent, and compare to BigSolDB-reported logS."""
    # Build name -> (a, b) map. Both files use solvent NAME (not canonical SMILES).
    # Lowercase to be safe.
    coef_map = {row["Solvent"].strip().lower(): (row["a"], row["b"]) for _, row in coef.iterrows()}
    mw_cache: dict[str, float] = {}
    resid: list[float] = []
    reasons = {"no_coef": 0, "invalid_smiles": 0, "missing_fields": 0, "ok": 0}
    worst: list[tuple[float, dict]] = []

    for _, row in df.iterrows():
        x = row.get("Solubility(mole_fraction)")
        logs = row.get("LogS(mol/L)")
        solv_name = str(row.get("Solvent", "")).strip().lower()
        solv_smi = row.get("SMILES_Solvent")
        T = row.get("Temperature_K")
        if pd.isna(x) or pd.isna(logs) or pd.isna(T):
            reasons["missing_fields"] += 1
            continue
        if x <= 0 or x > 1:
            reasons["missing_fields"] += 1
            continue
        if solv_name not in coef_map:
            reasons["no_coef"] += 1
            continue
        a, b = coef_map[solv_name]
        rho = a * T + b  # g/cm^3
        if solv_smi not in mw_cache:
            m = Chem.MolFromSmiles(solv_smi) if isinstance(solv_smi, str) else None
            if m is None:
                mw_cache[solv_smi] = float("nan")
            else:
                mw_cache[solv_smi] = sum(a.GetMass() for a in m.GetAtoms()) + \
                                     sum(a.GetTotalNumHs() for a in m.GetAtoms()) * 1.00794
        Mw = mw_cache[solv_smi]
        if not np.isfinite(Mw):
            reasons["invalid_smiles"] += 1
            continue
        # thermodynamically correct: S = x/(1-x) * rho*1000 / Mw
        S = x / (1.0 - x) * rho * 1000.0 / Mw
        if S <= 0:
            reasons["invalid_smiles"] += 1
            continue
        logs_bc = float(np.log10(S))
        r = logs_bc - float(logs)
        resid.append(r)
        reasons["ok"] += 1
        if len(worst) < 20 or abs(r) > abs(worst[-1][0]):
            worst.append((r, {
                "solute": row["SMILES_Solute"], "solvent": solv_name,
                "T": float(T), "x": float(x), "logS_reported": float(logs),
                "logS_back": logs_bc, "residual": r,
                "source": row["Source"],
            }))
            worst.sort(key=lambda t: abs(t[0]), reverse=True)
            worst = worst[:20]
    arr = np.asarray(resid)
    return {
        "reasons": reasons,
        "residual_stats": {
            "n": int(arr.size),
            "median_abs": float(np.median(np.abs(arr))) if arr.size else None,
            "mean_abs": float(np.mean(np.abs(arr))) if arr.size else None,
            "p95_abs": float(np.percentile(np.abs(arr), 95)) if arr.size else None,
            "p99_abs": float(np.percentile(np.abs(arr), 99)) if arr.size else None,
            "max_abs": float(np.max(np.abs(arr))) if arr.size else None,
            "frac_abs_gt_0p01": float(np.mean(np.abs(arr) > 0.01)) if arr.size else None,
            "frac_abs_gt_0p05": float(np.mean(np.abs(arr) > 0.05)) if arr.size else None,
            "frac_abs_gt_0p1": float(np.mean(np.abs(arr) > 0.10)) if arr.size else None,
        },
        "top20_worst_residuals": [w[1] for w in worst],
    }


def q9_doi_inventory(df: pd.DataFrame) -> dict:
    g = df.groupby("Source")
    rows = g.size()
    solv = g["SMILES_Solvent"].nunique()
    solu = g["SMILES_Solute"].nunique()
    return {
        "n_dois": int(df["Source"].nunique()),
        "rows_per_doi": {"median": float(rows.median()), "mean": float(rows.mean()),
                         "min": int(rows.min()), "max": int(rows.max()),
                         "p90": float(rows.quantile(0.9)), "p99": float(rows.quantile(0.99))},
        "solvents_per_doi": {"median": float(solv.median()), "mean": float(solv.mean()),
                             "max": int(solv.max())},
        "solutes_per_doi": {"median": float(solu.median()), "mean": float(solu.mean()),
                            "max": int(solu.max())},
        "top_10_dois_by_rows": rows.sort_values(ascending=False).head(10).to_dict(),
    }


def q10_precision_signature(df: pd.DataFrame) -> dict:
    """Number of decimals in LogS, grouped by DOI. DOIs that mostly report 1-2
    decimals are qualitatively different from ones that report 4+."""
    # We have to read the CSV as strings to preserve decimals. Re-read that column.
    raw = pd.read_csv(RAW / "BigSolDBv2.1.csv", dtype={"LogS(mol/L)": str})
    raw = raw.dropna(subset=["LogS(mol/L)"])
    dec = raw["LogS(mol/L)"].map(n_decimals)
    per_doi_med = raw.assign(dec=dec).groupby("Source")["dec"].median()
    dist = dec.value_counts().sort_index().to_dict()
    return {
        "overall_decimal_distribution": {int(k): int(v) for k, v in dist.items()},
        "per_doi_median_decimals": {
            "median": float(per_doi_med.median()),
            "p10": float(per_doi_med.quantile(0.1)),
            "p90": float(per_doi_med.quantile(0.9)),
        },
        "n_dois_median_le2": int((per_doi_med <= 2).sum()),
        "n_dois_median_ge5": int((per_doi_med >= 5).sum()),
    }


def q11_solvent_name_canon(df: pd.DataFrame) -> dict:
    """Distinct raw solvent NAMES that share a canonical SMILES and distinct
    canonical SMILES that share a name."""
    names = df[["Solvent", "SMILES_Solvent"]].dropna().drop_duplicates()
    names["canon"] = names["SMILES_Solvent"].map(canonical_smiles)
    by_canon = names.groupby("canon")["Solvent"].nunique()
    multi_name = by_canon[by_canon > 1]
    by_name = names.groupby("Solvent")["canon"].nunique()
    multi_canon = by_name[by_name > 1]
    examples = names[names["canon"].isin(multi_name.index)].sort_values("canon").head(20)
    return {
        "unique_solvent_names": int(names["Solvent"].nunique()),
        "unique_solvent_canon": int(names["canon"].nunique()),
        "canon_smiles_with_multiple_names": int(len(multi_name)),
        "solvent_names_with_multiple_canon": int(len(multi_canon)),
        "examples_multiple_names": examples.to_dict("records"),
    }


def q12_solute_collisions(df: pd.DataFrame) -> dict:
    """Different raw SMILES that canonicalize to the same — these are silent
    merges in the dataset (pre-tautomer)."""
    solu = df[["SMILES_Solute", "Compound_Name", "CAS", "PubChem_CID"]].drop_duplicates()
    solu["canon"] = solu["SMILES_Solute"].map(canonical_smiles)
    by_canon = solu.groupby("canon")["SMILES_Solute"].nunique()
    collisions = by_canon[by_canon > 1]
    # also check: same canon but DIFFERENT Compound_Name (the pentoxifylline / tolfenamic case)
    name_by_canon = solu.groupby("canon")["Compound_Name"].nunique()
    name_coll = name_by_canon[name_by_canon > 1]
    examples = solu[solu["canon"].isin(name_coll.index)].sort_values("canon").head(20)
    return {
        "unique_raw_solute_smiles": int(solu["SMILES_Solute"].nunique()),
        "unique_canonical_solute_smiles": int(solu["canon"].nunique()),
        "canon_smiles_with_multiple_raw_smiles": int(len(collisions)),
        "canon_smiles_with_multiple_compound_names": int(len(name_coll)),
        "examples_name_collisions": examples.to_dict("records"),
    }


# ---------------------------------------------------------------------------
def main():
    print("Loading …")
    df, dens, coef = load_raw()
    print(f"  raw df:      {df.shape}")
    print(f"  density df:  {dens.shape}")
    print(f"  coef df:     {coef.shape}")

    out: dict = {}
    print("\nQ1  schema …")
    out["Q1_schema"] = q1_schema(df)
    print("Q2  exact duplicates …")
    out["Q2_exact_duplicates"] = q2_exact_duplicates(df)
    print("Q3  intra-DOI near duplicates …")
    out["Q3_intra_doi_near_dupes"] = q3_intra_doi_near_dupes(df)
    print("Q4  cross-DOI exact matches (raw copycat)…")
    out["Q4_cross_doi_exact"] = q4_cross_doi_exact_matches(df)
    print("Q5  SMILES validity …")
    out["Q5_smiles_validity"] = q5_smiles_validity(df)
    print("Q6  temperature patterns …")
    out["Q6_temperature"] = q6_temperature(df)
    print("Q7  mole fraction validity …")
    out["Q7_mole_fraction"] = q7_mole_fraction(df)
    print("Q8  logS consistency (back-compute) …")
    out["Q8_logs_consistency"] = q8_logs_consistency(df, coef)
    print("Q9  DOI inventory …")
    out["Q9_doi_inventory"] = q9_doi_inventory(df)
    print("Q10 precision signature …")
    out["Q10_precision_signature"] = q10_precision_signature(df)
    print("Q11 solvent-name canonicalization …")
    out["Q11_solvent_name_canon"] = q11_solvent_name_canon(df)
    print("Q12 solute SMILES collisions …")
    out["Q12_solute_collisions"] = q12_solute_collisions(df)

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved → {OUT}")


if __name__ == "__main__":
    main()
