"""
Phase 0.5 — Deeper audit on suspicious patterns surfaced by 01_raw_audit.

D1  Verify pentoxifylline/tolfenamic case: row count, CAS match, per-DOI.
D2  Cross-DOI exact-match pairs: which DOIs are involved? Are they same-author?
D3  DOI-level systematic logS residual: are there OTHER DOIs (beyond the 8
    β-alanine/methanol rows from 10.1016/j.molliq.2017.02.075) with systematic
    back-computation errors >0.01? Could reveal new bad-DOI candidates.
D4  Intra-DOI near-duplicate deeper look.
D5  Cross-check v1 bad-DOI list against data: are all 9 really bad in the data
    we see? Any with no rows (already-removed by upstream)?
D6  What does the *mole fraction* precision signature look like? Since BigSolDB
    logS is computed, mole fraction may still carry per-DOI decimal signatures.
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw/bigsoldb_v2.1"
OUT = ROOT / "reports/02_deep_audit.json"

V1_BAD_DOIS = {
    "10.1021/acs.jced.4c00179",
    "10.1021/acs.jced.9b00728",
    "10.1021/acs.jced.6b00009",
    "10.1016/j.molliq.2022.119759",
    "10.1016/j.fluid.2011.09.033",
    "10.1016/j.fluid.2013.09.018",
    "10.1016/j.molliq.2013.06.011",
    "10.1016/j.molliq.2020.113867",
    "10.1016/j.fluid.2015.07.038",
}


def n_decimals(x) -> int:
    s = str(x)
    if "." not in s:
        return 0
    return len(s.split(".", 1)[1].rstrip("0"))


def load_all():
    df = pd.read_csv(RAW / "BigSolDBv2.1.csv")
    coef = pd.read_csv(RAW / "Coeffs.csv")
    coef_map = {row["Solvent"].strip().lower(): (row["a"], row["b"])
                for _, row in coef.iterrows()}
    return df, coef_map


def d1_pentoxifylline(df: pd.DataFrame) -> dict:
    target_smi = "Cc1c(Cl)cccc1Nc1ccccc1C(=O)O"
    sub = df[df["SMILES_Solute"] == target_smi]
    name_counts = sub["Compound_Name"].value_counts().to_dict()
    cas = sub["CAS"].value_counts().to_dict()
    cid = sub["PubChem_CID"].dropna().value_counts().to_dict()
    doi_name = (sub.groupby("Source")["Compound_Name"]
                   .agg(lambda x: Counter(x).most_common(1)[0][0]))
    return {
        "row_count": int(len(sub)),
        "compound_names_counts": {k: int(v) for k, v in name_counts.items()},
        "cas_counts": {str(k): int(v) for k, v in cas.items()},
        "pubchem_cid_counts": {str(k): int(v) for k, v in cid.items()},
        "n_distinct_dois": int(sub["Source"].nunique()),
        "dois_by_name_vote": doi_name.value_counts().to_dict(),
    }


def d2_cross_doi_exact(df: pd.DataFrame) -> dict:
    sub = df.dropna(subset=["LogS(mol/L)"]).copy()
    sub["T_r"] = sub["Temperature_K"].round(1)
    sub["logS_r"] = sub["LogS(mol/L)"].round(4)
    key = ["SMILES_Solute", "SMILES_Solvent", "T_r", "logS_r"]
    grp = sub.groupby(key).agg(
        n_rows=("Source", "size"),
        n_dois=("Source", "nunique"),
        dois=("Source", lambda x: sorted(set(x))),
    ).reset_index()
    copycat_keys = grp[grp["n_dois"] > 1]
    # Build a DOI-pair co-occurrence count across these copycat rows.
    pair_counter: Counter = Counter()
    for dois in copycat_keys["dois"]:
        for i in range(len(dois)):
            for j in range(i + 1, len(dois)):
                pair_counter[(dois[i], dois[j])] += 1
    top_pairs = [{"doi_a": a, "doi_b": b, "n_exact_matches": int(c)}
                 for (a, b), c in pair_counter.most_common(20)]
    # How many DOIs appear in ANY cross-DOI exact-match?
    dois_in_matches = set()
    for dois in copycat_keys["dois"]:
        dois_in_matches.update(dois)
    return {
        "n_copycat_keys": int(len(copycat_keys)),
        "n_dois_involved": int(len(dois_in_matches)),
        "top_20_doi_pairs": top_pairs,
    }


def d3_systematic_logs_residual(df: pd.DataFrame, coef_map: dict) -> dict:
    """For every DOI, compute the distribution of |residual| (back-computed vs
    reported logS). Flag DOIs where the MEDIAN residual > 0.01."""
    mw_cache: dict = {}

    def get_mw(smi: str) -> float | None:
        if smi in mw_cache:
            return mw_cache[smi]
        m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        mw = Descriptors.MolWt(m) if m is not None else None
        mw_cache[smi] = mw
        return mw

    per_doi: dict[str, list[float]] = {}
    for _, row in df.iterrows():
        x = row.get("Solubility(mole_fraction)")
        logs = row.get("LogS(mol/L)")
        T = row.get("Temperature_K")
        solv = str(row.get("Solvent", "")).strip().lower()
        solv_smi = row.get("SMILES_Solvent")
        doi = row["Source"]
        if pd.isna(x) or pd.isna(logs) or pd.isna(T) or x <= 0 or x >= 1:
            continue
        if solv not in coef_map:
            continue
        a, b = coef_map[solv]
        rho = a * T + b
        mw = get_mw(solv_smi) if isinstance(solv_smi, str) else None
        if mw is None or not np.isfinite(mw):
            continue
        S = x / (1.0 - x) * rho * 1000.0 / mw
        if S <= 0:
            continue
        r = float(np.log10(S) - logs)
        per_doi.setdefault(doi, []).append(r)

    results = []
    for doi, rs in per_doi.items():
        arr = np.asarray(rs)
        abs_arr = np.abs(arr)
        results.append({
            "doi": doi,
            "n": int(arr.size),
            "median_abs_residual": float(np.median(abs_arr)),
            "mean_abs_residual": float(np.mean(abs_arr)),
            "p95_abs_residual": float(np.percentile(abs_arr, 95)) if arr.size > 1 else float(abs_arr[0]),
            "max_abs_residual": float(np.max(abs_arr)),
            "mean_residual": float(np.mean(arr)),  # sign-preserving, detects systematic bias
        })
    dfr = pd.DataFrame(results).sort_values("median_abs_residual", ascending=False)
    suspicious = dfr[dfr["median_abs_residual"] > 0.01]
    return {
        "n_dois_tested": int(len(dfr)),
        "n_suspicious_dois_median_gt_0p01": int(len(suspicious)),
        "top_20_suspicious": suspicious.head(20).to_dict("records"),
    }


def d4_intra_doi_near_dupes_detail(df: pd.DataFrame) -> dict:
    """Which DOIs are responsible for intra-DOI near-dupes? Are they 1 DOI or spread?"""
    g = df.sort_values(["Source", "SMILES_Solute", "SMILES_Solvent", "Temperature_K"])
    per_doi = Counter()
    prev = None
    rows = []
    for _, row in g.iterrows():
        if prev is not None:
            same = (row["Source"] == prev["Source"]
                    and row["SMILES_Solute"] == prev["SMILES_Solute"]
                    and row["SMILES_Solvent"] == prev["SMILES_Solvent"])
            if same:
                dT = abs(row["Temperature_K"] - prev["Temperature_K"])
                dL = (abs(row["LogS(mol/L)"] - prev["LogS(mol/L)"])
                      if pd.notna(row["LogS(mol/L)"]) and pd.notna(prev["LogS(mol/L)"])
                      else np.inf)
                if dT < 0.1 and dL < 0.001:
                    per_doi[row["Source"]] += 1
                    rows.append({
                        "doi": row["Source"],
                        "solute": row["SMILES_Solute"],
                        "solvent": row["Solvent"],
                        "T1": float(prev["Temperature_K"]),
                        "T2": float(row["Temperature_K"]),
                    })
        prev = row
    return {
        "n_total_near_dupes": int(sum(per_doi.values())),
        "per_doi_counts": dict(per_doi),
        "sample_rows": rows[:20],
    }


def d5_bad_doi_audit(df: pd.DataFrame) -> dict:
    out = []
    for doi in sorted(V1_BAD_DOIS):
        sub = df[df["Source"] == doi]
        if len(sub) == 0:
            out.append({"doi": doi, "n_rows": 0, "in_dataset": False})
            continue
        out.append({
            "doi": doi,
            "n_rows": int(len(sub)),
            "n_solutes": int(sub["SMILES_Solute"].nunique()),
            "n_solvents": int(sub["Solvent"].nunique()),
            "T_range": [float(sub["Temperature_K"].min()),
                        float(sub["Temperature_K"].max())],
            "logS_range": [float(sub["LogS(mol/L)"].min()) if sub["LogS(mol/L)"].notna().any() else None,
                           float(sub["LogS(mol/L)"].max()) if sub["LogS(mol/L)"].notna().any() else None],
            "solvents": sorted(sub["Solvent"].unique().tolist()),
        })
    return {"v1_bad_dois_found": out,
            "n_v1_bad_total_rows": int(df["Source"].isin(V1_BAD_DOIS).sum())}


def d6_mole_fraction_precision(df: pd.DataFrame) -> dict:
    """Read mole-fraction column AS STRING and compute decimals per DOI.
    This should be a raw reported value (not a derived quantity), so precision
    may differentiate sources."""
    raw = pd.read_csv(RAW / "BigSolDBv2.1.csv", dtype={"Solubility(mole_fraction)": str})
    raw = raw.dropna(subset=["Solubility(mole_fraction)"])
    dec = raw["Solubility(mole_fraction)"].map(n_decimals)
    per_doi = raw.assign(dec=dec).groupby("Source")["dec"].agg(["median", "min", "max"])
    dist = dec.value_counts().sort_index().to_dict()
    return {
        "overall_decimal_distribution": {int(k): int(v) for k, v in dist.items()},
        "per_doi_median": {
            "median": float(per_doi["median"].median()),
            "p10": float(per_doi["median"].quantile(0.1)),
            "p90": float(per_doi["median"].quantile(0.9)),
        },
        "n_dois_median_le3": int((per_doi["median"] <= 3).sum()),
        "n_dois_median_ge5": int((per_doi["median"] >= 5).sum()),
        "n_dois_median_ge7": int((per_doi["median"] >= 7).sum()),
    }


def main():
    print("Loading…")
    df, coef_map = load_all()
    out: dict = {}
    print("D1 pentoxifylline/tolfenamic verification …")
    out["D1_pentoxifylline_case"] = d1_pentoxifylline(df)
    print("D2 cross-DOI exact match analysis …")
    out["D2_cross_doi_pairs"] = d2_cross_doi_exact(df)
    print("D3 systematic logS residual per DOI …")
    out["D3_logs_residual_per_doi"] = d3_systematic_logs_residual(df, coef_map)
    print("D4 intra-DOI near-dupe detail …")
    out["D4_intra_doi_dupes"] = d4_intra_doi_near_dupes_detail(df)
    print("D5 v1 bad-DOI audit …")
    out["D5_v1_bad_dois"] = d5_bad_doi_audit(df)
    print("D6 mole-fraction precision signature …")
    out["D6_mole_frac_precision"] = d6_mole_fraction_precision(df)

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved → {OUT}")


if __name__ == "__main__":
    main()
