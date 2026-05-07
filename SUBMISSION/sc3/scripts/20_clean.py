"""
Phase 2 — Cleaning waterfall.

Applied in order to the canonicalized data (data/interim/01_canonical.csv):

  W1  Drop rows from 10 bad DOIs = v1's 9 + new candidate 10.1016/j.molliq.2017.02.075
  W2  Drop rows with invalid solvent SMILES (polymer rows with "-")
  W3  Drop rows with a '.' in raw solute SMILES (salts / mixtures / co-crystals)
  W4  Drop rows where canonical solute has MW > 1000 Da
  W5  Recover NaN logS from mole fraction for rows with valid
      solvent density coefficients, then drop any remaining NaN
  W6  Drop rows with logS ∉ [-15, 2]
  W7  Deduplicate intra-DOI near-duplicates (same solute / solvent /
      round(T, 1) within one DOI — keep first occurrence)

Waterfall and per-step counts are saved to reports/20_waterfall.json.
Cleaned output saved to data/interim/02_cleaned.csv.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw/bigsoldb_v2.1"
INTERIM = ROOT / "data/interim"
REPORTS = ROOT / "reports"
INTERIM.mkdir(parents=True, exist_ok=True)


# ─── Constants ─────────────────────────────────────────────────────────────
# Bad-DOI list — kept in sync with the co-maintainer's manual audit (manual_corrections_log.txt)
# and Phase 0's logS back-computation residuals (D-03).
BAD_DOIS = {
    # v1 bad DOIs that the co-maintainer also confirmed as outliers (keep dropping):
    "10.1016/j.fluid.2011.09.033",        # the co-maintainer: "suspicious, drop"
    "10.1021/acs.jced.9b00728",            # Bisacodyl outlier (co-maintainer audit)
    "10.1021/acs.jced.4c00179",            # "don't include in train/test" (the co-maintainer)
    "10.1021/acs.jced.6b00009",            # TKX-50 outlier (the co-maintainer)
    "10.1016/j.molliq.2022.119759",        # 5-fluorouracil in THF outlier
    # NEW in v2 from co-maintainer + curator manual audit:
    "10.1021/je4000718",                   # the dataset curator: wrong flavonoid solubility (0.01 vs 1e-6 truth)
    "10.1016/s1004-9541(08)60201-3",       # the co-maintainer: emodin-in-ethanol, drop
    "10.1016/j.molliq.2016.11.036",        # Zhang flavonoids — Phase 0 ε_D residual 2.81 logS
    # NEW in v2 from the Phase 0 logS back-computation residual (D-03):
    "10.1016/j.molliq.2017.02.075",        # β-alanine/methanol, +0.36 logS systematic offset
    # v1's bad DOIs that the co-maintainer provided fixes for — NOW MOVED TO CORRECTION
    # PIPELINE (scripts/15_apply_manual_corrections.py), no longer dropped:
    #   10.1016/j.fluid.2015.07.038       (ethanol ↔ ethyl acetate swap)
    #   10.1016/j.fluid.2013.09.018       (×10 correction)
    #   10.1016/j.molliq.2013.06.011      (×10 correction)
    #   10.1016/j.molliq.2020.113867      (paracetamol/water specific replacements)
}

MW_MAX = 1000.0
LOGS_MIN = -15.0
LOGS_MAX = 2.0

# Solvent-name aliases for the `thermo` library (many long-tail names in
# BigSolDB are not recognized by thermo's chemical search).  Sourced from
# v1's SOLVENT_ALIASES plus additional aliases where needed.
SOLVENT_ALIASES = {
    "THF": "tetrahydrofuran",
    "DMF": "dimethylformamide",
    "DMSO": "dimethyl sulfoxide",
    "DMS": "methylthiomethane",
    "DMAC": "dimethylacetamide",
    "NMP": "1-methyl-2-pyrrolidone",
    "DEF": "diethylformamide",
    "n-heptane": "heptane",
    "n-hexane": "hexane",
    "n-pentane": "pentane",
    "n-octane": "octane",
    "n-decane": "decane",
    "n-propanol": "1-propanol",
    "n-butanol": "1-butanol",
    "n-pentanol": "1-pentanol",
    "n-hexanol": "1-hexanol",
    "n-octanol": "1-octanol",
    "2-ethyl-n-hexanol": "2-ethylhexanol",
    "sec-butanol": "2-butanol",
    "iso-butanol": "2-methyl-1-propanol",
    "isobutanol": "2-methyl-1-propanol",
    "isopropanol": "2-propanol",
    "3,6-dioxa-1-decanol": "butoxyethoxyethanol",
    # v2 additions — swept up the residual 62 unrecoverable rows.
    "\u03b5-caprolactone": "caprolactone",      # Greek epsilon
    "diisobutyl methanol": "108-82-7",           # CAS fallback
    "propanediol butyl ether": "1569-02-4",      # n-butoxy-1,2-propanediol
    "2-methyl-cyclohexyl acetate": "5726-19-2",
}


# ─── helpers ───────────────────────────────────────────────────────────────
def build_density_lookup(densities_path: Path, coeffs_path: Path):
    dens_df = pd.read_csv(densities_path)
    dens_df["Solvent"] = dens_df["Solvent"].astype(str).str.strip().str.lower()
    density_dict: dict = {}
    for _, r in dens_df.iterrows():
        try:
            rho = float(str(r["Density_g/cm^3"]).replace(",", "."))
        except Exception:
            continue
        density_dict[(r["Solvent"], round(float(r["Temperature_K"]), 2))] = rho
    coef_df = pd.read_csv(coeffs_path)
    coef_dict = {r["Solvent"].strip().lower(): (float(r["a"]), float(r["b"]))
                 for _, r in coef_df.iterrows()}
    return density_dict, coef_dict


_MW_CACHE: dict[str, float] = {}
def mw_of(smi: str) -> float | None:
    if smi in _MW_CACHE:
        return _MW_CACHE[smi]
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    mw = Descriptors.MolWt(m) if m is not None else None
    _MW_CACHE[smi] = mw
    return mw


_THERMO_RHO_CACHE: dict[tuple[str, float], float | None] = {}


def _thermo_density(name: str, T: float) -> float | None:
    """Fetch density (g/cm^3) from thermo.chemical.Chemical. Cached."""
    key = (name, round(T, 2))
    if key in _THERMO_RHO_CACHE:
        return _THERMO_RHO_CACHE[key]
    try:
        from thermo.chemical import Chemical
        c = Chemical(name, T=T)
        rho_kgm3 = c.rho
        if rho_kgm3 is not None and rho_kgm3 > 0:
            rho = rho_kgm3 / 1000.0
            _THERMO_RHO_CACHE[key] = rho
            return rho
    except Exception:
        pass
    _THERMO_RHO_CACHE[key] = None
    return None


def density_at(T: float, solvent_name: str,
               density_dict: dict, coef_dict: dict) -> tuple[float | None, str]:
    """Return (density_g_per_cm3, source_tag) or (None, 'unavailable')."""
    name = solvent_name.strip().lower()
    # 1. exact T match
    rho = density_dict.get((name, round(T, 2)))
    if rho is not None:
        return rho, "density_dict_exact"
    # 2. linear coefficient model
    if name in coef_dict:
        a, b = coef_dict[name]
        return a * T + b, "coef_linear"
    # 3. thermo library fallback (with alias substitution)
    alias = SOLVENT_ALIASES.get(solvent_name, solvent_name)
    for try_name in [solvent_name, alias]:
        rho = _thermo_density(try_name, T)
        if rho is not None and rho > 0:
            return rho, "thermo_library"
    return None, "unavailable"


def logs_back(x: float, T: float, solvent_name: str, solvent_canon_smi: str,
              density_dict: dict, coef_dict: dict) -> tuple[float | None, str]:
    if not (0 < x < 1):
        return None, "bad_x"
    rho, src = density_at(T, solvent_name, density_dict, coef_dict)
    if rho is None or rho <= 0:
        return None, f"density_{src}"
    mw = mw_of(solvent_canon_smi)
    if mw is None or not np.isfinite(mw) or mw <= 0:
        return None, "mw_invalid"
    S = x / (1.0 - x) * rho * 1000.0 / mw
    if S <= 0:
        return None, "S_nonpos"
    return float(np.log10(S)), f"ok_{src}"


# ─── main ──────────────────────────────────────────────────────────────────
def main():
    # Input is the manually-corrected canonicalized data (01b), not 01.
    df = pd.read_csv(INTERIM / "01b_manually_corrected.csv")
    density_dict, coef_dict = build_density_lookup(
        RAW / "BigSolDBv2.1_densities.csv", RAW / "Coeffs.csv"
    )

    waterfall = [("Input (canonicalized raw)", len(df))]

    # W1 — bad DOI removal
    mask = df["Source"].isin(BAD_DOIS)
    n_bad = int(mask.sum())
    bad_doi_counts = df[mask]["Source"].value_counts().to_dict()
    df = df[~mask].copy()
    waterfall.append((f"W1  remove {len(BAD_DOIS)} bad DOIs", len(df)))
    print(f"W1  removed {n_bad} rows from {len(BAD_DOIS)} bad DOIs")

    # W2 — invalid solvent SMILES (polymer rows)
    mask = df["Solvent_Canon"].isna()
    n_poly = int(mask.sum())
    poly_names = df[mask]["Solvent"].value_counts().to_dict()
    df = df[~mask].copy()
    waterfall.append(("W2  remove polymer / invalid solvent", len(df)))
    print(f"W2  removed {n_poly} polymer rows ({list(poly_names)})")

    # W3 — salt / mixture solutes
    mask = df["SMILES_Solute"].str.contains(r"\.", na=False)
    n_salt = int(mask.sum())
    df = df[~mask].copy()
    waterfall.append(("W3  remove salts / mixtures", len(df)))
    print(f"W3  removed {n_salt} salt / mixture rows")

    # W4 — MW filter on canonical solute
    print("W4  computing MW…")
    df["MW"] = df["Solute_Canon"].map(mw_of)
    mask = (df["MW"].isna()) | (df["MW"] > MW_MAX)
    n_mw = int(mask.sum())
    df = df[~mask].copy()
    waterfall.append((f"W4  MW ≤ {MW_MAX} Da", len(df)))
    print(f"W4  removed {n_mw} rows (MW > {MW_MAX} or MW compute failed)")

    # W5 — recover NaN logS from mole fraction
    nan_mask = df["LogS(mol/L)"].isna()
    n_nan = int(nan_mask.sum())
    print(f"W5  attempting recovery of {n_nan} NaN logS rows…")
    recovery_sources: dict[str, int] = {}
    recovered = 0
    for idx in df.index[nan_mask]:
        row = df.loc[idx]
        logs, src = logs_back(
            x=row["Solubility(mole_fraction)"],
            T=row["Temperature_K"],
            solvent_name=row["Solvent"],
            solvent_canon_smi=row["Solvent_Canon"],
            density_dict=density_dict,
            coef_dict=coef_dict,
        )
        recovery_sources[src] = recovery_sources.get(src, 0) + 1
        if logs is not None and np.isfinite(logs):
            df.at[idx, "LogS(mol/L)"] = logs
            recovered += 1
    still_nan = int(df["LogS(mol/L)"].isna().sum())
    df = df.dropna(subset=["LogS(mol/L)"]).copy()
    waterfall.append((f"W5  recovered {recovered} / dropped {still_nan} still-NaN", len(df)))
    print(f"W5  recovered {recovered} / {n_nan};  dropped {still_nan} unrecoverable")
    print(f"     recovery source breakdown: {recovery_sources}")

    # W6 — logS range filter
    mask = (df["LogS(mol/L)"] < LOGS_MIN) | (df["LogS(mol/L)"] > LOGS_MAX)
    n_range = int(mask.sum())
    df = df[~mask].copy()
    waterfall.append((f"W6  logS ∈ [{LOGS_MIN}, {LOGS_MAX}]", len(df)))
    print(f"W6  removed {n_range} rows outside logS range")

    # W7 — intra-DOI near-dupe dedupe
    # Key: (Source, Solute_Canon, Solvent_Canon, round(T, 1))
    df["_T_r"] = df["Temperature_K"].round(1)
    before = len(df)
    df = df.drop_duplicates(
        subset=["Source", "Solute_Canon", "Solvent_Canon", "_T_r"], keep="first"
    ).copy()
    df = df.drop(columns=["_T_r"])
    n_dup = before - len(df)
    waterfall.append(("W7  dedupe intra-DOI (T→0.1 K rounding)", len(df)))
    print(f"W7  removed {n_dup} intra-DOI near-duplicates")

    # Final cleaned output
    out_cols = ["SMILES_Solute", "Solute_Canon",
                "Solvent", "SMILES_Solvent", "Solvent_Canon",
                "Temperature_K", "Solubility(mole_fraction)",
                "Solubility(mol/L)", "LogS(mol/L)",
                "MW", "Compound_Name", "CAS", "PubChem_CID",
                "FDA_Approved", "Source"]
    out = df[out_cols].rename(columns={"LogS(mol/L)": "LogS"}).reset_index(drop=True)
    out_path = INTERIM / "02_cleaned.csv"
    out.to_csv(out_path, index=False)

    # Waterfall report
    print(f"\n{'='*70}\nCLEANING WATERFALL\n{'='*70}")
    for label, count in waterfall:
        print(f"  {label:55s} {count:>8,}")
    print(f"\nFinal: {len(out):,} rows  ({out['Solute_Canon'].nunique()} solutes, "
          f"{out['Solvent_Canon'].nunique()} solvents, {out['Source'].nunique()} DOIs, "
          f"T ∈ [{out['Temperature_K'].min():.2f}, {out['Temperature_K'].max():.2f}] K, "
          f"logS ∈ [{out['LogS'].min():.2f}, {out['LogS'].max():.2f}])")
    pair_src = out.groupby(["Solute_Canon", "Solvent_Canon"])["Source"].nunique()
    for n in (1, 2, 3, 5):
        c = int((pair_src >= n).sum())
        print(f"  (solute, solvent) pairs with ≥{n} DOIs: {c:,} ({100*c/len(pair_src):.1f}%)")

    report = {
        "waterfall": [{"step": s, "n_rows": n} for s, n in waterfall],
        "final": {
            "n_rows": int(len(out)),
            "n_solutes": int(out["Solute_Canon"].nunique()),
            "n_solvents": int(out["Solvent_Canon"].nunique()),
            "n_dois": int(out["Source"].nunique()),
            "T_range": [float(out["Temperature_K"].min()),
                        float(out["Temperature_K"].max())],
            "logS_range": [float(out["LogS"].min()), float(out["LogS"].max())],
            "pair_multi_source": {
                f"≥{n}_dois": int((pair_src >= n).sum()) for n in (1, 2, 3, 5)
            },
        },
        "per_step_details": {
            "W1_bad_doi_counts": {k: int(v) for k, v in bad_doi_counts.items()},
            "W2_polymer_solvent_names": {k: int(v) for k, v in poly_names.items()},
            "W5_recovery": {"nan_before": n_nan, "recovered": int(recovered),
                            "dropped_unrecoverable": still_nan,
                            "sources": {k: int(v) for k, v in recovery_sources.items()}},
            "W7_dedup_rows_removed": n_dup,
        },
        "bad_doi_list": sorted(list(BAD_DOIS)),
    }
    with open(REPORTS / "20_waterfall.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote cleaned → {out_path}")
    print(f"Wrote report  → reports/20_waterfall.json")


if __name__ == "__main__":
    main()
