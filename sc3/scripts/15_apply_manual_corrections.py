"""
Phase 1.5 — Apply the co-maintainer's manual corrections to the canonicalized raw data.

Source: manual_corrections_log.txt — manual audit log produced by the
BigSolDB v2.1 co-maintainer for the suspicious DOIs that Phase 3 flagged
as the Hall of Shame.

Corrections applied BEFORE the cleaning waterfall (W1…W7 in scripts/20_clean.py).

Correction classes:
  C1  Row-specific value replacements (paracetamol/water — specific mole
      fractions given).
  C2  Bulk value multiplication (×10 on logS means add +1.0 log S, for
      2 DOIs that were off by a decade).
  C3  Solvent-label swap (one DOI had ethanol ↔ ethyl acetate transposed).

Group B DOIs (confirmed outliers) are handled by the bad-DOI list in
scripts/20_clean.py, not here.

Input:  data/interim/01_canonical.csv
Output: data/interim/01b_manually_corrected.csv
        reports/15_manual_corrections.json
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent
INTERIM = ROOT / "data/interim"
REPORTS = ROOT / "reports"


# ── C1 — paracetamol/water specific values (manual correction, 2026-03-26) ────
# Paracetamol canonical SMILES
PARACETAMOL_RAW = "CC(=O)Nc1ccc(O)cc1"  # will canonicalize to check
PARACETAMOL_CANON = Chem.MolToSmiles(Chem.MolFromSmiles(PARACETAMOL_RAW),
                                      isomericSmiles=True)

WATER_CANON = "O"

# the co-maintainer's corrected mole fractions at (T, x)
PARACETAMOL_WATER_CORRECTIONS = {
    293.15: 0.0012,
    303.15: 0.0088,
    313.15: 0.00295,
    323.15: 0.0043,
}


# ── C2 — bulk ×10 multiplications (add +1.0 to logS) ─────────────────────
# manual correction, 2026-03-26
BULK_10X_DOIS = {
    "10.1016/j.fluid.2013.09.018",
    "10.1016/j.molliq.2013.06.011",
}


# ── C3 — solvent-label swap (ethanol ↔ ethyl acetate) ────────────────────
# manual correction, 2026-03-26
# For 10.1016/j.fluid.2015.07.038: all ethanol rows become ethyl acetate and
# all ethyl acetate rows become ethanol.
SWAP_SOLVENT_DOI = "10.1016/j.fluid.2015.07.038"
ETHANOL_RAW = "CCO"
ETHANOL_NAME = "ethanol"
ETHYL_ACETATE_RAW = "CCOC(C)=O"
ETHYL_ACETATE_NAME = "ethyl acetate"


def _mol_weight(smi: str) -> float:
    from rdkit.Chem import Descriptors
    return Descriptors.MolWt(Chem.MolFromSmiles(smi))


def main():
    df = pd.read_csv(INTERIM / "01_canonical.csv")
    corrections: list[dict] = []

    # ═══════════════════════════════════════════════════════════════════
    # C1 — paracetamol/water replacements
    # ═══════════════════════════════════════════════════════════════════
    doi_C1 = "10.1016/j.molliq.2020.113867"
    mask_C1 = (
        (df["Source"] == doi_C1)
        & (df["Solute_Canon"] == PARACETAMOL_CANON)
        & (df["Solvent_Canon"] == WATER_CANON)
    )
    # Water Mw, density at each T
    water_mw = _mol_weight(WATER_CANON)
    # Water density model — simple: 1.0 g/cm^3 at these Ts (≈0.1 % deviation,
    # below floating-point precision of BigSolDB logS formula).  Use linear
    # coefficients if available, else 1.0.
    coef = pd.read_csv(ROOT / "data/raw/bigsoldb_v2.1/Coeffs.csv")
    water_coef = coef[coef["Solvent"].str.lower() == "water"]
    if len(water_coef):
        a, b = float(water_coef.iloc[0]["a"]), float(water_coef.iloc[0]["b"])
        water_density = lambda T: a * T + b
    else:
        water_density = lambda T: 1.0
    n_C1 = 0
    for T, x_new in PARACETAMOL_WATER_CORRECTIONS.items():
        row_mask = mask_C1 & (df["Temperature_K"] == T)
        if not row_mask.any():
            print(f"  C1 WARN: no row at T={T} K for paracetamol/water in {doi_C1}")
            continue
        rho = water_density(T)
        S_new = x_new / (1 - x_new) * rho * 1000.0 / water_mw  # mol/L
        logS_new = float(np.log10(S_new))
        # Capture before/after for audit
        for idx in df.index[row_mask]:
            corrections.append({
                "class": "C1", "doi": doi_C1, "solute": PARACETAMOL_CANON,
                "solvent": WATER_CANON, "T": float(T),
                "x_old": float(df.at[idx, "Solubility(mole_fraction)"]),
                "x_new": float(x_new),
                "logS_old": float(df.at[idx, "LogS(mol/L)"])
                    if pd.notna(df.at[idx, "LogS(mol/L)"]) else None,
                "logS_new": logS_new,
            })
            df.at[idx, "Solubility(mole_fraction)"] = x_new
            df.at[idx, "LogS(mol/L)"] = logS_new
            df.at[idx, "Solubility(mol/L)"] = S_new
            n_C1 += 1
    print(f"C1 paracetamol/water: corrected {n_C1} rows "
          f"({len(PARACETAMOL_WATER_CORRECTIONS)} temperatures) in {doi_C1}")

    # ═══════════════════════════════════════════════════════════════════
    # C2 — bulk ×10 (add +1.0 to logS)
    # ═══════════════════════════════════════════════════════════════════
    n_C2_total = 0
    for doi in BULK_10X_DOIS:
        mask = df["Source"] == doi
        n = int(mask.sum())
        if n == 0:
            print(f"  C2 WARN: no rows for {doi}")
            continue
        # We update logS, mol/L, and mole_fraction.  Adding +1 to logS means
        # multiplying concentration by 10.  For mole fraction, the relationship
        # is S = x/(1-x)·ρ·1000/Mw; at small x, S ≈ x·ρ·1000/Mw, so x ≈ S/10
        # stays linear.  At larger x we need to invert S = x/(1-x)·k exactly.
        # Easiest: keep x, logS, mol/L consistent by recomputing.
        coef_map = {row["Solvent"].strip().lower(): (float(row["a"]), float(row["b"]))
                    for _, row in coef.iterrows()}
        for idx in df.index[mask]:
            solv = str(df.at[idx, "Solvent"]).strip().lower()
            T = float(df.at[idx, "Temperature_K"])
            # 10× the concentration: S_new = 10 · S_old
            S_old = float(df.at[idx, "Solubility(mol/L)"])
            if not np.isfinite(S_old) or S_old <= 0:
                # recompute from old x first
                x_old = float(df.at[idx, "Solubility(mole_fraction)"])
                a, b = coef_map.get(solv, (0.0, 1.0))
                rho = a * T + b
                solv_smi = df.at[idx, "Solvent_Canon"]
                Mw = _mol_weight(solv_smi)
                S_old = x_old / (1 - x_old) * rho * 1000.0 / Mw
            S_new = S_old * 10.0
            logS_new = float(np.log10(S_new))
            # Invert to get x_new: S = x/(1-x)·k → x = S·k / (1 + S·k)  where k = Mw/(ρ·1000)
            a, b = coef_map.get(solv, (0.0, 1.0))
            rho = a * T + b
            solv_smi = df.at[idx, "Solvent_Canon"]
            Mw = _mol_weight(solv_smi)
            k_inv = Mw / (rho * 1000.0)
            S_k = S_new * k_inv
            x_new = S_k / (1 + S_k)
            corrections.append({
                "class": "C2", "doi": doi, "T": T,
                "logS_old": float(df.at[idx, "LogS(mol/L)"]),
                "logS_new": logS_new,
                "x_old": float(df.at[idx, "Solubility(mole_fraction)"]),
                "x_new": float(x_new),
            })
            df.at[idx, "Solubility(mol/L)"] = S_new
            df.at[idx, "LogS(mol/L)"] = logS_new
            df.at[idx, "Solubility(mole_fraction)"] = x_new
            n_C2_total += 1
        print(f"C2 ×10 correction: {n} rows updated in {doi}")

    # ═══════════════════════════════════════════════════════════════════
    # C3 — swap ethanol ↔ ethyl acetate in one DOI
    # ═══════════════════════════════════════════════════════════════════
    mask = df["Source"] == SWAP_SOLVENT_DOI
    n_C3_total = 0
    if mask.any():
        EtOH_canon = Chem.MolToSmiles(Chem.MolFromSmiles(ETHANOL_RAW), isomericSmiles=True)
        EtOAc_canon = Chem.MolToSmiles(Chem.MolFromSmiles(ETHYL_ACETATE_RAW), isomericSmiles=True)

        # Swap Solvent_Canon (the key we use downstream) and Solvent (the name column)
        # and SMILES_Solvent (for auditability).
        swap_map_canon = {EtOH_canon: EtOAc_canon, EtOAc_canon: EtOH_canon}
        swap_map_name = {"ethanol": "ethyl acetate", "ethyl acetate": "ethanol"}
        swap_map_smi = {ETHANOL_RAW: ETHYL_ACETATE_RAW, ETHYL_ACETATE_RAW: ETHANOL_RAW}

        # But logS depends on solvent density and Mw — so logS and mol/L
        # need to be RECOMPUTED after swap (mole fraction stays, it's a ratio
        # that doesn't depend on solvent's density).
        coef_map = {row["Solvent"].strip().lower(): (float(row["a"]), float(row["b"]))
                    for _, row in coef.iterrows()}
        for idx in df.index[mask]:
            old_canon = df.at[idx, "Solvent_Canon"]
            if old_canon not in swap_map_canon:
                continue
            new_canon = swap_map_canon[old_canon]
            new_name = swap_map_name[str(df.at[idx, "Solvent"]).strip().lower()]
            new_smi = swap_map_smi[df.at[idx, "SMILES_Solvent"]]

            x = float(df.at[idx, "Solubility(mole_fraction)"])
            T = float(df.at[idx, "Temperature_K"])
            a, b = coef_map.get(new_name.lower(), (0.0, 1.0))
            rho = a * T + b
            Mw = _mol_weight(new_canon)
            if not np.isfinite(Mw) or Mw <= 0:
                continue
            S_new = x / (1 - x) * rho * 1000.0 / Mw
            logS_new = float(np.log10(S_new))

            corrections.append({
                "class": "C3", "doi": SWAP_SOLVENT_DOI, "T": T,
                "solvent_old_canon": old_canon, "solvent_new_canon": new_canon,
                "logS_old": float(df.at[idx, "LogS(mol/L)"]),
                "logS_new": logS_new,
            })
            df.at[idx, "Solvent"] = new_name
            df.at[idx, "SMILES_Solvent"] = new_smi
            df.at[idx, "Solvent_Canon"] = new_canon
            df.at[idx, "Solubility(mol/L)"] = S_new
            df.at[idx, "LogS(mol/L)"] = logS_new
            n_C3_total += 1
        print(f"C3 solvent swap: {n_C3_total} rows re-labeled in {SWAP_SOLVENT_DOI}")
    else:
        print(f"  C3 WARN: no rows for {SWAP_SOLVENT_DOI}")

    # ═══════════════════════════════════════════════════════════════════
    # Save
    # ═══════════════════════════════════════════════════════════════════
    out_path = INTERIM / "01b_manually_corrected.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved corrected → {out_path}")

    summary = {
        "source": "manual_corrections_log.txt (manual audit by the BigSolDB v2.1 co-maintainer)",
        "C1_paracetamol_water": {
            "doi": doi_C1, "n_rows": n_C1,
            "temperatures": list(PARACETAMOL_WATER_CORRECTIONS.keys()),
        },
        "C2_bulk_10x": {
            "dois": sorted(list(BULK_10X_DOIS)),
            "n_rows_total": n_C2_total,
        },
        "C3_solvent_swap": {
            "doi": SWAP_SOLVENT_DOI,
            "swap": "ethanol ↔ ethyl acetate",
            "n_rows": n_C3_total,
        },
        "total_rows_modified": n_C1 + n_C2_total + n_C3_total,
    }
    with open(REPORTS / "15_manual_corrections.json", "w") as f:
        json.dump({"summary": summary, "corrections": corrections}, f, indent=2)
    print(f"Total rows modified: {n_C1 + n_C2_total + n_C3_total}")
    print(f"Wrote audit → reports/15_manual_corrections.json")


if __name__ == "__main__":
    main()
