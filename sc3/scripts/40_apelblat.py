"""
Phase 4 — Thermodynamic consistency (Apelblat + van't Hoff).

Fits temperature-dependent log S models to (solute, solvent, independence_group)
triples.  Units: T in Kelvin; logS in log10(mol / L).

Models:
  Apelblat:  logS = A + B/T + C·ln(T)      (needs ≥ 3 unique T points)
  van't Hoff: logS = A + B/T               (for exactly 2 unique T points)

No extrapolation beyond [T_min, T_max] of the triple — evaluation of fits is
strictly within the measured range (D-13).

Outputs:
  data/interim/04_fits.csv    (solute_canon, solvent_canon, gid, model,
                               A, B, C, R2, RMSE, n_points, T_min, T_max, ...)
  reports/40_apelblat.json    (summary counts, fit-quality distribution)
  reports/40_monotonicity.csv (per triple: Spearman ρ vs T, #reversals, worst drop)
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
INTERIM = ROOT / "data/interim"
REPORTS = ROOT / "reports"


def apelblat(T, A, B, C):
    return A + B / T + C * np.log(T)


def vant_hoff(T, A, B):
    return A + B / T


def fit_apelblat(T: np.ndarray, y: np.ndarray):
    """Return (A, B, C, R2, RMSE) or None on failure."""
    try:
        p0 = [y.mean(), 0.0, 0.0]
        popt, _ = curve_fit(apelblat, T, y, p0=p0, maxfev=5000)
        yhat = apelblat(T, *popt)
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
        rmse = float(np.sqrt(ss_res / len(y)))
        return (*[float(p) for p in popt], float(r2), rmse)
    except Exception:
        return None


def fit_vant(T: np.ndarray, y: np.ndarray):
    try:
        p0 = [y.mean(), 0.0]
        popt, _ = curve_fit(vant_hoff, T, y, p0=p0, maxfev=5000)
        yhat = vant_hoff(T, *popt)
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
        rmse = float(np.sqrt(ss_res / len(y)))
        return (*[float(p) for p in popt], float(r2), rmse)
    except Exception:
        return None


def main():
    df = pd.read_csv(INTERIM / "02_cleaned.csv")
    gmap = pd.read_csv(INTERIM / "03_doi_groups.csv").set_index("doi")["group_id"].to_dict()
    df["gid"] = df["Source"].map(gmap)
    print(f"Loaded {len(df):,} rows, {df['gid'].nunique()} independence groups")

    triple_g = df.groupby(["Solute_Canon", "Solvent_Canon", "gid"])

    fits: list[dict] = []
    monot: list[dict] = []
    n_apelblat = n_vhf = n_single = 0
    for (solute, solvent, gid), sub in triple_g:
        # Collapse duplicate T (can happen if multiple DOIs in same group measure
        # at same T) by averaging logS at that T.
        g = sub.groupby("Temperature_K")["LogS"].mean().reset_index()
        T = g["Temperature_K"].to_numpy(dtype=float)
        y = g["LogS"].to_numpy(dtype=float)
        order = np.argsort(T)
        T, y = T[order], y[order]
        n = len(T)

        row = {
            "Solute_Canon": solute,
            "Solvent_Canon": solvent,
            "gid": int(gid),
            "n_points": int(n),
            "T_min": float(T.min()),
            "T_max": float(T.max()),
            "T_range": float(T.max() - T.min()),
            "logS_min": float(y.min()),
            "logS_max": float(y.max()),
            "logS_range": float(y.max() - y.min()),
        }
        if n >= 3:
            res = fit_apelblat(T, y)
            if res is not None:
                A, B, C, R2, rmse = res
                row.update({"model": "apelblat", "A": A, "B": B, "C": C,
                            "R2": R2, "RMSE": rmse})
                n_apelblat += 1
            else:
                row.update({"model": "apelblat_failed", "R2": None, "RMSE": None})
        elif n == 2:
            res = fit_vant(T, y)
            if res is not None:
                A, B, R2, rmse = res
                row.update({"model": "vanthoff", "A": A, "B": B,
                            "R2": R2, "RMSE": rmse})
                n_vhf += 1
            else:
                row.update({"model": "vanthoff_failed"})
        else:
            row.update({"model": "single_point"})
            n_single += 1
        fits.append(row)

        # Monotonicity analysis (only meaningful for n ≥ 3)
        if n >= 3:
            rho, _ = spearmanr(T, y)
            n_up = int(np.sum(np.diff(y) > 0))
            n_down = int(np.sum(np.diff(y) < 0))
            worst_drop = float(np.min(np.diff(y)))  # most negative step
            monot.append({
                "Solute_Canon": solute,
                "Solvent_Canon": solvent,
                "gid": int(gid),
                "n_points": int(n),
                "spearman_rho": float(rho) if not np.isnan(rho) else None,
                "n_up_steps": n_up,
                "n_down_steps": n_down,
                "monotone_increasing": n_down == 0,
                "worst_drop": worst_drop,  # negative = T↑ but logS↓
                "delta_logS_range": float(y.max() - y.min()),
            })

    fits_df = pd.DataFrame(fits)
    monot_df = pd.DataFrame(monot)

    fits_df.to_csv(INTERIM / "04_fits.csv", index=False)
    monot_df.to_csv(REPORTS / "40_monotonicity.csv", index=False)

    # ── Summary ────────────────────────────────────────────────────────
    ap_fits = fits_df[fits_df["model"] == "apelblat"]
    r2 = ap_fits["R2"].astype(float)
    rmse_col = ap_fits["RMSE"].astype(float)
    frac_r2_ge_095 = float((r2 >= 0.95).mean())
    frac_r2_ge_099 = float((r2 >= 0.99).mean())

    # monotonicity summary
    mi = monot_df["monotone_increasing"]
    rho = monot_df["spearman_rho"].astype(float)
    big_drops = int((monot_df["worst_drop"] < -1.0).sum())
    med_drops = int((monot_df["worst_drop"] < -0.5).sum())

    print(f"\nApelblat fits: {n_apelblat:,}")
    print(f"  median R²: {r2.median():.4f}  mean R²: {r2.mean():.4f}")
    print(f"  frac R² ≥ 0.95: {frac_r2_ge_095:.3%}")
    print(f"  frac R² ≥ 0.99: {frac_r2_ge_099:.3%}")
    print(f"  RMSE median: {rmse_col.median():.4f}  P95: {rmse_col.quantile(0.95):.4f}")
    print(f"\nvan't Hoff (2-point) fits: {n_vhf:,}")
    print(f"Single-point triples (no fit): {n_single:,}")
    print(f"\nMonotonicity (n ≥ 3 triples): {len(monot_df):,}")
    print(f"  strictly monotone-increasing: {int(mi.sum()):,}  ({mi.mean():.1%})")
    print(f"  Spearman ρ > 0.8: {int((rho > 0.8).sum()):,}  ({(rho > 0.8).mean():.1%})")
    print(f"  worst-drop < −0.5 logS: {med_drops:,}")
    print(f"  worst-drop < −1.0 logS: {big_drops:,}")

    # bad-fit candidates for flagging: R² < 0.8 or RMSE > 0.3 with n ≥ 3
    bad_fits = ap_fits[(ap_fits["R2"] < 0.8) | (ap_fits["RMSE"] > 0.3)]
    print(f"\nBad fits (R² < 0.8 OR RMSE > 0.3): {len(bad_fits):,} of {n_apelblat:,}")

    summary = {
        "input": {"rows": int(len(df)), "triples": int(len(fits_df))},
        "fits": {
            "apelblat": int(n_apelblat),
            "vant_hoff": int(n_vhf),
            "single_point": int(n_single),
            "apelblat_R2_median": float(r2.median()),
            "apelblat_R2_mean": float(r2.mean()),
            "apelblat_frac_R2_geq_0_95": frac_r2_ge_095,
            "apelblat_frac_R2_geq_0_99": frac_r2_ge_099,
            "apelblat_RMSE_median": float(rmse_col.median()),
            "bad_fits_count": int(len(bad_fits)),
        },
        "monotonicity": {
            "n_triples_tested": int(len(monot_df)),
            "monotone_increasing_count": int(mi.sum()),
            "monotone_increasing_frac": float(mi.mean()),
            "spearman_rho_gt_0_8_frac": float((rho > 0.8).mean()),
            "big_drop_lt_minus_1_logS_count": big_drops,
            "med_drop_lt_minus_0_5_logS_count": med_drops,
        },
    }
    with open(REPORTS / "40_apelblat.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote fits → {INTERIM / '04_fits.csv'}")
    print(f"Wrote summary → reports/40_apelblat.json")


if __name__ == "__main__":
    main()
