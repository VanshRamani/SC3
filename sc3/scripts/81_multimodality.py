"""
Phase 8 — Motivating analysis for the metric suite.

Builds a numerical case for why standard metrics (RMSE, R², MAPE) mislead
on multi-solvent solubility, and why PS-RMSE / MedAE / Z-RMSE are better.

Each section saves figure-ready numbers for the paper-figure agent.

  §1  Multimodality.
      Per-solvent logS distributions; the dataset-level distribution is a
      mixture over 206 solvent-specific unimodal distributions that shift
      by > 7 log units in location.  Saves per-solvent (n, mean, std,
      median) for all solvents.

  §2  Variance decomposition.
      One-way ANOVA: total variance = between-solvent + within-solvent.
      The between-solvent fraction is the R² that a dummy model gets
      "for free" just by identifying which solvent a point is in.

  §3  Dummy R² baseline.
      Empirical verification — fit a model that predicts the per-solvent
      mean on training data and evaluate R² on eval, OOD, and tiers.
      This is the floor aggregate R² inherits, independent of solute
      chemistry.

  §4  Count domination.
      Cumulative row share by top-N solvents: shows why aggregate RMSE
      is a count-weighted average dominated by a handful of common
      solvents.

  §5  MAPE diagnostic.
      Fraction of rows with |logS| below various thresholds.  Anywhere
      |logS| < ε, MAPE = |ŷ − y| / |y| blows up.  For SC³, the fraction
      of divergent rows is not negligible.

  §6  Heavy-tail of residuals.
      Our Phase-5 aleatoric |Δ| distribution is heavy-tailed (gamma shape
      0.68).  Model residuals on literature-noise-dominated data inherit
      a similar tail.  RMSE is tail-sensitive; MedAE is not.  Reports the
      robustness ratio (mean / median) on the training-label distribution
      as a proxy for how much the two would disagree on model residuals.

  §7  Per-split metric computational domain.
      For every (split × metric), what n can the metric actually be
      computed on?  Key fact: Z-RMSE requires σ, which is NaN for ~23 %
      of tier rows.

Outputs:
  reports/81_multimodality.json        (numeric summary)
  reports/81_per_solvent_stats.csv     (§1, per-solvent table)
  reports/81_metric_domains.csv        (§7, split × metric × n)
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import f_oneway

ROOT = Path(__file__).resolve().parent.parent
INTERIM = ROOT / "data/interim"
SPLITS = ROOT / "data/splits"
SC3 = ROOT / "data/sc3"
REPORTS = ROOT / "reports"


def load():
    cleaned = pd.read_csv(INTERIM / "02_cleaned.csv")
    train = pd.read_csv(SPLITS / "bench_train.csv")
    evalu = pd.read_csv(SPLITS / "bench_eval.csv")
    ood = pd.read_csv(SPLITS / "bench_ood.csv")
    # Tier CSVs use LogS_consensus; rename to unify column semantics
    gold = pd.read_csv(SC3 / "gold.csv").rename(columns={"LogS_consensus": "LogS"})
    silver = pd.read_csv(SC3 / "silver.csv").rename(columns={"LogS_consensus": "LogS"})
    bronze = pd.read_csv(SC3 / "bronze.csv").rename(columns={"LogS_consensus": "LogS"})
    return cleaned, train, evalu, ood, gold, silver, bronze


def s1_multimodality(cleaned: pd.DataFrame) -> pd.DataFrame:
    g = (cleaned.groupby("Solvent_Canon")["LogS"]
         .agg(n="size", mean="mean", std="std", median="median",
              p10=lambda x: np.percentile(x, 10),
              p90=lambda x: np.percentile(x, 90))
         .reset_index()
         .sort_values("n", ascending=False))
    # attach the name of the solvent (first raw name from cleaned)
    name_map = cleaned.groupby("Solvent_Canon")["Solvent"].first()
    g["solvent_name"] = g["Solvent_Canon"].map(name_map)
    return g


def s2_variance_decomposition(df: pd.DataFrame, name: str) -> dict:
    groups = [df.loc[df["Solvent_Canon"] == s, "LogS"].to_numpy()
              for s in df["Solvent_Canon"].unique() if df.loc[df["Solvent_Canon"] == s].shape[0] > 0]
    # Classical one-way ANOVA
    # total_ss = sum((y_i - mean)^2)
    # between_ss = sum_k n_k (mean_k - mean)^2
    y = df["LogS"].to_numpy()
    grand = float(np.mean(y))
    total_ss = float(np.sum((y - grand) ** 2))
    between_ss = float(sum(len(g) * (np.mean(g) - grand) ** 2 for g in groups if len(g)))
    within_ss = total_ss - between_ss
    # F-test (optional for paper)
    try:
        f_stat, p_val = f_oneway(*[g for g in groups if len(g) > 1])
    except Exception:
        f_stat, p_val = float("nan"), float("nan")
    return {
        "split": name,
        "n_rows": int(len(y)),
        "n_solvents": int(df["Solvent_Canon"].nunique()),
        "grand_mean_logS": grand,
        "total_var": total_ss / len(y),
        "between_frac": between_ss / total_ss if total_ss > 0 else 0.0,
        "within_frac": within_ss / total_ss if total_ss > 0 else 1.0,
        "f_stat": float(f_stat) if not np.isnan(f_stat) else None,
        "p_value": float(p_val) if not np.isnan(p_val) else None,
    }


def s3_dummy_r2(train: pd.DataFrame, other: pd.DataFrame, other_name: str) -> dict:
    """A model that predicts `solvent_mean(train)` at each eval row.
    When a solvent is absent from training (e.g., OOD), fall back to
    grand mean of training."""
    solv_mean = train.groupby("Solvent_Canon")["LogS"].mean().to_dict()
    grand = float(train["LogS"].mean())
    y = other["LogS"].to_numpy()
    yhat = np.asarray([solv_mean.get(s, grand) for s in other["Solvent_Canon"]], dtype=float)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse_val = float(np.sqrt(np.mean((y - yhat) ** 2)))
    return {
        "target_split": other_name,
        "n_rows": int(len(y)),
        "dummy_R2": r2,
        "dummy_RMSE": rmse_val,
        "solvents_missing_from_train": int(
            sum(1 for s in other["Solvent_Canon"].unique() if s not in solv_mean)
        ),
    }


def s4_count_domination(pool: pd.DataFrame) -> dict:
    counts = pool["Solvent_Canon"].value_counts().to_numpy()
    total = counts.sum()
    cum = counts.cumsum() / total
    return {
        "top_1_frac":   float(cum[0]) if len(cum) >= 1 else 0.0,
        "top_5_frac":   float(cum[4]) if len(cum) >= 5 else float(cum[-1]),
        "top_10_frac":  float(cum[9]) if len(cum) >= 10 else float(cum[-1]),
        "top_25_frac":  float(cum[24]) if len(cum) >= 25 else float(cum[-1]),
        "top_50_frac":  float(cum[49]) if len(cum) >= 50 else float(cum[-1]),
        "n_solvents":   int(len(counts)),
    }


def s5_mape_diagnostic(df: pd.DataFrame) -> dict:
    y = df["LogS"].to_numpy()
    thresholds = [0.01, 0.05, 0.1, 0.2, 0.5]
    return {f"frac_|logS|<{t}": float(np.mean(np.abs(y) < t)) for t in thresholds}


def s6_heavy_tail_of_labels(df: pd.DataFrame) -> dict:
    y = df["LogS"].to_numpy()
    y0 = y - np.mean(y)
    abs_y0 = np.abs(y0)
    return {
        "mean_|y−mean|": float(np.mean(abs_y0)),
        "median_|y−mean|": float(np.median(abs_y0)),
        "mean_over_median_ratio": float(np.mean(abs_y0) / max(np.median(abs_y0), 1e-9)),
        "P95_|y−mean|": float(np.percentile(abs_y0, 95)),
        "P99_|y−mean|": float(np.percentile(abs_y0, 99)),
    }


def s7_metric_domains(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict] = []
    for name, df in dfs.items():
        n = len(df)
        n_solvents = int(df["Solvent_Canon"].nunique()) if n else 0
        if "sigma" in df.columns:
            n_sigma = int(df["sigma"].notna().sum())
        else:
            n_sigma = 0
        rows.append({
            "split": name,
            "n_rows": int(n),
            "n_solutes": int(df["Solute_Canon"].nunique()) if n else 0,
            "n_solvents": n_solvents,
            "n_for_rmse": n,
            "n_for_mae": n,
            "n_for_medae": n,
            "n_groups_for_ps_rmse": n_solvents,
            "n_for_z_rmse": n_sigma,
            "z_rmse_coverage": float(n_sigma / n) if n > 0 else 0.0,
        })
    return pd.DataFrame(rows)


def main():
    cleaned, train, evalu, ood, gold, silver, bronze = load()

    # §1 — Multimodality: per-solvent stats
    per_solv = s1_multimodality(cleaned)
    per_solv.to_csv(REPORTS / "81_per_solvent_stats.csv", index=False)

    # §2 — Variance decomposition, per split
    var_decomp = [
        s2_variance_decomposition(cleaned, "cleaned"),
        s2_variance_decomposition(train, "train"),
        s2_variance_decomposition(evalu, "eval"),
        s2_variance_decomposition(ood, "ood"),
        s2_variance_decomposition(gold, "gold"),
        s2_variance_decomposition(silver, "silver"),
        s2_variance_decomposition(bronze, "bronze"),
    ]

    # §3 — Dummy R² baseline (what solvent-identification gives for free)
    dummy = []
    for other, name in [(evalu, "eval"), (ood, "ood"),
                        (gold, "gold"), (silver, "silver"), (bronze, "bronze")]:
        dummy.append(s3_dummy_r2(train, other, name))

    # §4 — Count domination (on cleaned, training pool, and full training-row pool)
    count_dom_cleaned = s4_count_domination(cleaned)
    count_dom_train = s4_count_domination(train)

    # §5 — MAPE diagnostic (on cleaned = worst-case)
    mape_diag = s5_mape_diagnostic(cleaned)

    # §6 — Heavy-tail label distribution (centered)
    heavy_tail = s6_heavy_tail_of_labels(cleaned)

    # §7 — Metric computational domains, per split
    dfs_for_domain = {
        "train": train, "eval": evalu, "ood": ood,
        "gold": gold, "silver": silver, "bronze": bronze,
    }
    dom = s7_metric_domains(dfs_for_domain)
    dom.to_csv(REPORTS / "81_metric_domains.csv", index=False)

    # ═══════════════════════════════════════════════════════════════════
    # Print reader-oriented narrative
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'═'*72}\n§1  MULTIMODALITY\n{'═'*72}")
    print(f"Cleaned dataset spans {cleaned['Solvent_Canon'].nunique()} solvents with "
          f"per-solvent logS means ranging "
          f"[{per_solv['mean'].min():.2f}, {per_solv['mean'].max():.2f}] "
          f"(span = {per_solv['mean'].max() - per_solv['mean'].min():.2f} log units).")
    print(f"Top-5 solvents by row count:")
    for _, r in per_solv.head(5).iterrows():
        print(f"  {r['solvent_name']:<18} n={int(r['n']):>6}  "
              f"mean={r['mean']:+.2f}  std={r['std']:.2f}")

    print(f"\n{'═'*72}\n§2  VARIANCE DECOMPOSITION\n{'═'*72}")
    for v in var_decomp:
        print(f"  {v['split']:<10} n={v['n_rows']:>7,}  "
              f"between-solvent frac = {v['between_frac']:.3f}  "
              f"(within = {v['within_frac']:.3f})")

    print(f"\n{'═'*72}\n§3  DUMMY-R² BASELINE (predicting solvent mean)\n{'═'*72}")
    for d in dummy:
        print(f"  {d['target_split']:<10} dummy R² = {d['dummy_R2']:+.3f}  "
              f"RMSE = {d['dummy_RMSE']:.3f}")
    print("  → Any model that can identify the solvent inherits these R² values "
          "for free,\n    before learning any solute chemistry.")

    print(f"\n{'═'*72}\n§4  COUNT DOMINATION (cumulative row share)\n{'═'*72}")
    print(f"  (cleaned dataset, {count_dom_cleaned['n_solvents']} solvents)")
    for k in ("top_1_frac", "top_5_frac", "top_10_frac", "top_25_frac"):
        print(f"    {k:<15}  {count_dom_cleaned[k]:.3f}")
    print(f"  → Count-weighted RMSE is dominated by a handful of common solvents.")

    print(f"\n{'═'*72}\n§5  MAPE DIAGNOSTIC\n{'═'*72}")
    for k, v in mape_diag.items():
        print(f"  {k}: {v:.4f}")
    print("  → MAPE = |ŷ−y|/|y| diverges as |y|→0; fractions above show how much "
          "data\n    sits in the divergent regime.")

    print(f"\n{'═'*72}\n§6  HEAVY-TAIL OF LABEL DISTRIBUTION\n{'═'*72}")
    print(f"  mean|y−μ| = {heavy_tail['mean_|y−mean|']:.3f}  "
          f"median|y−μ| = {heavy_tail['median_|y−mean|']:.3f}")
    print(f"  mean/median ratio = {heavy_tail['mean_over_median_ratio']:.2f}  "
          f"(heavy-tail factor)")

    print(f"\n{'═'*72}\n§7  METRIC COMPUTATIONAL DOMAIN\n{'═'*72}")
    print(dom.to_string(index=False))

    # Save summary
    summary = {
        "s1_multimodality": {
            "n_solvents": int(cleaned["Solvent_Canon"].nunique()),
            "logS_solvent_mean_span": float(per_solv["mean"].max() - per_solv["mean"].min()),
            "logS_solvent_mean_min": float(per_solv["mean"].min()),
            "logS_solvent_mean_max": float(per_solv["mean"].max()),
        },
        "s2_variance_decomposition": var_decomp,
        "s3_dummy_R2": dummy,
        "s4_count_domination_cleaned": count_dom_cleaned,
        "s4_count_domination_train": count_dom_train,
        "s5_mape_diagnostic": mape_diag,
        "s6_heavy_tail_label_dist": heavy_tail,
        "metric_suite_summary": {
            "rmse":    "standard; retained for comparability but count-weighted",
            "mae":     "standard; less tail-sensitive than RMSE",
            "medae":   "median absolute error — robust to heavy-tailed residuals",
            "ps_rmse": "per-solvent mean of RMSE — strips count-weighting AND between-solvent inflation",
            "z_rmse":  "(error / sigma)-RMSE on rows with defined σ — error in units of the aleatoric floor",
            "mape":    "diagnostic only — diverges for |logS| < ε",
        },
    }
    with open(REPORTS / "81_multimodality.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nWrote: reports/81_multimodality.json, 81_per_solvent_stats.csv, "
          f"81_metric_domains.csv")


if __name__ == "__main__":
    main()
