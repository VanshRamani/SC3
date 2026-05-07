"""
Phase 7 — Benchmark splits (train / eval / OOD).

Narrative for a first-time reader:
  SC³ has THREE types of test set, each probing a different axis of
  generalization:
    (1) Eval  — pairs inside the top-N solvents but with held-out
        (solute, solvent) combinations; tests pair-coverage generalization
        inside the familiar-solvent regime.
    (2) OOD   — all rows of long-tail solvents not seen during training;
        tests SOLVENT generalization.
    (3) Gold / Silver / Bronze (Phase 6) — 148 solutes with multi-source
        calibrated ground truth, entirely removed from training; test
        SOLUTE generalization with known label uncertainty σ.

Policy (D-16):
  • Target row mix: ~75 % train, ~10 % eval, ~15 % OOD (by cleaned-row count,
    BEFORE tier removal).  Equivalently, top-N solvents = ~85 % of training
    pool → OOD is ~15 % by construction, eval is 10 % of ID pool.
  • N is chosen as the smallest number of solvents whose cumulative row
    share in the training pool is ≥ 0.85 (data-driven, no hardcoding).
  • Anti-leakage: tier-Bronze's 148 solutes (the union across all 3 tiers
    since nested) are removed from the training pool under ALL solvents.
  • Eval hold-out: for each ID-set solvent, a fixed 10 % of its
    (solute, solvent) pairs are moved to eval; all temperature measurements
    of a held-out pair stay together.  Seed = 42.
  • OOD is all rows of the non-top-N solvents, with tier solutes already
    removed.

Input:
  data/interim/02_cleaned.csv
  data/sc3/tier_pairs.csv

Output:
  data/splits/bench_train.csv
  data/splits/bench_eval.csv
  data/splits/bench_ood.csv
  reports/70_splits.json
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INTERIM = ROOT / "data/interim"
SC3 = ROOT / "data/sc3"
SPLITS = ROOT / "data/splits"
REPORTS = ROOT / "reports"
SPLITS.mkdir(parents=True, exist_ok=True)

SEED = 42
ID_COVERAGE_TARGET = 0.85         # top-N solvents ≥ 85 % of training pool
EVAL_FRACTION = 0.10              # per-solvent pair hold-out


def main():
    df = pd.read_csv(INTERIM / "02_cleaned.csv")
    tier = pd.read_csv(SC3 / "tier_pairs.csv")

    # Tier test solutes = every solute that appears in Bronze (which ⊃ Silver ⊃ Gold)
    tier_solutes = set(tier[tier["tier_bronze"]]["Solute_Canon"])
    print(f"Tier test solutes: {len(tier_solutes)}")

    # Training pool = cleaned rows minus any row whose solute is a tier solute
    pool_mask = ~df["Solute_Canon"].isin(tier_solutes)
    pool = df[pool_mask].copy()
    pool.attrs["description"] = "tier-solute-removed training pool"
    print(f"Training pool after tier-solute removal: {len(pool):,} rows "
          f"({100*len(pool)/len(df):.1f}% of cleaned)")

    # Top-N selection: smallest N whose cumulative row share ≥ 0.85
    solv_counts = (pool["Solvent_Canon"].value_counts()
                   .rename_axis("Solvent_Canon").reset_index(name="n_rows"))
    solv_counts["cum_share"] = solv_counts["n_rows"].cumsum() / len(pool)
    idx_N = int(np.searchsorted(solv_counts["cum_share"].values, ID_COVERAGE_TARGET) + 1)
    top_N_solvents = set(solv_counts.iloc[:idx_N]["Solvent_Canon"])
    ood_solvents = set(solv_counts.iloc[idx_N:]["Solvent_Canon"])
    print(f"Top-N selection: N = {idx_N} solvents cover "
          f"{100*solv_counts.iloc[idx_N - 1]['cum_share']:.2f}% of pool")
    print(f"  → ID  (train + eval): {idx_N} solvents")
    print(f"  → OOD: {len(ood_solvents)} solvents")

    # ID pool vs OOD pool
    id_pool = pool[pool["Solvent_Canon"].isin(top_N_solvents)].copy()
    ood = pool[pool["Solvent_Canon"].isin(ood_solvents)].copy()
    print(f"ID pool:  {len(id_pool):,} rows "
          f"({100*len(id_pool)/len(pool):.1f}% of pool)")
    print(f"OOD set:  {len(ood):,} rows "
          f"({100*len(ood)/len(pool):.1f}% of pool)")

    # Eval hold-out: per-solvent, 10 % of (solute, solvent) pairs, pair-level
    rng = np.random.default_rng(SEED)
    eval_keys: list[tuple[str, str]] = []
    for solvent, sub in id_pool.groupby("Solvent_Canon"):
        pairs = sub[["Solute_Canon", "Solvent_Canon"]].drop_duplicates()
        n_eval = max(1, int(np.ceil(len(pairs) * EVAL_FRACTION)))
        picked = pairs.sample(n=n_eval, random_state=int(rng.integers(0, 1 << 31)))
        eval_keys.extend([(r.Solute_Canon, r.Solvent_Canon) for r in picked.itertuples()])
    eval_set = set(eval_keys)
    id_pool["_pair"] = list(zip(id_pool["Solute_Canon"], id_pool["Solvent_Canon"]))
    eval_mask = id_pool["_pair"].isin(eval_set)
    bench_eval = id_pool[eval_mask].drop(columns=["_pair"]).copy()
    bench_train = id_pool[~eval_mask].drop(columns=["_pair"]).copy()
    print(f"\nTrain: {len(bench_train):,} rows, {bench_train['Solute_Canon'].nunique()} solutes, "
          f"{bench_train['Solvent_Canon'].nunique()} solvents")
    print(f"Eval:  {len(bench_eval):,} rows, {bench_eval['Solute_Canon'].nunique()} solutes, "
          f"{bench_eval['Solvent_Canon'].nunique()} solvents, "
          f"{bench_eval[['Solute_Canon', 'Solvent_Canon']].drop_duplicates().shape[0]} pairs")
    print(f"OOD:   {len(ood):,} rows, {ood['Solute_Canon'].nunique()} solutes, "
          f"{ood['Solvent_Canon'].nunique()} solvents")

    # ═══════════════════════════════════════════════════════════════════
    # Anti-leakage verification
    # ═══════════════════════════════════════════════════════════════════
    tol = {}
    solutes = {
        "train": set(bench_train["Solute_Canon"]),
        "eval":  set(bench_eval["Solute_Canon"]),
        "ood":   set(ood["Solute_Canon"]),
        "tier":  tier_solutes,
    }
    anti_leak_pass = True
    for a in ["train", "eval", "ood"]:
        overlap = solutes[a] & solutes["tier"]
        tol[f"{a} ∩ tier"] = len(overlap)
        if overlap:
            anti_leak_pass = False
            print(f"  ❌ LEAK: {a} shares {len(overlap)} solutes with tier")

    # Train and eval SHOULD overlap in solutes (pair-level split); report it
    tol["train ∩ eval (solutes, expected > 0)"] = len(solutes["train"] & solutes["eval"])
    # Train and OOD may overlap in solutes (different solvents); report it
    tol["train ∩ ood (solutes, expected > 0)"] = len(solutes["train"] & solutes["ood"])
    # Train ∩ eval at PAIR level SHOULD be 0
    train_pairs = set(zip(bench_train["Solute_Canon"], bench_train["Solvent_Canon"]))
    eval_pairs = set(zip(bench_eval["Solute_Canon"], bench_eval["Solvent_Canon"]))
    pair_leak = len(train_pairs & eval_pairs)
    tol["train ∩ eval (pairs, must be 0)"] = pair_leak
    if pair_leak:
        anti_leak_pass = False
        print(f"  ❌ LEAK: {pair_leak} pairs in both train and eval")
    # All pair-level leaks must be 0
    for tier_name in ["gold", "silver", "bronze"]:
        tier_df = pd.read_csv(SC3 / f"{tier_name}.csv")
        tier_pairs = set(zip(tier_df["Solute_Canon"], tier_df["Solvent_Canon"]))
        for split_name, split_pairs in [("train", train_pairs), ("eval", eval_pairs),
                                         ("ood", set(zip(ood["Solute_Canon"], ood["Solvent_Canon"])))]:
            overlap = len(tier_pairs & split_pairs)
            tol[f"{split_name} ∩ {tier_name} (pairs, must be 0)"] = overlap
            if overlap:
                anti_leak_pass = False
                print(f"  ❌ LEAK: {overlap} pairs in both {split_name} and {tier_name}")

    print(f"\nAnti-leakage verification: {'✓ PASS' if anti_leak_pass else '✗ FAIL'}")
    for k, v in tol.items():
        print(f"  {k}: {v}")

    # ═══════════════════════════════════════════════════════════════════
    # Save splits
    # ═══════════════════════════════════════════════════════════════════
    # Columns kept: Solute_Canon, Solvent_Canon, Solvent, Temperature_K, LogS,
    # Solubility(mole_fraction), MW, Source, plus canonical SMILES raw
    out_cols = [
        "Solute_Canon", "Solvent_Canon", "Solvent",
        "Temperature_K", "LogS", "Solubility(mole_fraction)",
        "MW", "Source",
    ]
    bench_train[out_cols].to_csv(SPLITS / "bench_train.csv", index=False)
    bench_eval[out_cols].to_csv(SPLITS / "bench_eval.csv", index=False)
    ood[out_cols].to_csv(SPLITS / "bench_ood.csv", index=False)
    print(f"\nWrote: data/splits/bench_train.csv, bench_eval.csv, bench_ood.csv")

    # ═══════════════════════════════════════════════════════════════════
    # Report
    # ═══════════════════════════════════════════════════════════════════
    report = {
        "policy": {
            "seed": SEED,
            "ID_coverage_target": ID_COVERAGE_TARGET,
            "eval_fraction_per_solvent": EVAL_FRACTION,
            "top_N_solvents": int(idx_N),
            "top_N_cumulative_share": float(solv_counts.iloc[idx_N - 1]["cum_share"]),
        },
        "tier_test_solutes": int(len(tier_solutes)),
        "counts": {
            "cleaned_rows": int(len(df)),
            "training_pool_rows": int(len(pool)),
            "training_pool_frac_of_cleaned": float(len(pool) / len(df)),
            "train": {
                "rows": int(len(bench_train)),
                "solutes": int(bench_train["Solute_Canon"].nunique()),
                "solvents": int(bench_train["Solvent_Canon"].nunique()),
                "pairs": int(bench_train[["Solute_Canon", "Solvent_Canon"]]
                            .drop_duplicates().shape[0]),
            },
            "eval": {
                "rows": int(len(bench_eval)),
                "solutes": int(bench_eval["Solute_Canon"].nunique()),
                "solvents": int(bench_eval["Solvent_Canon"].nunique()),
                "pairs": int(bench_eval[["Solute_Canon", "Solvent_Canon"]]
                            .drop_duplicates().shape[0]),
            },
            "ood": {
                "rows": int(len(ood)),
                "solutes": int(ood["Solute_Canon"].nunique()),
                "solvents": int(ood["Solvent_Canon"].nunique()),
                "pairs": int(ood[["Solute_Canon", "Solvent_Canon"]]
                            .drop_duplicates().shape[0]),
            },
        },
        "target_vs_actual_mix": {
            "target_train_frac": 0.75,
            "target_eval_frac": 0.10,
            "target_ood_frac": 0.15,
            "actual_train_frac": float(len(bench_train) / len(pool)),
            "actual_eval_frac": float(len(bench_eval) / len(pool)),
            "actual_ood_frac": float(len(ood) / len(pool)),
        },
        "anti_leakage": {
            "pass": bool(anti_leak_pass),
            "checks": {k: int(v) for k, v in tol.items()},
        },
    }
    with open(REPORTS / "70_splits.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote report → reports/70_splits.json")

    # Summary table for console
    print(f"\n{'═'*72}")
    print(f"SPLIT SUMMARY")
    print(f"{'═'*72}")
    print(f"{'Split':<10}{'Rows':>10}{'Solutes':>10}{'Solvents':>10}{'Pairs':>10}{'% pool':>10}")
    print(f"{'─'*72}")
    frac = lambda n: f"{100*n/len(pool):.1f}%"
    print(f"{'Train':<10}{len(bench_train):>10,}"
          f"{bench_train['Solute_Canon'].nunique():>10,}"
          f"{bench_train['Solvent_Canon'].nunique():>10,}"
          f"{bench_train[['Solute_Canon','Solvent_Canon']].drop_duplicates().shape[0]:>10,}"
          f"{frac(len(bench_train)):>10}")
    print(f"{'Eval':<10}{len(bench_eval):>10,}"
          f"{bench_eval['Solute_Canon'].nunique():>10,}"
          f"{bench_eval['Solvent_Canon'].nunique():>10,}"
          f"{bench_eval[['Solute_Canon','Solvent_Canon']].drop_duplicates().shape[0]:>10,}"
          f"{frac(len(bench_eval)):>10}")
    print(f"{'OOD':<10}{len(ood):>10,}"
          f"{ood['Solute_Canon'].nunique():>10,}"
          f"{ood['Solvent_Canon'].nunique():>10,}"
          f"{ood[['Solute_Canon','Solvent_Canon']].drop_duplicates().shape[0]:>10,}"
          f"{frac(len(ood)):>10}")


if __name__ == "__main__":
    main()
