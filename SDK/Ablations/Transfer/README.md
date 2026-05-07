# Ablations / Transfer Learning

This experiment asks whether a related chemical property (solvation
free energy) can guide solubility prediction. A FastProp MLP is
**pretrained** on a large auxiliary dataset of COSMO-RS solvation free
energies (CombiSolv-QM, ~1 M solute–solvent pairs) and then
**fine-tuned** on a fraction of the SC³ training set.

## Hypothesis

If solvation free energy and solubility share enough underlying
chemistry — both are governed by solute–solvent interactions — then a
model pretrained on ΔG_solv should give better solubility predictions
than a model trained from scratch, especially in the **low-data
regime** where the SC³ training fraction is small.

## Why CombiSolv-QM?

| Property                | Match to SC³                                    |
|-------------------------|-------------------------------------------------|
| Input shape             | (solute SMILES, solvent SMILES, T) → scalar     |
| Featurization           | RDKit 2-D descriptors per molecule + temperature |
| Architecture compatible | FastProp MLP with one regression head            |
| Scale                   | 999 743 rows, 11 029 unique solutes, 284 solvents |
| Label quality           | COSMO-RS computed, low experimental noise        |

CombiSolv data is supplied by the user via the path configured in
`cache_combisolv_features.py`. It must be pair-level cleaned against
the SC³ holdouts; `leakage_check.py` re-verifies that against the
current `sc3_gold/silver/bronze` splits.

## Protocol

| Protocol  | Variant      | What changes                                                   |
|-----------|--------------|----------------------------------------------------------------|
| `scratch` | `full`       | FastProp trained from scratch on an SC³ training fraction.     |
| `scratch` | `head_only`  | (Sanity) only the final linear head trains; trunk is random-init frozen. |
| `qm`      | `full`       | Pretrain on CombiSolv-QM, swap head, fine-tune all parameters on SC³ logS. |
| `qm`      | `head_only`  | Pretrain on CombiSolv-QM, freeze trunk, train only the new head. |

A second variant in `run_transfer_298k.py` restricts fine-tuning to
room-temperature (~298 K) data only, both as `filter` (real
measurements with 295 ≤ T ≤ 301 K) and `interp` (every pair evaluated
at exactly 298.15 K via the Apelblat / van't Hoff fit from the SC³
cleaning pipeline). The 298 K-locked variant removes the temperature
confound between pretraining and fine-tuning.

## Files

| File                          | Purpose                                                 |
|-------------------------------|---------------------------------------------------------|
| `cache_combisolv_features.py` | One-time RDKit-feature cache for CombiSolv (set the input path here). |
| `leakage_check.py`            | Pair-level overlap re-verification against SC³ splits.  |
| `transfer_trainers.py`        | Pretraining and fine-tuning routines (FastProp).        |
| `run_transfer.py`             | Multi-T driver (default protocol).                      |
| `run_transfer_298k.py`        | 298 K-locked driver (filter + interp variants).         |
| `build_298k_data.py`          | Builds the 298 K-locked SC³ fine-tune set from the splits. |

## Running

```bash
cd Ablations/Transfer

# 0. (one-time) Verify leakage and build the CombiSolv feature cache.
python leakage_check.py
python cache_combisolv_features.py

# 1. Pretrain the FastProp trunk on CombiSolv-QM.
python run_transfer.py --pretrain-only --gpu 0

# 2. Full sweep (default fractions, seeds, protocols, variants).
python run_transfer.py --gpu 0

# 3. (optional) 298 K-locked variant.
python build_298k_data.py
python run_transfer_298k.py --gpu 0
```

## Notes

- Both pretraining and fine-tuning use input normalisation fit on the
  pretraining set so the pretrained trunk sees inputs in the same scale
  at fine-tune time. For `scratch` runs the normalisation is fit on the
  SC³ training fraction itself.
- BatchNorm running statistics are re-calibrated with one no-grad
  forward pass over the fine-tune training data before the first
  evaluation; the same calibration is applied to the scratch model for
  an apples-to-apples comparison.
- All training (pretrain and fine-tune) early-stops on the full
  `bench_eval` split with patience = 20.
- Subsampling of the SC³ training set is stratified by solvent
  (`Solvent_Name`) so even at 5 % fractions every common solvent is
  represented.
