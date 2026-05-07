# Ablations / Representation

This experiment isolates the contribution of the *molecular
representation* by **fixing the model** to a single tuned LightGBM and
**varying the featurizer**. It is the design control for the
Interpretability ablation.

## Hypothesis

If accuracy is dominated by representation, swapping featurizers under a
fixed estimator should produce a clear, ordered separation in test
RMSE / MAE. If accuracy is dominated by the estimator, then under a
strong baseline (LightGBM) every reasonable representation should land
in a similar band, and the gap to the deep models studied in the data-
scaling ablation must come from the model class, not the features.

## Design

| Knob | Value |
|------|-------|
| Model | LightGBM (fixed) |
| HPs | `SDK/configs/best_hps.json["lgb_rdkit"]` (held constant across featurizers) |
| Early stopping | 50 rounds on the full `eval` split |
| Eval splits | `eval`, `ood`, `sc3_gold`, `sc3_silver`, `sc3_bronze` |
| Metrics | RMSE, MAE, R², PS-RMSE, Z-RMSE |

All featurizers are passed through the same `build_features` pipeline
(concat solute + solvent + 4 temperature features) so the model sees a
solute–solvent–T input vector and only the chemistry encoding changes.

## Files

| File                    | Purpose                                                  |
|-------------------------|----------------------------------------------------------|
| `run_representation.py` | Driver: sweep one fixed LightGBM across featurizers.     |

## Running

```bash
cd Ablations/Representation
python run_representation.py            # default: all featurizers, seed 42
python run_representation.py --help     # CLI knobs
```
