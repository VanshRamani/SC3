# Ablations / Interpretability

This experiment attributes the predictions of a fixed LightGBM model
back to the chemistry, across the same 7 featurizers used in the
Representation ablation. It is the *interpretation* arm of the question
"what has the model actually learned?".

## Method

* **Tabular models** (`lgb_rdkit` on each of 7 featurizers, single seed):
  exact Tree-SHAP via `shap.TreeExplainer` (`feature_perturbation =
  "tree_path_dependent"`) on the full `eval` and `ood` splits. SHAP is
  also computed per-solvent for the top-25 in-distribution solvents,
  enabling a SHAP-fingerprint–based clustering of solvents.
* **Graph model** (`gcn`, single seed): reuses the trained checkpoint
  (no retraining) and produces atom-level occlusion attribution
  `a_v = f(G) − f(G \ {v})` on a sample of `eval` rows, then aggregates
  to BRICS fragments.

## Files

| File                  | Purpose                                                 |
|-----------------------|---------------------------------------------------------|
| `run_shap.py`         | Computes Tree-SHAP for each (featurizer, split) pair.   |
| `run_gcn_explain.py`  | Computes atom-occlusion attributions for the GCN.       |

Both runners assume the corresponding LightGBM / GCN model checkpoint
has already been produced by the main SDK (see `SDK/sc3_bench/train.py`).

## Running

```bash
cd Ablations/Interpretability

# 1. Tree-SHAP across featurizers (LightGBM trained on the fly per featurizer).
python run_shap.py

# 2. Graph attribution for the GCN (loads a pretrained checkpoint).
python run_gcn_explain.py
```
