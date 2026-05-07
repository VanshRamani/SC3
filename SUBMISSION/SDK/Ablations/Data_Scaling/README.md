# Ablations / Data Scaling

This experiment characterises how three architectures of decreasing
inductive bias scale with training-set size on the SC³ benchmark.
The goal is to disentangle the contributions of *model capacity* and
*data volume* under a fixed featurization.

## Hypothesis

Simple, high-bias models (LightGBM on RDKit descriptors) should saturate
after seeing 20–40% of the training set. Larger, data-hungry models
(FastProp deep MLP, MolMerger AttentiveFP) should keep improving as more
data is added. If the larger models do not plateau on the full SC³
training set, that is evidence that data volume — rather than
architecture — is the bottleneck for the corpus.

## Methods compared

| Key         | Description                                                | Family            |
|-------------|------------------------------------------------------------|-------------------|
| `lgb_rdkit` | LightGBM on RDKit 2-D descriptors + temperature features   | Tabular tree      |
| `fastprop`  | Deep MLP (512–256–128, BN, dropout) on RDKit + T           | Deep descriptor   |
| `molmerger` | AttentiveFP on Gasteiger-merged solute–solvent graphs      | Merged-graph GNN  |

All three reuse the SDK code paths and best hyperparameters from
`SDK/configs/best_hps.json`.

## Files

| File                  | Purpose                                              |
|-----------------------|------------------------------------------------------|
| `run_data_scaling.py` | Driver: sweeps `(method, fraction, seed)` and writes one JSON per run. |
| `scaling_trainers.py` | Per-method training routines (subsample → train → evaluate on every split). |
| `cache_graphs.py`     | One-time graph-feature caching utility for the GNN run. |

## Running

```bash
cd Ablations/Data_Scaling
python cache_graphs.py        # one-time graph cache for molmerger
python run_data_scaling.py    # full sweep (see --help for fractions / seeds / methods)
```
