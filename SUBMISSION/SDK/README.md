# SC³ Modeling SDK

A unified Python SDK for benchmarking solubility-prediction models on the
[SC³ corpus](../sc3/). Every method below is registered in a single
``METHOD_REGISTRY`` and dispatched through one CLI entry point. The tree,
descriptor-NN, GNN, and merged-graph branches share a common training,
caching, and evaluation infrastructure; method-specific code is confined to
``sc3_bench/models/``.

```
SDK/
├── README.md                  this file
├── requirements.txt
├── configs/
│   └── best_hps.json          best hyperparameters per method (search results)
├── sc3_bench/
│   ├── __init__.py
│   ├── registry.py            METHOD_REGISTRY (single source of truth)
│   ├── data.py                splits + feature cache loaders
│   ├── featurizers.py         RDKit / Mordred / Morgan / Dissolvr / graph
│   ├── train.py               unified per-seed training dispatcher
│   ├── evaluate.py            metrics (RMSE, PS-RMSE, Z-RMSE, ...)
│   ├── collect.py             aggregate per-method summaries → benchmark table
│   └── models/
│       ├── __init__.py
│       ├── tree_models.py     LightGBM / CatBoost / XGBoost / RF / DT / GP
│       ├── descriptor_models.py  FastProp / FastSolv / MLP
│       ├── gnn_models.py      GCN / GAT / GIN dual-encoder
│       ├── molmerger.py       AttentiveFP on merged solute-solvent graph
│       └── external/          additional baselines (see "External" below)
│           ├── __init__.py
│           ├── base.py        BaseMethod fit/predict protocol
│           ├── soltrannet.py
│           ├── unimol_method.py
│           ├── unifac_method.py
│           ├── solvaformer.py
│           ├── rilood.py
│           └── chemprop_method.py
├── scripts/
│   └── run_analyticals.py     non-trainable baselines (Abraham, ESOL, GSE, ...)
└── Ablations/                 ablation studies (code only — see READMEs)
    ├── Data_Scaling/
    ├── Interpretability/
    ├── Representation/
    └── Transfer/
```

---

## Method registry

The registry is the single source of truth for every method this SDK
supports. Each entry declares the method's family, the featurizer it
consumes, and the dispatch protocol used during training.

| Key                | Family            | Featurizer        | model_type     |
|--------------------|-------------------|-------------------|----------------|
| `lgb_rdkit`        | Desc + Tree       | RDKit             | tree           |
| `catboost_rdkit`   | Desc + Tree       | RDKit             | tree           |
| `xgb_rdkit`        | Desc + Tree       | RDKit             | tree           |
| `rf_rdkit`         | Desc + Tree       | RDKit             | tree           |
| `dt_rdkit`         | Desc + Tree       | RDKit             | tree           |
| `lgb_dissolvr`     | Domain (Dissolvr) | Dissolvr          | tree           |
| `tayyebi_mordred`  | Desc + Tree       | Mordred           | tree           |
| `lgb_morgan`       | FP + Tree         | Morgan            | tree           |
| `catboost_morgan`  | FP + Tree         | Morgan            | tree           |
| `xgb_morgan`       | FP + Tree         | Morgan            | tree           |
| `rf_morgan`        | FP + Tree         | Morgan            | tree           |
| `gp_morgan`        | FP + GP           | Morgan (Tanimoto) | tree           |
| `fastprop`         | Deep Descriptor   | RDKit             | descriptor_nn  |
| `fastprop_big`     | Deep Descriptor   | RDKit             | descriptor_nn  |
| `fastprop_xl`      | Deep Descriptor   | RDKit             | descriptor_nn  |
| `fastsolv`         | Deep Descriptor   | RDKit (+ T)       | descriptor_nn  |
| `mlp`              | Deep Descriptor   | RDKit             | descriptor_nn  |
| `gcn`              | GNN               | dual graph        | gnn            |
| `gat`              | GNN               | dual graph        | gnn            |
| `gin`              | GNN               | dual graph        | gnn            |
| `molmerger`        | Merged GNN        | merged graph      | molmerger      |
| `soltrannet`       | Transformer       | SMILES tokens     | external       |
| `unimol`           | Pretrained        | Uni-Mol2 (frozen) | external       |
| `unimol_catboost`  | Pretrained        | Uni-Mol2 (frozen) | external       |
| `unifac`           | Group Contrib     | UNIFAC groups     | external       |
| `solvaformer`      | SE(3) Transformer | 3D conformer      | external       |
| `rilood`           | Invariant GNN     | dual graph        | external       |
| `chemprop`         | GNN (D-MPNN)      | SMILES CSV        | external       |

Plus the analytical baselines run from `scripts/run_analyticals.py`:
`abraham_lfer`, `abraham_ml`, `esol`, `gse`.

---

## Dispatch protocol

`train.py` selects the per-seed trainer by ``model_type``:

| `model_type`    | Per-seed trainer            | Featurization                |
|-----------------|-----------------------------|------------------------------|
| `tree`          | `_train_tree_seed`          | from feature cache (`data.py`) |
| `descriptor_nn` | `_train_descriptor_nn_seed` | from feature cache           |
| `gnn`           | `_train_gnn_seed`           | dual-graph cache             |
| `molmerger`     | `_train_molmerger_seed`     | merged-skeleton cache        |
| `external`      | `_train_external_seed`      | method-specific (see below)  |

External methods declare an additional ``external_kind``:

| `external_kind` | What the dispatcher does                                                                 |
|-----------------|------------------------------------------------------------------------------------------|
| `base_method`   | Imports the entry-point class, instantiates `cls(seed, **params)`, calls `.fit()` then `.predict()` for each split. |
| `functional`    | Imports the entry-point module and calls its `train_chemprop`/`predict_chemprop` helpers.|
| `custom`        | Raises `NotImplementedError`. The model class is exposed for use by a hand-written training script — see the method's source docstring (this applies to `solvaformer` and `rilood`, both of which require bespoke optimization regimes). |

The full registry — including the entry-point string for each external
method — lives in `sc3_bench/registry.py`.

---

## Methods at a glance

### Tree / GP families
* **LightGBM / CatBoost / XGBoost / RF / DT** with RDKit physico-chemical
  descriptors and Morgan fingerprints. Parameters and seeding go through
  `tree_models.TREE_BUILDERS`. CatBoost is also used as a regression head
  for the UNIFAC and Uni-Mol2 features.
* **Tayyebi (Mordred)** — Random Forest on the Mordred descriptor set
  following Tayyebi et al. (2023).
* **Dissolvr** — domain-specific solute/solvent descriptors fed to LightGBM
  (Vermeire et al., 2022 inspired feature set).
* **GP (Tanimoto)** — Gaussian Process with the Tanimoto kernel on Morgan
  fingerprints.

### Descriptor neural nets
* **FastProp / FastProp-Big / FastProp-XL** — Kelvin-style RDKit descriptor
  MLP at three capacities (Vermeire 2022, Kelvin 2023).
* **FastSolv** — Sobolev-regularized descriptor MLP with explicit
  temperature features `(T, 1/T, T², log T)` and a temperature-derivative
  loss term.
* **MLP** — vanilla MLP baseline on RDKit descriptors.

### GNN dual-encoder
* **GCN / GAT / GIN** — share `DualGNNSolubility` from
  `models/gnn_models.py`, parameterized only by the convolution type.
* **MolMerger** — AttentiveFP applied to a merged solute-solvent skeleton
  with temperature stamping.

### External baselines

The implementations in `sc3_bench/models/external/` were contributed under
the `BaseMethod` protocol from `external/base.py`.

* **SolTranNet** — `SolTranNetMethod` — Molecule-Attention-Transformer
  (Maziarka et al., 2020) extended with a dual encoder for (solute,
  solvent) inputs and temperature features.
* **Uni-Mol2** — `UniMolMethod` (MLP head) and `UniMolCatBoostMethod`
  (CatBoost head). Uses the 84M-parameter Uni-Mol2 base model
  (Lu et al., arXiv:2406.14969) as a frozen feature extractor; the head is
  trained from scratch on each seed.
* **UNIFAC + CatBoost** — `UNIFACMLModel` — group-contribution UNIFAC
  residual with a CatBoost correction.
* **Solvaformer** — torch model class only. SE(3)-equivariant graph
  transformer with PaiNN-style scalar+vector features and a scalar
  cross-attention bridge between the solute and solvent encoders
  (Broadbent et al., arXiv:2511.09774). Training driver is bespoke; see
  the module docstring.
* **RIL-OOD** — torch model class only. Relational invariant learning with
  CIGIN-style bidirectional interaction maps + MCVAE + MCAR fusion (Chen
  et al., ICML 2025). Training driver is bespoke.
* **Chemprop D-MPNN** — functional wrapper over the chemprop CLI in
  multicomponent mode (Yang et al., 2019). The dispatcher invokes
  `train_chemprop` / `predict_chemprop` exposed by the module.

### Analyticals (no training)
Run from `scripts/run_analyticals.py`:

* **ESOL** (Delaney 2004), **GSE** (general solubility equation),
  **Abraham LFER** (linear free-energy relation), **Abraham ML** (Abraham
  parameters fit by ML).

---

## Reading the code

This SDK is intended to be code-readable end-to-end:

1. **Start at `sc3_bench/registry.py`** to see every method and its
   metadata in one place.
2. **`sc3_bench/train.py`** shows how every method is trained — each
   `_train_<kind>_seed` function is self-contained (~60-100 lines).
3. **`sc3_bench/models/`** contains the model definitions; nothing
   in this directory imports from the dispatcher.
4. **`sc3_bench/models/external/`** keeps the additional baselines
   (SolTranNet, Uni-Mol, UNIFAC, Solvaformer, RIL-OOD, Chemprop)
   self-contained — each file is independently readable.
5. **`Ablations/`** contains the four ablation studies referenced in the
   paper:
   - `Data_Scaling/` — performance vs training-set size sweeps.
   - `Interpretability/` — SHAP analyses of each featurizer + SHAP-on-graph
     for the GCN.
   - `Representation/` — featurizer ablation.
   - `Transfer/` — transfer learning from external (CombiSolv 298K)
     pretraining to the SC³ benchmark distribution.

Each ablation directory has its own `README.md` and (where applicable) a
`FINDINGS.md` summarizing the take-away.

---

## Reproducing results

> **Note for reviewers.** This SDK is shipped without trained-model
> weights, feature caches, or per-seed metric files. Everything the
> reviewer sees is code, configuration, and documentation. To reproduce
> a method end-to-end:

1. Install the dependencies from `requirements.txt` plus the
   method-specific extras listed below.
2. Build the feature cache from the SC³ splits in
   `../sc3/data/splits/`. The cache layout is documented in
   `sc3_bench/data.py` (`CACHE_DIR` constant) and is materialized via
   `sc3_bench.featurizers.cache_features(featurizer_name)`.
3. Train via the registry:

   ```python
   from sc3_bench.train import train_method
   train_method("lgb_rdkit", seeds=[42])             # tree
   train_method("fastsolv",  seeds=[42], gpu=0)      # descriptor NN
   train_method("gcn",       seeds=[42], gpu=0)      # GNN
   train_method("soltrannet",seeds=[42], gpu=0)      # external (BaseMethod)
   ```

4. Aggregate cross-method results with `sc3_bench.collect`.

### Method-specific dependencies

| Method                          | Extras                                                           |
|---------------------------------|------------------------------------------------------------------|
| `gcn` / `gat` / `gin` / `molmerger` | `torch`, `torch-geometric`                                   |
| `fastprop*` / `fastsolv` / `mlp`    | `torch`                                                      |
| `unimol*`                       | `unimol-tools` (downloads ~84M-parameter base model on first run) |
| `chemprop`                      | `chemprop` v1.x (CLI tools in PATH)                              |
| `solvaformer`                   | `torch`, `rdkit` (3D conformer generation via ETKDGv3)           |
| `rilood`                        | `torch`, `torch-geometric`, `rdkit`                              |
| `gp_morgan`                     | `gpytorch`                                                       |

### Hyperparameters

`configs/best_hps.json` ships the best hyperparameters discovered for
each in-house method through the search reported in the paper. External
methods that don't appear in the JSON fall back to their published
defaults (each method's source file documents the defaults inline).

---

## Conventions

* **Splits**: `train` / `eval` / `ood` / `sc3_gold` / `sc3_silver` /
  `sc3_bronze` — defined and produced by `../sc3/scripts/70_splits.py`.
* **Target**: `LogS` (decadic log-mole-fraction or molality solubility,
  per the column convention in `../sc3/DECISIONS.md`).
* **Seeds**: the canonical seed set is `[42, 101, 123, 456, 789]`
  (`registry.DEFAULT_SEEDS`).
* **Metrics**: `evaluate.compute_metrics` returns RMSE, MAE, R², plus
  the per-solvent and uncertainty-normalized variants used in the paper
  (`PS_RMSE`, `Z_RMSE`).
