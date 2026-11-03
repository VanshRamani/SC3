# SC³ — Benchmark & Modeling SDK

This repository accompanies the NeurIPS 2026 Datasets & Benchmarks paper
*SC³: A Multi-Solvent Solubility Challenge and Benchmark*
(Ramani, Arora, Kuchhal, Tatarin, Krasnov, Ranu, Karmakar). It ships
**(i)** the SC³ benchmark dataset together with its full curation
pipeline and **(ii)** a unified Python SDK for training and evaluating
solubility-prediction models on that benchmark.

The two halves are fully decoupled: the SDK consumes the splits the
curation pipeline produces, but neither imports the other.

```
.
├── README.md      this file
├── sc3/           Dataset + reproducible curation pipeline
└── SDK/           Modeling SDK + ablation studies
```

---

## Quick orientation

| If you want to …                                                    | Read this                                   |
|---------------------------------------------------------------------|---------------------------------------------|
| Understand the SC³ corpus and how it was built                      | [`sc3/README.md`](sc3/README.md)            |
| Inspect the shipped tier files (gold / silver / bronze)             | `sc3/data/sc3/`                             |
| Inspect the benchmark splits (train / eval / OOD)                   | `sc3/data/splits/`                          |
| Learn the methods catalogue and how training is dispatched          | [`SDK/README.md`](SDK/README.md)            |
| See the registry of every method in one place                       | `SDK/sc3_bench/registry.py`                 |
| Read a single method end-to-end                                     | files under `SDK/sc3_bench/models/`         |
| Read the four ablation studies                                      | `SDK/Ablations/{Data_Scaling,Interpretability,Representation,Transfer}/` |

---

## Repository layout (one screen)

```
sc3/
├── README.md                  Pipeline phases, shipped artifacts, reproduction
├── data/
│   ├── raw/                   BigSolDB v2.1 — placed by the user (see data/raw/README.md)
│   ├── interim/               Per-phase checkpoints (canonicalized → cleaned → ...)
│   ├── sc3/                   Final tiered dataset (gold/silver/bronze)
│   └── splits/                bench_train / bench_eval / bench_ood
└── scripts/                   Numbered curation pipeline (Phase 0 → Phase 8)

SDK/
├── README.md                  Method catalog, dispatch protocol, conventions
├── requirements.txt
├── configs/best_hps.json      Best hyperparameters per method
├── sc3_bench/                 Importable Python package
│   ├── registry.py            METHOD_REGISTRY (single source of truth)
│   ├── train.py               Per-seed training dispatcher
│   ├── data.py / featurizers.py / evaluate.py / collect.py
│   └── models/
│       ├── tree_models.py / descriptor_models.py / gnn_models.py / molmerger.py
│       └── external/          Additional baselines (SolTranNet, Uni-Mol2,
│                              UNIFAC, Solvaformer, RIL-OOD, Chemprop)
├── scripts/run_analyticals.py Non-trainable baselines (Abraham, ESOL, GSE, …)
└── Ablations/
    ├── Data_Scaling/
    ├── Interpretability/
    ├── Representation/
    └── Transfer/
```

---

## How the two halves connect

The SDK reads exactly four artifacts from `sc3/`:

| SDK consumer                          | sc3/ artifact                         |
|---------------------------------------|---------------------------------------|
| `sc3_bench.data.load_all_splits()`    | `sc3/data/splits/{bench_train,bench_eval,bench_ood}.csv` |
| Same loader                           | `sc3/data/sc3/{gold,silver,bronze}.csv` |
| `featurizers.cache_features()`        | The above CSV columns: `Solute_Canon`, `Solvent_Canon`, `Solvent_Name`, `Temperature_K`, `LogS`, `Uncertainty` |

Re-running the curation pipeline produces these CSVs deterministically;
the SDK has no other implicit dependency on `sc3/`.

---

## What is shipped vs. what isn't

This repository is **code, configuration, documentation, and
the curated dataset only**. To keep it lightweight, the following are **not** included:

* trained model weights (`*.pt`, `*.pth`, `*.cbm`, `*.pkl`),
* feature caches (`feature_cache/`, intermediate `*.npz`),
* per-run training logs and per-seed result JSONs,
* aggregated benchmark tables / experiment trackers,
* paper PDFs, slide decks, and other manuscript artifacts.

The shipped pieces — curation scripts, methods, hyperparameter configs,
and ablation drivers — are sufficient to reproduce every result in the
paper end-to-end.

---

## Reproduction at a glance

```bash
# 1. Install dependencies (see SDK/requirements.txt for the base set;
#    method-specific extras are listed in SDK/README.md).
pip install -r SDK/requirements.txt

# 2. Build the SC3 dataset from the raw BigSolDB v2.1 archive.
#    See sc3/README.md and sc3/data/raw/README.md for the upstream source.
cd sc3
python scripts/01_raw_audit.py
# … run scripts in numerical order through 81_multimodality.py

# 3. Build the feature cache from the produced splits.
cd ../SDK
python -c "from sc3_bench.featurizers import cache_features; cache_features('rdkit')"

# 4. Train any registered method (e.g. LightGBM on RDKit descriptors).
python -c "from sc3_bench.train import train_method; train_method('lgb_rdkit', seeds=[42])"

# 5. Aggregate results across methods.
python -c "from sc3_bench import collect; collect.build_table()"
```

Every method registered in `SDK/sc3_bench/registry.py` follows the same
`train_method(method_key, seeds=[...])` interface, regardless of whether
it is a tree, a descriptor MLP, a GNN, or an external baseline.

---

## Conventions

* **Splits** are named `train`, `eval`, `ood`, plus the consensus tiers
  `sc3_gold`, `sc3_silver`, `sc3_bronze`. Definitions live in
  `sc3/scripts/70_splits.py`.
* **Target column** is `LogS` = log₁₀(mole fraction).
* **Default seed set** is `[42, 101, 123, 456, 789]`
  (`SDK/sc3_bench/registry.py: DEFAULT_SEEDS`).
* **Metric suite** (`SDK/sc3_bench/evaluate.py`): RMSE, MAE, R²,
  per-solvent RMSE (PS-RMSE), uncertainty-normalised RMSE (Z-RMSE).
  Rationale for the non-standard variants is in `sc3/scripts/81_multimodality.py`.

---

## Citation

If you use SC³, please cite the paper and the upstream BigSolDB dataset:

```bibtex
@inproceedings{ramani2026sc3,
  title     = {{SC}$^3$: A Multi-Solvent Solubility Challenge and Benchmark},
  author    = {Ramani, Vansh and Arora, Har Ashish and Kuchhal, Dhairya and
               Tatarin, Sergei and Krasnov, Lev and Ranu, Sayan and Karmakar, Tarak},
  booktitle = {Advances in Neural Information Processing Systems
               (Datasets and Benchmarks Track)},
  year      = {2026},
  url       = {https://github.com/VanshRamani/SC3}
}

@article{krasnov2025bigsoldb,
  title   = {BigSolDB 2.0, dataset of solubility values for organic compounds
             in different solvents at various temperatures},
  author  = {Krasnov, Lev and Malikov, Dmitry and Kiseleva, Marina and
             Tatarin, Sergei V. and Sosnin, Sergey and Bezzubov, Stanislav},
  journal = {Scientific Data},
  volume  = {12},
  pages   = {1236},
  year    = {2025},
  doi     = {10.1038/s41597-025-05559-8}
}
```

SC³ is built on the BigSolDB v2.1 tabular release
(<https://doi.org/10.5281/zenodo.18552681>, CC-BY-4.0).

---

## License

* **Code** (`SDK/`, `sc3/scripts/`): MIT License — see [`LICENSE`](LICENSE).
* **Data** (`sc3/data/`, `sc3/croissant.json`): Creative Commons
  Attribution 4.0 International (CC-BY-4.0) — see
  [`LICENSE-DATA`](LICENSE-DATA). The data are a curated derivative of
  BigSolDB (CC-BY-4.0); please credit both SC³ and BigSolDB.
