# SC³ — Solubility Curation & Consistency Corpus

This directory contains the curation pipeline and shipped data artifacts for
the **SC³ benchmark**: a tiered, thermodynamically-consistent multi-solvent
solubility dataset derived from BigSolDB v2.1.

The pipeline is reproducible: starting from the raw archive, every script
under `scripts/` is numbered by phase and produces a deterministic artifact
in `data/`. The shipped tiers (`gold`, `silver`, `bronze`) and the
benchmark splits (`bench_train`, `bench_eval`, `bench_ood`) are the inputs
consumed by every method in the companion SDK (`../SDK/`).

---

## Directory layout

```
sc3/
├── DECISIONS.md            Curation decision log (every "D-XX" rule)
├── croissant.json          MLCommons Croissant 1.1 + RAI 1.0 metadata
│                           (regenerate via scripts/90_generate_croissant.py)
├── data/
│   ├── raw/                BigSolDB v2.1 (place archive here — see data/raw/README.md)
│   ├── interim/            Intermediate per-phase artifacts (canonicalized,
│   │                       cleaned, copycat-flagged, Apelblat fits, ...)
│   ├── sc3/                Final tiered dataset (gold / silver / bronze)
│   └── splits/             Benchmark splits (train / eval / OOD)
└── scripts/                Numbered curation pipeline (Phase 0 → Phase 8)
                            plus 90_generate_croissant.py (metadata)
```

---

## Curation pipeline

Scripts are grouped by phase. Each script reads from `data/interim/` (or
`data/raw/` for Phase 0) and writes the next intermediate artifact.
Re-running a phase from scratch reproduces every downstream artifact bit-for-bit.

| Phase | Script(s) | Purpose |
|-------|-----------|---------|
| 0 — Audit | `01_raw_audit.py`, `02_deep_audit.py`, `03_targeted_checks.py` | Sanity-check the raw archive: row counts, suspect DOIs, residual patterns. |
| 1 — Canonicalize | `10_canonicalize.py`, `11_merge_audit.py`, `15_apply_manual_corrections.py` | Canonicalize SMILES under decision **D-01** (Option D); audit every merge group; apply manual corrections from the domain expert. |
| 2 — Clean | `20_clean.py` | Cleaning waterfall (units, duplicates, malformed rows, temperature ranges). |
| 3 — Source integrity | `30_source_integrity.py`, `45_source_integrity_interp.py`, `55_threshold_sensitivity.py` | Detect bit-exact and gray-zone copycats; rank DOIs by reliability. |
| 4 — Thermodynamic fits | `40_apelblat.py` | Per-(solute, solvent, group) Apelblat / van't Hoff fits for interpolation. |
| 5 — Aleatoric limit | `50_aleatoric.py` | Estimate the irreducible aleatoric noise floor (D-12 / D-14). |
| 6 — Tiers | `60_tiers.py` | Build SC³ gold / silver / bronze tiers per **D-15**. |
| 7 — Splits | `70_splits.py` | Construct benchmark train / eval / OOD splits. |
| 8 — Metrics | `80_metrics.py`, `81_multimodality.py` | Metric definitions (importable module) + motivating analysis. |
| 9 — Metadata | `90_generate_croissant.py` | Build/refresh the Croissant 1.1 + RAI 1.0 metadata file (`croissant.json`). |

The full decision log — every `D-XX` referenced above — lives in
[`DECISIONS.md`](DECISIONS.md).

---

## Shipped artifacts

The benchmark consumers in `../SDK/` read from:

| File | Rows | Description |
|------|-----:|-------------|
| `data/sc3/gold.csv` | tier-A pairs | Highest-confidence pairs (post-Apelblat-consistent, multi-source agreement). |
| `data/sc3/silver.csv` | tier-B pairs | Single-source consistent pairs that pass copycat filtering. |
| `data/sc3/bronze.csv` | tier-C pairs | Best-effort consensus where higher tiers are unavailable. |
| `data/sc3/tier_pairs.csv` | all tiers | Long-form tier assignments. |
| `data/sc3/tier_summary.json` | — | Tier population, mean σ, etc. |
| `data/splits/bench_train.csv` | train | All-tier training rows for the benchmark. |
| `data/splits/bench_eval.csv` | eval | In-distribution evaluation rows. |
| `data/splits/bench_ood.csv` | OOD | Held-out held-out solute and held-out solvent rows. |

Column conventions are documented in `DECISIONS.md` and asserted in `80_metrics.py`.

---

## Reproducing from raw

1. Place the BigSolDB v2.1 archive under `data/raw/bigsoldb_v2.1/` (see
   `data/raw/README.md`).
2. Run the phases in order:
   ```bash
   python scripts/01_raw_audit.py
   python scripts/02_deep_audit.py
   python scripts/03_targeted_checks.py
   python scripts/10_canonicalize.py
   python scripts/11_merge_audit.py
   python scripts/15_apply_manual_corrections.py
   python scripts/20_clean.py
   python scripts/30_source_integrity.py
   python scripts/40_apelblat.py
   python scripts/45_source_integrity_interp.py
   python scripts/55_threshold_sensitivity.py
   python scripts/50_aleatoric.py
   python scripts/60_tiers.py
   python scripts/70_splits.py
   python scripts/81_multimodality.py
   ```

   `80_metrics.py` is an importable module; it is not called as a stand-alone
   step.

3. Inspect `data/sc3/` and `data/splits/` for the produced artifacts.

---

## Dependencies

The curation pipeline uses only well-established scientific-Python tooling.
A minimal environment is:

```
python>=3.10
numpy
pandas
scipy
scikit-learn
rdkit
matplotlib   # only for downstream figure scripts (not shipped here)
```

No GPU is required for any step in this directory.

---

## Notes for reviewers

- Every non-trivial design choice in this pipeline is logged in `DECISIONS.md`
  with a date stamp and rationale. Where a manual correction was applied
  (Phase 1.5), the source (e.g., expert annotation transcript) is cited in
  the decision entry.
- Intermediate artifacts in `data/interim/` are checkpoints — they let you
  jump into the pipeline at any phase without re-running upstream steps.
- This directory is **data + reproducible curation only**. No model
  predictions, training logs, or evaluation tables live here; those belong
  to the modeling SDK in `../SDK/`.

---

## Croissant metadata

`croissant.json` ships a self-contained MLCommons Croissant 1.1 + RAI 1.0
description of every shipped artifact (the three tier CSVs, the long-form
tier-pair table, and the three benchmark splits). The file conforms to:

- `http://mlcommons.org/croissant/1.1` — Croissant core spec.
- `http://mlcommons.org/croissant/RAI/1.0` — Responsible-AI extension. The
  full NeurIPS 2026 minimal RAI set (`rai:dataLimitations`, `rai:dataBiases`,
  `rai:personalSensitiveInformation`, `rai:dataUseCases`, `rai:dataSocialImpact`,
  `rai:hasSyntheticData`, `prov:wasDerivedFrom`, `prov:wasGeneratedBy`) is
  populated, plus the extended RAI block (`rai:dataCollection*`,
  `rai:data*Protocol`, `rai:dataAnnotation*`, `rai:dataReleaseMaintenancePlan`).
  Fields that are not applicable to a literature-curation pipeline
  (annotators, demographics, annotation platforms) are explicitly marked
  *"Not applicable"* with a one-line justification rather than left blank.

For each shipped CSV, the metadata records the byte size, a SHA-256
checksum computed at build time, the encoding (`text/csv`), and a
per-column field map with `dataType` (`sc:Text`, `sc:Float`, `sc:Integer`,
`sc:Boolean`).

### Regenerating

```bash
# defaults target the anonymous review repo SC3-Benchmark; override via env
# vars to retarget (e.g., for the camera-ready public release):
SC3_ANON_REPO_URL=https://anonymous.4open.science/r/SC3-Benchmark \
SC3_DATASET_HOMEPAGE=https://anonymous.4open.science/r/SC3-Benchmark \
    python scripts/90_generate_croissant.py
```

The generator recomputes file sizes and SHA-256 hashes from the local CSVs
on every run, so it stays consistent with whatever rebuild of the dataset
the curator just ran. It uses only the Python standard library (no
`mlcroissant` dependency required).

### Validating

The Croissant editor and checker live online and accept either an uploaded
file or a URL pointing to the metadata file:

- Croissant Checker — <https://huggingface.co/spaces/mlcommons/croissant-checker>
- Croissant Editor —  <https://huggingface.co/spaces/MLCommons/croissant-editor>

The anonymous repo is live at
`https://anonymous.4open.science/r/SC3-Benchmark/`. Paste
`https://anonymous.4open.science/r/SC3-Benchmark/sc3/croissant.json`
into the checker. Locally, on Python ≥ 3.10:

```bash
pip install mlcroissant
mlcroissant validate --jsonld croissant.json
```

> **Note for the camera-ready release.** When the dataset moves from the
> anonymous review URL to its public home (CC-BY-4.0 on Hugging Face
> Datasets or Zenodo per `rai:dataReleaseMaintenancePlan`), simply rerun
> the generator with the new URL exported via `SC3_ANON_REPO_URL` /
> `SC3_DATASET_HOMEPAGE` and re-validate.
