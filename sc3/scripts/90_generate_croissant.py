"""Generate ``SUBMISSION/sc3/croissant.json``.

The output is a self-contained JSON-LD document that conforms to:

* MLCommons Croissant 1.0  -- ``http://mlcommons.org/croissant/1.0``
* MLCommons RAI 1.0        -- ``http://mlcommons.org/croissant/RAI/1.0``

It is intended for the SC3 anonymous NeurIPS submission.  We deliberately
build the JSON-LD with the standard library only so the script runs on
any Python >= 3.8 without pulling ``mlcroissant`` (which requires Python
3.10+).

Override the anon-repo URL via the ``SC3_ANON_REPO_URL`` env var, e.g.:

    SC3_ANON_REPO_URL=https://anonymous.4open.science/r/SC3-0371 \\
        python scripts/90_generate_croissant.py

Defaults match the URL the curator submitted to NeurIPS 2026 D&B.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

THIS_FILE = Path(__file__).resolve()
SC3_ROOT = THIS_FILE.parent.parent
OUTPUT_PATH = SC3_ROOT / "croissant.json"

ANON_REPO_URL = os.environ.get(
    "SC3_ANON_REPO_URL",
    "https://anonymous.4open.science/r/SC3-0371",
).rstrip("/")
DATASET_HOMEPAGE = os.environ.get(
    "SC3_DATASET_HOMEPAGE",
    "https://anonymous.4open.science/status/SC3-0371",
)

# ---------------------------------------------------------------------------
# Field type vocabulary
# ---------------------------------------------------------------------------

TEXT = "sc:Text"
FLOAT = "sc:Float"
INT = "sc:Integer"
BOOL = "sc:Boolean"
URL = "sc:URL"

# ---------------------------------------------------------------------------
# Shipped data files (relative to SC3_ROOT)
# ---------------------------------------------------------------------------

# Schema for the three tier files (gold / silver / bronze) is identical.
TIER_FIELDS = [
    ("Solute_Canon", TEXT,
     "RDKit canonical SMILES of the solute (isomericSmiles=True; chirality and "
     "geometric isomerism preserved per decision D-01 'Option D')."),
    ("Solvent_Canon", TEXT,
     "RDKit canonical SMILES of the solvent."),
    ("Temperature_K", FLOAT,
     "Reference temperature in Kelvin at which the consensus log S is reported. "
     "Restricted to measured temperatures only (no extrapolation; see D-15)."),
    ("LogS_consensus", FLOAT,
     "Consensus log10 solubility computed as the mean of fitted Apelblat / "
     "van't Hoff curves (one per non-Hall-of-Shame independence group whose fit "
     "range covers Temperature_K). Units follow D-15 (decadic log mole-fraction "
     "or molality, consistent with the upstream BigSolDB v2.1 column)."),
    ("sigma", FLOAT,
     "Sample standard deviation (ddof=1) of contributing fits at Temperature_K, "
     "floored at 0.012 log S. NaN when only one independence group's fit covers "
     "Temperature_K (~23% of rows)."),
    ("n_contributing_groups", INT,
     "Number of independence groups whose Apelblat/van't Hoff fit range covers "
     "Temperature_K and contributes to LogS_consensus / sigma."),
    ("pair_MAE", FLOAT,
     "Pair-level inter-group MAE (log S) used to assign the (solute, solvent) "
     "pair to a tier: Gold <= 0.1, Silver <= 0.2, Bronze <= 0.5."),
]

TIER_PAIRS_FIELDS = [
    ("Solute_Canon", TEXT,
     "RDKit canonical SMILES of the solute."),
    ("Solvent_Canon", TEXT,
     "RDKit canonical SMILES of the solvent."),
    ("pair_MAE", FLOAT,
     "Pair-level inter-group MAE in log S."),
    ("n_contributing_groups", INT,
     "Number of independence groups in this pair (post-Stage A + Stage B' merging)."),
    ("n_rows", INT,
     "Total measured-temperature rows for this pair."),
    ("n_rows_sigma_defined", INT,
     "Number of rows whose sigma is finite (i.e., >=2 contributing groups at that T)."),
    ("median_sigma", FLOAT,
     "Median of the per-row sigma values (NaN-safe; computed over sigma-defined rows)."),
    ("tier_gold", BOOL, "True if the pair belongs to the Gold tier (pair_MAE <= 0.1)."),
    ("tier_silver", BOOL, "True if the pair belongs to the Silver tier (pair_MAE <= 0.2)."),
    ("tier_bronze", BOOL, "True if the pair belongs to the Bronze tier (pair_MAE <= 0.5)."),
]

BENCH_FIELDS = [
    ("Solute_Canon", TEXT,
     "RDKit canonical SMILES of the solute."),
    ("Solvent_Canon", TEXT,
     "RDKit canonical SMILES of the solvent."),
    ("Solvent", TEXT,
     "Human-readable solvent name from the upstream BigSolDB row. "
     "Informational only -- the curation pipeline keys on Solvent_Canon."),
    ("Temperature_K", FLOAT,
     "Measurement temperature in Kelvin."),
    ("LogS", FLOAT,
     "Decadic log solubility as reported in the cleaned BigSolDB row. "
     "Pre-tier construction; this is the per-measurement label, not the "
     "consensus value."),
    ("Solubility(mole_fraction)", FLOAT,
     "Solubility expressed as a mole fraction, as carried through from "
     "BigSolDB v2.1."),
    ("MW", FLOAT,
     "Molecular weight (g/mol) of the solute, computed via "
     "rdkit.Chem.Descriptors.MolWt on Solute_Canon."),
    ("Source", TEXT,
     "DOI of the publication that reported this measurement (or the BigSolDB "
     "DOI alias when the upstream row carries no DOI)."),
]

FILE_SPECS = [
    {
        "id": "sc3_gold",
        "rel_path": "data/sc3/gold.csv",
        "name": "sc3-gold",
        "description": (
            "SC3 Gold tier: 335 (solute, solvent) pairs across 26 solvents, "
            "4,507 measured-temperature rows. Pair-level inter-lab MAE <= 0.1 "
            "log S. Highest-confidence multi-source consensus rows; sigma is "
            "defined for 76.5% of rows. Decision D-15."
        ),
        "fields": TIER_FIELDS,
        "key": ["Solute_Canon", "Solvent_Canon", "Temperature_K"],
    },
    {
        "id": "sc3_silver",
        "rel_path": "data/sc3/silver.csv",
        "name": "sc3-silver",
        "description": (
            "SC3 Silver tier: 400 pairs / 5,475 rows, pair-level MAE <= 0.2 "
            "log S. Superset of Gold (Gold subset Silver). Decision D-15."
        ),
        "fields": TIER_FIELDS,
        "key": ["Solute_Canon", "Solvent_Canon", "Temperature_K"],
    },
    {
        "id": "sc3_bronze",
        "rel_path": "data/sc3/bronze.csv",
        "name": "sc3-bronze",
        "description": (
            "SC3 Bronze tier: 469 pairs / 6,331 rows, pair-level MAE <= 0.5 "
            "log S. Superset of Silver. Decision D-15."
        ),
        "fields": TIER_FIELDS,
        "key": ["Solute_Canon", "Solvent_Canon", "Temperature_K"],
    },
    {
        "id": "sc3_tier_pairs",
        "rel_path": "data/sc3/tier_pairs.csv",
        "name": "sc3-tier-pairs",
        "description": (
            "Long-form tier-membership table: one row per tier-eligible "
            "(solute, solvent) pair (n=469), with pair-level statistics and "
            "boolean tier flags. Drives the per-pair tier assignments used by "
            "gold.csv / silver.csv / bronze.csv."
        ),
        "fields": TIER_PAIRS_FIELDS,
        "key": ["Solute_Canon", "Solvent_Canon"],
    },
    {
        "id": "bench_train",
        "rel_path": "data/splits/bench_train.csv",
        "name": "bench-train",
        "description": (
            "Benchmark training split. 61,403 rows, 1,144 solutes, 25 ID "
            "solvents, 6,840 (solute, solvent) pairs. Anti-leakage: every "
            "solute that appears in any tier (148 solutes) is excluded from "
            "training under all solvents. Decision D-16."
        ),
        "fields": BENCH_FIELDS,
        "key": ["Solute_Canon", "Solvent_Canon", "Temperature_K", "Source"],
    },
    {
        "id": "bench_eval",
        "rel_path": "data/splits/bench_eval.csv",
        "name": "bench-eval",
        "description": (
            "In-distribution evaluation split: 6,969 rows, 534 solutes, 25 "
            "ID solvents, 771 (solute, solvent) pairs. 10% of pairs per ID "
            "solvent held out at the pair level (seed 42). All temperatures "
            "of a held-out pair stay together. Decision D-16."
        ),
        "fields": BENCH_FIELDS,
        "key": ["Solute_Canon", "Solvent_Canon", "Temperature_K", "Source"],
    },
    {
        "id": "bench_ood",
        "rel_path": "data/splits/bench_ood.csv",
        "name": "bench-ood",
        "description": (
            "Out-of-distribution split (unseen solvents): 11,940 rows, 586 "
            "solutes, 161 OOD solvents, 1,450 pairs. Stress test for solvent "
            "generalization. Decision D-16."
        ),
        "fields": BENCH_FIELDS,
        "key": ["Solute_Canon", "Solvent_Canon", "Temperature_K", "Source"],
    },
]


# ---------------------------------------------------------------------------
# JSON-LD helpers
# ---------------------------------------------------------------------------

CONTEXT = OrderedDict([
    ("@language", "en"),
    ("@vocab", "https://schema.org/"),
    ("citeAs", "cr:citeAs"),
    ("column", "cr:column"),
    ("conformsTo", "dct:conformsTo"),
    ("cr", "http://mlcommons.org/croissant/"),
    ("rai", "http://mlcommons.org/croissant/RAI/"),
    ("data", OrderedDict([("@id", "cr:data"), ("@type", "@json")])),
    ("dataType", OrderedDict([("@id", "cr:dataType"), ("@type", "@vocab")])),
    ("dct", "http://purl.org/dc/terms/"),
    ("examples", OrderedDict([("@id", "cr:examples"), ("@type", "@json")])),
    ("extract", "cr:extract"),
    ("field", "cr:field"),
    ("fileProperty", "cr:fileProperty"),
    ("fileObject", "cr:fileObject"),
    ("fileSet", "cr:fileSet"),
    ("format", "cr:format"),
    ("includes", "cr:includes"),
    ("isLiveDataset", "cr:isLiveDataset"),
    ("jsonPath", "cr:jsonPath"),
    ("key", "cr:key"),
    ("md5", "cr:md5"),
    ("parentField", "cr:parentField"),
    ("path", "cr:path"),
    ("recordSet", "cr:recordSet"),
    ("references", "cr:references"),
    ("regex", "cr:regex"),
    ("repeated", "cr:repeated"),
    ("replace", "cr:replace"),
    ("sc", "https://schema.org/"),
    ("separator", "cr:separator"),
    ("source", "cr:source"),
    ("subField", "cr:subField"),
    ("transform", "cr:transform"),
])


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _row_count(path: Path) -> int:
    """Count data rows in a CSV (excluding header)."""
    with open(path, newline="") as fh:
        return sum(1 for _ in csv.reader(fh)) - 1


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_file_object(spec: dict[str, Any]) -> OrderedDict:
    rel_path = spec["rel_path"]
    abs_path = SC3_ROOT / rel_path
    if not abs_path.is_file():
        raise FileNotFoundError(f"Missing data file: {abs_path}")
    return OrderedDict([
        ("@type", "cr:FileObject"),
        ("@id", spec["id"]),
        ("name", spec["name"]),
        ("description", spec["description"]),
        ("contentUrl", f"{ANON_REPO_URL}/SUBMISSION/sc3/{rel_path}"),
        ("encodingFormat", "text/csv"),
        ("contentSize", f"{abs_path.stat().st_size} B"),
        ("sha256", _sha256(abs_path)),
    ])


def build_record_set(spec: dict[str, Any]) -> OrderedDict:
    file_id = spec["id"]
    rs_id = f"{file_id}_records"
    fields = []
    for col, dtype, desc in spec["fields"]:
        fields.append(OrderedDict([
            ("@type", "cr:Field"),
            ("@id", f"{rs_id}/{col}"),
            ("name", col),
            ("description", desc),
            ("dataType", dtype),
            ("source", OrderedDict([
                ("fileObject", OrderedDict([("@id", file_id)])),
                ("extract", OrderedDict([("column", col)])),
            ])),
        ]))
    rs = OrderedDict([
        ("@type", "cr:RecordSet"),
        ("@id", rs_id),
        ("name", spec["name"].replace("-", "_")),
        ("description", spec["description"]),
        ("field", fields),
    ])
    if spec.get("key"):
        rs["key"] = [
            OrderedDict([("@id", f"{rs_id}/{c}")]) for c in spec["key"]
        ]
    return rs


# ---------------------------------------------------------------------------
# Top-level dataset
# ---------------------------------------------------------------------------

def build_dataset() -> OrderedDict:
    distribution = [build_file_object(s) for s in FILE_SPECS]
    record_set = [build_record_set(s) for s in FILE_SPECS]

    description = (
        "SC3 (Solubility Curation & Consistency Corpus) is a tiered, "
        "thermodynamically-consistent multi-solvent solid-solubility dataset "
        "derived from BigSolDB v2.1 (Krasnov et al., 2024). The corpus ships "
        "three nested quality tiers -- Gold (335 pairs / 4,507 measurements, "
        "pair-level inter-lab MAE <= 0.1 log S), Silver (400 / 5,475, MAE "
        "<= 0.2), and Bronze (469 / 6,331, MAE <= 0.5) -- plus three solute-"
        "disjoint benchmark splits: bench_train (61,403 rows), bench_eval "
        "(6,969 rows, in-distribution), and bench_ood (11,940 rows, unseen "
        "solvents). Every (solute, solvent, T) row is a literature-reported "
        "measurement; consensus tier labels are the mean of independent "
        "Apelblat/van't Hoff fits, with per-point sigma calibrated from "
        "fit-level disagreement (floored at 0.012 log S). The empirical "
        "aleatoric noise floor is epsilon_A = 0.11 log S. The full curation "
        "decision log lives in DECISIONS.md and the reproducible pipeline is "
        "shipped under scripts/01..81 (Phases 0-8)."
    )

    cite_as = (
        r"@misc{sc3_anon_2026,"
        r"  title  = {SC$^3$: Solubility Curation \& Consistency Corpus},"
        r"  author = {Anonymous},"
        r"  year   = {2026},"
        r"  note   = {NeurIPS 2026 Datasets and Benchmarks Track (under review). "
        r"Anonymous submission.},"
        r"  url    = {" + DATASET_HOMEPAGE + r"}"
        r"}"
    )

    rai = OrderedDict([
        ("rai:dataCollection",
         "SC3 is a curated derivative of BigSolDB v2.1 (Krasnov et al., 2024), "
         "an open-source compilation of solid-solubility measurements harvested "
         "from peer-reviewed publications and distributed via figshare "
         "(https://doi.org/10.6084/m9.figshare.21118034). On top of the upstream "
         "archive, SC3 applies (i) RDKit canonicalization preserving stereo and "
         "geometric isomerism (decision D-01 'Option D'); (ii) a manually-audited "
         "bad-DOI exclusion list jointly maintained with the BigSolDB v2.1 "
         "co-maintainer (D-03, D-12); (iii) two-stage copycat detection -- "
         "bit-exact (D-05 Stage A) and interpolated MAE-based (D-14 Stage B'); "
         "(iv) per-independence-group Apelblat / van't Hoff fits (D-13); "
         "(v) inter-lab aleatoric noise estimation (D-13, D-14); (vi) three "
         "nested tiers (D-15); and (vii) anti-leakage solute-disjoint "
         "train/eval/OOD splits (D-16)."),

        ("rai:dataCollectionType",
         "Curated derivative of a literature compilation. No human subjects, no "
         "annotators, no demographic data: every row corresponds to a published "
         "thermodynamic measurement."),

        ("rai:dataCollectionMissingData",
         "(a) Per-row sigma is NaN for ~23% of tier rows where only one "
         "independence group's fit range covers the reference temperature "
         "(documented in tier_summary.json). "
         "(b) The bench_* splits expose LogS, not sigma; sigma is defined only "
         "for tier rows and is intentionally absent from training rows. "
         "(c) The Solvent (human-readable name) column is informational; the "
         "pipeline keys on Solvent_Canon (SMILES). "
         "(d) Rows with logS outside [-15, 2], MW > 1000 Da, polymeric solvents, "
         "or salt/mixture solutes are dropped, not imputed (D-07/D-08/D-09)."),

        ("rai:dataCollectionRawData",
         "BigSolDB v2.1 (Krasnov et al., 'BigSolDB: solubility dataset of "
         "compounds in organic solvents and water in a wide range of "
         "temperatures', 2024; figshare DOI 10.6084/m9.figshare.21118034). "
         "SC3 does not redistribute the raw BigSolDB rows. Reviewers and "
         "downstream users must download the upstream archive separately and "
         "re-run scripts/01..81 to regenerate the SC3 artifacts (the pipeline "
         "is deterministic; row-by-row reproducibility is asserted by "
         "scripts/03_targeted_checks.py)."),

        ("rai:dataCollectionTimeframe",
         "Curation v2 was conducted between Q4 2025 and Q1 2026. The underlying "
         "BigSolDB v2.1 source measurements are drawn from peer-reviewed "
         "publications dating up to 2025."),

        ("rai:dataImputationProtocol",
         "No imputation. Rows with missing or unparseable SMILES, missing "
         "temperatures, missing logS, or values outside the validity bounds in "
         "decisions D-07/D-08/D-09 are dropped rather than imputed."),

        ("rai:dataPreprocessingProtocol",
         "Eight-phase deterministic pipeline (see DECISIONS.md and "
         "scripts/01..81): "
         "Phase 0 audit -> Phase 1 SMILES canonicalization (RDKit "
         "isomericSmiles=True, no tautomer enumeration, no chirality stripping; "
         "D-01) -> Phase 1.5 manual corrections (4 DOIs, D-12) -> Phase 2 "
         "cleaning waterfall (bad DOIs, polymers, salts, MW <=1000 Da, "
         "logS in [-15, 2], intra-DOI rounding dedupe; D-03/D-07/D-08/D-09/D-10) "
         "-> Phase 3 source integrity (Stage A bit-exact copycats; D-05) "
         "-> Phase 4 Apelblat/van't Hoff fits per independence group "
         "-> Phase 3B (45_*) interpolated Stage B' copycats and Stage C' "
         "reliability (D-14) -> Phase 5 aleatoric estimation epsilon_A = 0.11 "
         "log S (D-13) -> Phase 6 tier construction Gold/Silver/Bronze "
         "(D-15) -> Phase 7 splits train/eval/OOD with solute-level "
         "anti-leakage (D-16) -> Phase 8 metric suite RMSE/MAE/MedAE/PS-RMSE/"
         "Z-RMSE (D-17)."),

        ("rai:dataManipulationProtocol",
         "Solute and solvent SMILES are RDKit-canonicalized once at Phase 1 "
         "and remain immutable thereafter. Temperatures are rounded to 0.1 K "
         "only for intra-DOI duplicate detection at Phase 2 W7. LogS values "
         "from four DOIs are corrected at Phase 1.5 per the manual audit "
         "(D-12): two DOIs receive +1.0 log shifts, one receives a "
         "ethanol/ethyl-acetate label swap with re-derivation, one receives "
         "an explicit value replacement at four temperatures. All other LogS "
         "values are passed through unmodified."),

        ("rai:dataAnnotationProtocol",
         "Not applicable: SC3 is fully algorithmic curation. The two 'manual' "
         "elements -- the bad-DOI list (D-03) and the targeted corrections "
         "(D-12) -- are domain-expert review of automatically-flagged "
         "candidates; both are recorded in DECISIONS.md with the supporting "
         "evidence (residual analyses, co-maintainer audit log, "
         "PubChem/CAS verification) cited inline."),

        ("rai:dataAnnotationPlatform", "Not applicable (no annotation task)."),
        ("rai:dataAnnotationAnalysis", "Not applicable (no annotation task)."),
        ("rai:annotationsPerItem",
         "Not applicable. Each (solute, solvent, T) row is a single "
         "literature-reported equilibrium measurement; for tier rows, the "
         "n_contributing_groups column reports the number of independent "
         "literature sources whose fits cover that temperature."),
        ("rai:annotatorDemographics", "Not applicable (no human annotators)."),
        ("rai:machineAnnotationTools",
         "RDKit (canonicalization, MW, parseability), SciPy (Apelblat / "
         "van't Hoff non-linear least-squares fits), scikit-learn (split "
         "construction with seed 42), pandas/numpy (cleaning waterfall, "
         "groupwise statistics)."),

        ("rai:dataBiases",
         "(1) Solvent count imbalance -- top-5 solvents account for 37.5% of "
         "cleaned rows; top-25 cover 84.5%. Aggregate RMSE is dominated by "
         "water/ethanol/methanol; PS-RMSE (per-solvent macro-average) "
         "mitigates this. "
         "(2) Pharma-leaning solute distribution -- BigSolDB sources skew "
         "toward drug-like molecules; petrochemicals, organometallics, dyes, "
         "and polymers will be out-of-distribution. "
         "(3) Salt/hydrate exclusion -- D-08 drops multi-component solutes, so "
         "pharmaceutically relevant salt solubilities are absent. "
         "(4) Stereo coverage gap -- D- and L- enantiomers are unevenly "
         "represented in BigSolDB; under D-01 (Option D) they are kept as "
         "distinct solutes. "
         "(5) Temperature coverage -- nominal range 243-426 K is heavily "
         "concentrated at 293-323 K; sub-zero and high-T behaviour is poorly "
         "covered. "
         "(6) Reliability coverage -- ~80% of DOIs (1,094 / 1,493) have no "
         "peer-overlapping group, hence cannot be reliability-tested under "
         "D-14 Stage C'. They enter the training pool but their per-row "
         "error bars are uncalibrated."),

        ("rai:dataUseCases",
         "(a) Benchmarking SMILES -> log S regression models across "
         "in-distribution (Eval), unseen-solvent (OOD), and high-quality "
         "consensus (Gold/Silver/Bronze) splits. "
         "(b) Calibration analysis: comparing model error to the empirical "
         "aleatoric floor epsilon_A = 0.11 log S using the Z-RMSE metric "
         "(D-17). Z-RMSE = 1 means model error matches measurement noise. "
         "(c) Solvent-balanced evaluation via PS-RMSE, designed to strip the "
         "count-weighting + between-solvent inflation that contaminates "
         "aggregate RMSE on multi-solvent solubility data. "
         "(d) Studies of label noise, multi-source disagreement, and "
         "inter-lab reproducibility in physico-chemical literature data."),

        ("rai:dataLimitations",
         "(1) Not exhaustive: only (solute, solvent, T) triples surviving "
         "D-01..D-15 are retained. Many BigSolDB rows are intentionally "
         "filtered. "
         "(2) Equilibrium thermodynamic solubility only -- not a kinetics or "
         "intrinsic-solubility dataset. "
         "(3) Crystal-form information is lost: polymorphs, solvates, salt "
         "forms, and amorphous forms are either filtered or merged. "
         "(4) The aleatoric floor epsilon_A = 0.11 log S is a lower bound for "
         "any error metric on Gold; a model that reports lower RMSE on Gold is "
         "fitting the consensus label, not improving over the literature. "
         "(5) The OOD split is a stress test, not a uniform evaluation: many "
         "OOD solvents have <10 rows. "
         "(6) Source measurements pre-date the curation; outliers can be "
         "flagged but not re-measured. "
         "(7) Splits are stratified by solute identity, not by publication year "
         "-- there is no temporal/causal split."),

        ("rai:dataSocialImpact",
         "SC3 targets molecular-property prediction for drug-likeness and "
         "process-chemistry solvent selection. Downstream models trained on "
         "SC3 may inform pharmaceutical formulation decisions and green-"
         "solvent screening pipelines. The dataset itself contains no "
         "human-impacting decision signal: only molecular structures, "
         "thermodynamic measurements, and DOI source identifiers. Indirect "
         "risks: a model that systematically under-predicts solubility for "
         "under-represented solvents or solute classes may, if deployed in "
         "early-stage screening, bias which compounds advance. This is "
         "explicitly mitigated by (a) reporting PS-RMSE alongside RMSE, "
         "(b) shipping the OOD split to expose unseen-solvent generalization, "
         "(c) shipping per-row sigma so downstream predictions can be "
         "uncertainty-weighted."),

        ("rai:personalSensitiveInformation",
         "None. SC3 contains only molecular structures (SMILES), "
         "thermodynamic measurements (T in K, log S, mole fraction, MW), and "
         "DOI source identifiers. No personally identifiable information, "
         "biometric data, demographic data, health data, or other sensitive "
         "categories are present at any phase of the pipeline."),

        ("rai:dataReleaseMaintenancePlan",
         "SC3 v2 is the version submitted to NeurIPS 2026 Datasets & "
         "Benchmarks for review. The reproducible curation pipeline "
         "(scripts/01..81) is shipped alongside the dataset and regenerates "
         "every artifact bit-exactly from BigSolDB v2.1. Future versions "
         "(v3+) are planned to track upstream BigSolDB releases; each version "
         "will retain the full DECISIONS.md log with date stamps. Anonymous "
         "review repository for double-blind purposes only -- by the "
         "camera-ready deadline, the dataset will be re-released publicly "
         "under CC-BY-4.0 on a stable archival platform (Hugging Face "
         "Datasets or Zenodo, TBD), with the current anonymous URL "
         "deprecated. The maintainers commit to keeping the public release "
         "available for at least five years post-publication."),
    ])

    dataset = OrderedDict([
        ("@context", CONTEXT),
        ("@type", "sc:Dataset"),
        ("conformsTo", [
            "http://mlcommons.org/croissant/1.0",
            "http://mlcommons.org/croissant/RAI/1.0",
        ]),
        ("name", "SC3"),
        ("description", description),
        ("alternateName", [
            "SC^3",
            "Solubility Curation and Consistency Corpus",
        ]),
        ("keywords", [
            "solubility",
            "log S",
            "molecular property prediction",
            "chemistry",
            "drug discovery",
            "BigSolDB",
            "benchmark",
            "out-of-distribution",
            "aleatoric uncertainty",
            "thermodynamic consistency",
        ]),
        ("license", "https://creativecommons.org/licenses/by/4.0/"),
        ("url", DATASET_HOMEPAGE),
        ("version", "2.0"),
        ("datePublished", "2026-05-07"),
        ("citeAs", cite_as),
        ("creator", OrderedDict([
            ("@type", "sc:Person"),
            ("name", "Anonymous (NeurIPS 2026 D&B double-blind submission)"),
        ])),
        ("publisher", OrderedDict([
            ("@type", "sc:Organization"),
            ("name", "Anonymous"),
        ])),
        ("inLanguage", "en"),
        ("isLiveDataset", False),
        ("distribution", distribution),
        ("recordSet", record_set),
    ])

    for key, value in rai.items():
        dataset[key] = value

    return dataset


def assert_consistent(dataset: OrderedDict) -> None:
    """Light shape checks before writing -- catches obvious template errors."""
    assert dataset["@type"] == "sc:Dataset"
    assert "http://mlcommons.org/croissant/1.0" in dataset["conformsTo"]
    assert "http://mlcommons.org/croissant/RAI/1.0" in dataset["conformsTo"]

    file_ids = {fo["@id"] for fo in dataset["distribution"]}
    assert len(file_ids) == len(dataset["distribution"]), "duplicate FileObject @id"

    rs_field_ids: set[str] = set()
    for rs in dataset["recordSet"]:
        assert rs["@type"] == "cr:RecordSet"
        for f in rs["field"]:
            assert f["@type"] == "cr:Field"
            assert f["@id"] not in rs_field_ids, f"duplicate field @id: {f['@id']}"
            rs_field_ids.add(f["@id"])
            src = f["source"]
            assert src["fileObject"]["@id"] in file_ids, (
                f"field {f['@id']} references unknown FileObject "
                f"{src['fileObject']['@id']}"
            )

    rai_required = {
        "rai:dataCollection", "rai:dataCollectionType",
        "rai:dataCollectionMissingData", "rai:dataCollectionRawData",
        "rai:dataCollectionTimeframe", "rai:dataImputationProtocol",
        "rai:dataPreprocessingProtocol", "rai:dataManipulationProtocol",
        "rai:dataAnnotationProtocol", "rai:dataAnnotationPlatform",
        "rai:dataAnnotationAnalysis", "rai:annotationsPerItem",
        "rai:annotatorDemographics", "rai:machineAnnotationTools",
        "rai:dataBiases", "rai:dataUseCases", "rai:dataLimitations",
        "rai:dataSocialImpact", "rai:personalSensitiveInformation",
        "rai:dataReleaseMaintenancePlan",
    }
    missing = rai_required - dataset.keys()
    assert not missing, f"missing RAI fields: {sorted(missing)}"


def report_row_counts() -> None:
    print("File row counts (header excluded):")
    for s in FILE_SPECS:
        path = SC3_ROOT / s["rel_path"]
        print(f"  {s['rel_path']:<30}  {_row_count(path):>7,} rows")


def main() -> None:
    print(f"Generating Croissant metadata at {OUTPUT_PATH}")
    print(f"  ANON_REPO_URL    = {ANON_REPO_URL}")
    print(f"  DATASET_HOMEPAGE = {DATASET_HOMEPAGE}")
    print()
    report_row_counts()

    dataset = build_dataset()
    assert_consistent(dataset)

    OUTPUT_PATH.write_text(json.dumps(dataset, indent=2) + "\n")
    print(f"\nWrote {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
