# SC³ v2 — Data curation decisions log

Every policy choice, its rationale, the numbers supporting it, and where in the
code/reports it is implemented.  Intended to be the reference the paper's data
section cites.

---

## D-01 · Solute SMILES canonicalization policy = **"Option D"**

**Policy.** Plain RDKit canonical SMILES with `isomericSmiles=True`.
No tautomer enumeration. No chirality stripping. No geometric-isomer stripping.
Every stereoisomer present in BigSolDB remains a distinct solute.

**Alternatives considered.**

| Option | Definition | Unique solutes | Merges |
|--------|------------|---------------:|-------:|
| A — v1 exact | tautomer-enumerate + strip `@/@@` + strip `/\\` | 1 506 | 19 |
| B — no tautomer | strip `@/@@` + strip `/\\` | 1 508 | 17 |
| C — strip chirality only | strip `@/@@`, keep `/\\`, no tautomer | 1 510 | 15 |
| **D — keep all stereo** | plain canonical, keep `@/@@` and `/\\`, no tautomer | **1 525** | **0** |

**Why D, not C:**

An empirical audit (`scripts/11_merge_audit.py`) was run over every Option-C merge
group.  For each group we found all `(solvent, temperature)` cells where ≥2
distinct raw-SMILES members have measurements and computed `|Δ logS|` across
them.

If the textbook assumption "enantiomers have identical solubility in achiral
solvents" held, these deltas should be at or below the aleatoric floor
(~0.06 log S).  In practice, **10 of 15 Option-C merge groups show median
|Δ logS| between 0.29 and 1.31**, i.e., 5–20× the aleatoric floor:

| Merged compound | Overlap pairs | Median \|Δ logS\| | Max |
|---|---:|---:|---:|
| Brassinolide (2 stereoisomers) | 36 | **1.31** | 2.01 |
| Camphor-like diol | 3 | 0.70 | 0.85 |
| D / L-psicose | 36 | 0.61 | 0.89 |
| D / L-malic acid | 21 | 0.52 | 0.99 |
| D / L-tryptophan | 84 | 0.43 | 0.97 |
| D / L-norbornene anhydride | 117 | 0.40 | 0.56 |
| L-leucine / mixed | 8 | 0.39 | 0.45 |
| D / L-tyrosine | 16 | 0.37 | 1.07 |
| N-acetyl-methionine D / L | 45 | 0.37 | 0.73 |
| Ibuprofen (S vs mixed) | 1 | 0.29 | 0.29 |

The four "silent merges" (isoleucine, ofloxacin, naproxen, tartaric acid —
no overlap cells, so no empirical check possible) are all enantiomer pairs
structurally analogous to the tested cases; by symmetry we expect similar
disagreement.

**Interpretation.** The 0.3–1.3 log S stereo-partner delta reflects some
mixture of (a) real crystal-form / polymorphism effects, (b) lab ×
chirality-label correlation in measurement protocol, and (c) silent
mislabelling of racemate as L (or vice versa) at the BigSolDB extraction
stage.  For a benchmark that aims to calibrate measurement reproducibility,
we cannot disambiguate these — and we should not hide them by averaging.

**Consequence for models.** Featurizers that are chirality-blind
(RDKit 2D descriptors, many Morgan variants) will predict the same value for
L- and D-enantiomers.  Under Option D, the benchmark correctly exposes this
featurization blindness as model error on the stereo-partner points.  Under
A/B/C it was averaged away inside the label.

**Artifacts.**
* `scripts/10_canonicalize.py` (Option D)
* `reports/10_canonicalization.json`
* `reports/11_merge_audit_empirical.csv`
* `reports/11_merge_audit_summary.csv`
* `data/interim/01_canonical.csv`

**Paper one-liner.** *"We canonicalize solute SMILES with plain RDKit canonical
form (`isomericSmiles=True`), preserving chirality and geometric isomerism.
An empirical audit of v1's alternative policies found 10 of 15 stereo-merge
groups disagreeing by 0.3–1.3 log S at matched (solvent, T) — 5–20× the
aleatoric floor — indicating that chirality stripping destroys real variance
we cannot recover."*

---

## D-02 · Solvent SMILES canonicalization policy

**Policy.** Plain RDKit canonical SMILES with `isomericSmiles=True`.  No
chirality issues arise for solvents in this dataset (Phase 0 confirmed no
solvent with stereo descriptors).

* 218 raw solvent NAMES → 214 raw SMILES → 212 canonical SMILES (1 parse
  failure: the `-` token used for polymer rows, which the polymer filter
  removes; 1 alias pair: diisobutyl methanol ≡ 2,6-dimethyl-4-heptanol).

---

## D-03 · Bad-DOI list

**Policy.** Start from v1's 9 DOIs, re-verify them, and extend with new
candidates identified by systematic quality checks.

**v1's 9 DOIs (438 rows total, all verified present):**
* `10.1016/j.fluid.2011.09.033` (126 rows)
* `10.1021/acs.jced.9b00728` (105 rows)
* `10.1021/acs.jced.4c00179` (91 rows)
* `10.1021/acs.jced.6b00009` (49 rows)
* `10.1016/j.fluid.2015.07.038` (24 rows)
* `10.1016/j.fluid.2013.09.018` (20 rows)
* `10.1016/j.molliq.2013.06.011` (10 rows)
* `10.1016/j.molliq.2022.119759` (9 rows)
* `10.1016/j.molliq.2020.113867` (4 rows)

**New candidate added: `10.1016/j.molliq.2017.02.075` (8 rows).**

Detected by back-computing `logS = log10(x/(1−x)·ρ(T)·1000/Mw)` and comparing
to BigSolDB's own reported logS.  The median absolute residual across all
109 278 testable rows is 1 × 10⁻⁵ log S (floating-point noise).  This DOI's
8 rows — β-alanine (`NCCC(=O)O`) in methanol at 288.15–323.15 K — show a
systematic residual of −0.35 to −0.36 log S (reported values ~2.3× too high
in molarity).

Whether this is a unit swap, a density error, or a formula inversion is
unclear; the bias is systematic enough to disqualify those 8 rows.  Added
to the bad-DOI list.

**Artifacts.** `scripts/02_deep_audit.py` (D3); `reports/02_deep_audit.json`.

---

## D-04 · "Pentoxifylline ≡ tolfenamic acid" case

**Policy.** No action taken beyond documenting the correction.

The v1 appendix (§A) treats this as a canonicalization bug where two
chemically distinct molecules collapse onto the same SMILES.  This is
incorrect.  All 261 rows share one raw SMILES
(`Cc1c(Cl)cccc1Nc1ccccc1C(=O)O` — tolfenamic acid structure), one CAS
(`13710-19-5` — tolfenamic acid CAS), and one PubChem CID (`610479` —
tolfenamic acid CID).  The 261 rows come from two DOIs; one of them
mislabelled its 144 rows as "Pentoxifylline" in the `Compound_Name` column
only.  Pentoxifylline itself has SMILES
`Cn1c(=O)c2c(ncn2C)n(CCCCC(C)=O)c1=O` and is not in the dataset.

Since the pipeline keys on SMILES, predictions are unaffected.  The v1
appendix paragraph should be corrected when we write paper v2.

**Artifacts.** `scripts/02_deep_audit.py` (D1); `reports/02_deep_audit.json`.

---

## D-05 · Copycat detection strategy

**Policy.** Two-stage:

* **Stage A — bit-exact matches.** Any two DOIs that report the same
  `(solute_smiles, solvent_smiles, round(T, 1), round(logS, 4))` tuple are
  declared copies deterministically — no threshold.
* **Stage B — near-exact matches (gray zone).** After Stage A merging,
  any remaining DOI-pair with pairwise MAE < 0.01 at shared temperatures
  (within the same merged group) is additionally merged.  This threshold is
  reported as the selected value from v1's plateau analysis
  (`θ = 0.01` is insensitive-zone — varying it from 0 to 0.01 does not shift
  summary statistics).

Phase-0 snapshot of Stage A on raw data: 88 keys match, 176 rows involved,
46 DOIs implicated.  Top offender pair:
`10.1016/j.jct.2024.107365` ↔ `10.1016/j.molliq.2025.127552` with 29 exact
matches.  (Full pair list in `reports/02_deep_audit.json`.)

Stage B will run after cleaning and canonicalization, since Stage A catches
only bit-identical rows which may survive or not survive the cleaning steps.

---

## D-06 · Intra-DOI near-duplicate handling

**Policy.** Within any single DOI, if two rows share
`(solute, solvent, round(T, 1))` they are duplicates.  Keep the one with the
lower row index (i.e., the first occurrence in BigSolDB) and drop the rest.

Phase-0 snapshot: exactly 16 duplicate rows from one DOI,
`10.1016/j.jct.2020.106137`, where each nominal temperature is reported at
`T = x.15 K` and `T = x.16 K` with logS matching at the 4ᵗʰ decimal — a data
entry rounding artifact.  No other DOI shows this.

---

## D-07 · Polymer / invalid-solvent filter

**Policy.** Drop any row where the solvent SMILES fails to parse with RDKit
or equals the literal `"-"`.  No hard-coded list of solvent names.

In raw BigSolDB v2.1 this filter removes **319 rows** — all with solvent
SMILES `"-"`.  The affected solvent names are PEG-400, PEG-200, PEG-300,
PEG-600, PEGDME 250, span 80 (same as v1's hard-coded list).  By testing
SMILES parseability we make the filter self-documenting.

---

## D-08 · Salt / mixture filter

**Policy.** Drop any row whose solute SMILES contains a `.` (indicating a
multi-component structure: salt, hydrate, co-crystal, ionic-liquid pair).

Phase-0 snapshot: 10 042 rows affected (193 unique multi-component forms).
Justification matches v1: counterion affects crystal lattice energy and
dissolution mechanism in ways that are not captured by molecular-graph
features alone, so averaging salt and neutral forms under one SMILES is
inconsistent.

Limitation acknowledged: this removes pharmaceutically relevant salt data
(HCl, Na, hydrates).  Future v3 could retain salts with an explicit
"salt / free base" flag, but v2 keeps the v1 policy.

---

## D-09 · MW and logS range filters

**Policy.** Drop rows with solute MW > 1 000 Da (29 rows in v1 cleaning;
recomputed in v2 on Option-D solutes) or logS ∉ [−15, 2] (102 rows in v1
cleaning; recomputed in v2).

Justification: MW > 1 000 Da rows are large polymers / excipients that have
inconsistent crystal-vs-amorphous solubility; logS outside this range is
either below typical detection limit (logS < −15 implies < 10⁻¹⁵ mol / L,
well below UV-HPLC limits) or above practical solute concentration
(logS > 2 implies > 10² mol / L, physically unreachable for most solids in
most solvents).  Filter bounds match v1 for comparability.

---

## D-10 · Phase 2 outcome (cleaning waterfall)

Applied `scripts/20_clean.py` to `data/interim/01_canonical.csv`:

```
Input (canonicalized raw)                                112 465
W1  remove 10 bad DOIs                                   112 019   (-446)
W2  remove polymer / invalid solvent                     111 705   (-314)
W3  remove salts / mixtures                              101 663   (-10 042)
W4  MW ≤ 1000 Da                                         101 634   (-29)
W5  recovered 2 617 / dropped 14 still-NaN               101 620   (-14 net)
W6  logS ∈ [-15, 2]                                      101 575   (-45)
W7  dedupe intra-DOI (T → 0.1 K rounding)                101 535   (-40)
```

Final dataset characteristics:

| | v2 (this) | v1 (for reference) |
|---|---:|---:|
| rows | **101 535** | 101 535 |
| unique solutes (canonical) | **1 327** | 1 311 |
| unique solvents | **206** | 204 |
| unique DOIs | **1 493** | 1 494 |
| multi-source (solute, solvent) pairs | **746** | 791 |
| T range (K) | 243.15 – 425.77 | same |
| logS range | −7.60 – 1.99 | same |

Key deltas vs v1, all traceable:

* +16 solutes from Option D (keeping stereo-distinct molecules distinct).
* +2 solvents recovered from the long-tail via CAS-based thermo aliases
  (ε-caprolactone, 2-methyl-cyclohexyl acetate).
* −1 DOI (new bad DOI `10.1016/j.molliq.2017.02.075`).
* −8 rows from the new bad DOI.
* −40 rows from intra-DOI rounding dedupe (W7 — new in v2).
* −45 multi-source pairs because Option D splits v1's merged stereoisomers.

The row count happens to match v1 to the exact value because the offsets
(+48 newly recovered long-tail rows, −48 from new bad DOI + intra-DOI dedup)
cancel to zero. Not by design.

**Artifacts.** `scripts/20_clean.py`, `reports/20_waterfall.json`,
`data/interim/02_cleaned.csv`.

---

## D-11 · Phase 3 outcome (source integrity)

Applied `scripts/30_source_integrity.py` to `data/interim/02_cleaned.csv`.

**Two-stage copycat detection**:

* **Stage A — bit-exact.** 86 `(solute, solvent, T, logS)` keys matched by
  ≥2 DOIs.  Rows affected: 172.  DOI unions: 21.
* **Stage B — gray zone.** DOI-pair weighted MAE < 0.01 at shared T.
  38 additional unions.

Combined: 1 493 DOIs → **1 434 independence groups**
(1 381 singletons, 48 pairs, 4 triples, 1 quadruple).

MAE distribution among different-group DOI pairs is clean:
minimum is **0.010 log S**, confirming the θ_B = 0.01 plateau empirically
(no pair with MAE < 0.01 remains in different groups).  P50 = 0.060,
P90 = 0.42, P95 = 0.75.

**Multi-source pool**:

|  | pairs | rows |
|---|---:|---:|
| total (solute, solvent) | 10 938 | 101 535 |
| ≥2 independent groups | **623** (5.7 %) | 10 720 (10.6 %) |
| ≥3 independent groups | 76 | — |
| ≥5 independent groups | 2 | — |

For comparison, v1 reported 588 multi-source pairs post-copycat-correction.
The 623 figure is slightly higher because Stage B uses a weighted MAE across *all*
pairs a DOI-pair shares, rather than per-pair independently.

**DOI reliability (Stage C)**:

* 324 of 1 493 DOIs have cross-group overlap and can be reliability-tested
  (the other 1 169 have no overlapping independent group — untested).
* Median deviation from consensus: **0.066 log S** (mean 0.19).
* Hall of Fame (≤ 0.2): **258** (80 % of tested).
* Hall of Shame (≥ 0.6): **22** (7 % of tested, 1 110 rows).

**Outliers worth flagging.**  The top-3 Hall of Shame DOIs have deviations
~4 log S — almost certainly unit errors in the source publication or its
BigSolDB transcription, uncatchable by Phase 0 because their internal
(x → logS) consistency is preserved:

* `10.1021/je4000718` (18 rows, mean dev 4.35)
* `10.1021/je5001654` (14 rows, 3.99)
* `10.1016/j.molliq.2016.11.036` (130 rows, 2.81)

Plus several lower-magnitude paired-identical-deviation DOI pairs that
are probably copycats Stage B missed because they don't share
temperatures (caught in Phase 4 via Apelblat interpolation).

**Handling policy.**  Keep these DOIs in the training pool (models should
see realistic data with outliers) but exclude Hall of Shame from tier
consensus construction in Phase 6.  They also get a `quality_flag` column
in the final cleaned CSV so downstream users can filter.

**Artifacts**:

* `scripts/30_source_integrity.py`
* `data/interim/03_doi_groups.csv`
* `reports/30_doi_pair_mae.csv`
* `reports/30_doi_reliability.csv`
* `reports/30_source_integrity.json`

---

## D-12 · Co-maintainer manual corrections

Source: `manual_corrections_log.txt` (manual audit log produced jointly by the dataset curator and the BigSolDB v2.1 co-maintainer,
March 2026).  The BigSolDB v2.1 co-maintainer manually audited
our Phase-3 Hall of Shame candidates and supplied specific corrections.

**Fixable DOIs (applied in `scripts/15_apply_manual_corrections.py`)**:

| DOI | Fix | Rows |
|---|---|---:|
| `10.1016/j.molliq.2020.113867` | paracetamol/water mole fractions replaced with the co-maintainer's values at 293.15 / 303.15 / 313.15 / 323.15 K | 4 |
| `10.1016/j.fluid.2013.09.018` | logS + 1.0 (×10 correction) | 20 |
| `10.1016/j.molliq.2013.06.011` | logS + 1.0 (×10 correction) | 10 |
| `10.1016/j.fluid.2015.07.038` | ethanol ↔ ethyl acetate labels swapped, logS / mol/L re-derived | 24 |

Total rows modified: 58.

**Confirmed outliers (added to bad-DOI list, dropped)**:

| DOI | Audit | Rows in raw |
|---|---|---:|
| `10.1021/je4000718` | flavonoid solubility ~25 000× off (curator → co-maintainer confirm) | 18 |
| `10.1016/s1004-9541(08)60201-3` | emodin/ethanol outlier (the co-maintainer) | 16 |
| `10.1016/j.molliq.2016.11.036` | Zhang et al. flavonoids, wrong (Phase 0 residual + the co-maintainer) | 130 |

**Confirmed-OK DOIs (kept as-is)**: `je900540d`, `je0603978`, `molliq.2020.115058`,
`s10953-016-0526-2` — not bad, appeared in the HoS only because compared against
an outlier peer.

**Updated bad-DOI list (v2 final)**, 9 entries:

1. `10.1016/j.fluid.2011.09.033`   (v1 + the co-maintainer)
2. `10.1021/acs.jced.9b00728`      (v1 + the co-maintainer)
3. `10.1021/acs.jced.4c00179`      (v1 + the co-maintainer)
4. `10.1021/acs.jced.6b00009`      (v1 + the co-maintainer)
5. `10.1016/j.molliq.2022.119759`  (v1 + the co-maintainer)
6. `10.1021/je4000718`             (NEW — curator + co-maintainer)
7. `10.1016/s1004-9541(08)60201-3` (NEW — the co-maintainer)
8. `10.1016/j.molliq.2016.11.036`  (NEW — curator + co-maintainer)
9. `10.1016/j.molliq.2017.02.075`  (NEW — Phase 0 residual)

(Coincidentally same count as v1, but different composition: 4 of v1's
originals were fixable; 4 new ones came from the audit.)

**Artifacts**: `scripts/15_apply_manual_corrections.py`,
`data/interim/01b_manually_corrected.csv`, `reports/15_manual_corrections.json`.

---

## D-13 · Phase 5 — Aleatoric limit methodology (updated)

**Policy**: compute |Δ logS| between independence groups by evaluating their
fitted Apelblat / van't Hoff curves at a uniform 1 K reference grid inside
the intersection of their fit ranges (≥ 5 K overlap required).  No
measured-T privileged points — every comparison is an interpolated-vs-
interpolated pair.  Single-point groups and bad-fit groups (R² < 0.80 or
RMSE > 0.30) are excluded from the aleatoric computation.

**Headline definition**: ε_A = mean over multi-source pairs of the pair-
level MAE, where pair MAE = mean across within-pair group-pair MAEs.
("Error in expectation" interpretation — per your request.)

**Primary (HoS-excluded, community-consensus-agreeing)**:
ε_A = **0.121 log S**, 95 % CI [0.100, 0.145], n = 512 pairs.
Supporting: median 0.044, P90 0.277, P95 0.421, RMSE 0.289, max 3.43.

**Inclusive (all 18 post-correction HoS groups included)**:
ε_A = 0.150 log S, 95 % CI [0.126, 0.179], n = 537 pairs.  P95 = 0.605
coincides exactly with Palmer & Mitchell's 0.6 log S, confirming that
range represents the tail of a heavy-tailed inter-lab distribution.

**Gamma fit** to the atom-level |Δ| distribution (primary): shape = 0.59,
scale = 0.18, KS = 0.094 — sub-exponential tail.

**Per-solvent (primary)**: tightest common solvent is DMF (mean 0.029,
10 pairs); water is noisiest among common solvents (0.197, 70 pairs).
DMSO only has 8 multi-source pairs in v2 (too few for a "tightest"
claim like v1 made).

**Artifacts**: `scripts/50_aleatoric.py`, `data/interim/05_pair_mae.csv`,
`data/interim/05_atom_deltas.csv`, `reports/50_aleatoric.json`,
`reports/50_per_solvent_aleatoric.csv`.

---

## D-14 · Phase 3 split into Stage A (bit-exact) + Phase 3B (interpolated Stage B' / C')

**Rationale.** Phase 3 v1 used shared-T exact matches across Stages A + B + C,
which is inconsistent with Phase 5's interpolated-only methodology (D-13).
Two concrete consequences: (1) gray-zone copycat detection misses DOI pairs
that don't share temperatures (e.g., the co-maintainer's "Lansoprazole, both suspicious"
and "Valine in ethanol" pairs); (2) DOI reliability ranking cannot test
groups that don't happen to overlap in T with a peer, leaving them in the
"untested" pool and unable to land in Hall of Shame even when they deviate
strongly.

**Fix.** Restructure the pipeline:

1. `scripts/30_source_integrity.py` — Stage A (bit-exact) only →
   preliminary DOI groups.  1 470 prelim groups, 21 Stage-A DOI unions.
2. `scripts/40_apelblat.py` — Apelblat / van't Hoff fits per preliminary
   group.  11 166 fits at median R² = 0.9993.
3. `scripts/45_source_integrity_interp.py` — Stage B' (interpolated MAE
   < 0.01 at 1 K grid on fit-range intersection ≥ 5 K) and Stage C'
   (interpolated reliability against consensus of other final groups).
   Produces 1 415 final independence groups (55 new Stage-B' unions on
   top of Stage A), tests 399 DOIs for reliability (up from 324 with
   shared-T), identifies 27 Hall of Shame (up from 18).
4. `scripts/40_apelblat.py` — re-fit per FINAL group (curves change only
   trivially for merged groups that agreed to < 0.01 log S).  11 063
   final fits at median R² = 0.9993.
5. `scripts/50_aleatoric.py` — aleatoric on final groups and final fits,
   HoS-excluded primary + inclusive secondary.

**Final Phase 5 numbers (fully interpolated pipeline, D-13 + D-14)**:

* Primary (HoS-excluded): **ε_A = 0.106 log S**, 95 % CI [0.093, 0.120],
  n = 481 multi-source pairs.  Supporting distribution: median 0.046,
  P90 0.258, P95 0.385, RMSE 0.182, max 1.22.
* Inclusive (all 27 HoS groups included): ε_A = 0.158 log S, 95 % CI
  [0.132, 0.186], n = 511 pairs, P95 0.627 (coincides with Palmer 0.6).
* Gamma fit to atom-level |Δ| (primary): shape = 0.68, scale = 0.15,
  KS = 0.09.

**Paper framing.** ε_A = 0.11 log S (mean of per-pair inter-lab MAE,
community-consensus-agreeing sources only) is the benchmark's noise floor.
Palmer & Mitchell's 0.6 log S is the P95 of the heavy-tailed distribution,
not the expected error.

**Artifacts**: `scripts/45_source_integrity_interp.py`,
`data/interim/03_doi_groups.csv` (final), `reports/30_doi_reliability.csv`
(interpolated), `reports/30_source_integrity.json`, `reports/50_aleatoric.json`.

---

## D-15 · Phase 6 — SC³ tier construction

**Policy (finalized)**:

* **Eligibility**.  Tier-eligible = 481 multi-source pairs from the Phase-5
  HoS-excluded primary pool.  Pairs with any HoS group are **not** tier-
  eligible (they stay in the training pool, though).
* **Thresholds** on pair-level MAE (from Phase-5 pure-interpolation
  methodology, D-13): Gold ≤ 0.1, Silver ≤ 0.2, Bronze ≤ 0.5 log S.
  Nested: Gold ⊂ Silver ⊂ Bronze.  Same boundaries as v1's Hard / Medium /
  Easy, but renamed because v1's "Hard" actually meant "tightest ground
  truth" = easiest for models, which reverses the usual intuition.
* **Consensus labels**.  For each pair, at each reference T measured by any
  contributing non-HoS group, `LogS_consensus = mean` of the Apelblat /
  van't Hoff fits (of contributing groups whose fit range covers T).
  No extrapolation beyond a group's [T_min, T_max].
* **Per-point σ**.  `σ = std` of contributing fits at T (sample std,
  ddof=1), floored at **0.012 log S** (≈ median Apelblat fit RMSE of 0.005
  ×  conservative factor; prevents degenerate zeros in Z-RMSE).  `σ = NaN`
  when only one group's fit covers T.
* **Benchmark rows are at measured temperatures only** — no synthetic grid
  points — so every row corresponds to a physical measurement that some
  non-HoS group actually performed.

**Output (final v2 tiers)**:

| Tier | Pairs | Rows | Solutes | Solvents | σ-coverage | Median σ |
|---|---:|---:|---:|---:|---:|---:|
| Gold   (≤ 0.1) | 335 | 4 507 | 129 | 26 | 76.5 % | 0.019 |
| Silver (≤ 0.2) | 400 | 5 475 | 141 | 27 | 76.9 % | 0.024 |
| Bronze (≤ 0.5) | 469 | 6 331 | 148 | 30 | 77.1 % | 0.031 |

Row contributor histogram across all tiers:
* n_contrib = 1  (σ = NaN): 23 % of rows — reference T at the edge of
  one group's fit range, only one fit covers.
* n_contrib = 2: 68 %  — proper σ, computed from 2 fit evaluations.
* n_contrib ≥ 3: 9 %   — multi-group consensus (strongest calibration).

**Deltas vs v1** (v1 Hard / Medium / Easy → v2 Gold / Silver / Bronze):

* Pairs: 217/290/390 → 335/400/469 (+54 / +38 / +20 %)
* Rows : 2 286/3 126/4 092 → 4 507/5 475/6 331 (+97 / +75 / +55 %)
* σ-cov: 72–77 % → 76.5–77.1 %

Gains trace to: (a) the co-maintainer's corrections rescuing 58 rows into the
eligible pool, (b) interpolated Stage B' catching more exact-copy clusters
so more pairs qualify, (c) no threshold mismatch (v1 used θ = 0.01 for
aleatoric and θ = 0.02 for tier construction, losing 165 pairs; v2 is
consistent at θ_B = 0.01 throughout).

**Paper one-liner.** *"SC³ provides three nested tiers (Gold ⊂ Silver ⊂
Bronze) of (solute, solvent) multi-source pairs, with per-temperature
consensus log S labels built from the mean of community-consensus-
agreeing independence groups' Apelblat fits, and per-point σ calibrated
from the fit-level disagreement (floored at 0.012 log S, the median
Apelblat RMSE).  Gold contains 335 pairs × 4 507 measurements at
inter-lab MAE ≤ 0.1 log S; Silver and Bronze relax the MAE bound to 0.2
and 0.5 respectively."*

**Artifacts**: `scripts/60_tiers.py`, `data/sc3/gold.csv`,
`data/sc3/silver.csv`, `data/sc3/bronze.csv`, `data/sc3/tier_pairs.csv`,
`data/sc3/tier_summary.json`.

---

## D-16 · Phase 7 — Train / Eval / OOD splits

**Policy**:

* **Anti-leakage**.  Every solute that appears in any tier (union across
  Gold ⊂ Silver ⊂ Bronze = **148 solutes**) is removed from the training
  pool under *all* solvents.  Yields a pool of **80 312 rows** (79.2 %
  of cleaned 101 429 rows).
* **ID vs OOD by solvent**.  Top-N solvents are chosen as the smallest N
  whose cumulative row share in the training pool is ≥ 0.85.  This gives
  N = **25** data-driven (coincidentally matches v1).  The remaining 161
  solvents form the solvent-OOD split.
* **Eval hold-out**.  For each ID solvent, 10 % of its (solute, solvent)
  pairs are held out (`ceil(n_pairs × 0.10)`, seed 42).  All temperature
  measurements of a held-out pair stay together.  Matches v1 (pair-level
  10 % per solvent).
* **Three generalization axes exposed**:
  * Eval  →  new (solute, solvent) pair in a familiar solvent.
  * OOD   →  any solute in an unseen solvent.
  * Tiers →  new solute (solute-disjoint from training) with calibrated σ.

**Realized split (seed 42)**:

| Split | Rows | Solutes | Solvents | Pairs | % pool |
|---|---:|---:|---:|---:|---:|
| Train  | 61 403 | 1 144 | 25  | 6 840 | 76.5 % |
| Eval   |  6 969 |   534 | 25  |   771 |  8.7 % |
| OOD    | 11 940 |   586 | 161 | 1 450 | 14.9 % |

Target was 75 % / 10 % / 15 %; actual ratio is 76.5 / 8.7 / 14.9 — close.

**Anti-leakage verification** (all must be 0): train/eval/ood ∩ tier
solutes and train/eval/ood ∩ gold/silver/bronze pairs — all ✓ PASS.

Expected overlaps (by construction): train ∩ eval share 520 solutes
(pair-level split); train ∩ OOD share 566 solutes (same solute in ID and
OOD solvents).

**Artifacts**: `scripts/70_splits.py`, `data/splits/{bench_train,
bench_eval, bench_ood}.csv`, `reports/70_splits.json`.

---

## D-17 · Phase 8 — Metric suite + motivating analysis

**Policy**: four metrics are reported for every model on every split, each
chosen to address a specific failure mode of the naive RMSE / R² pair on
multi-solvent solubility data.

### Motivating analyses (what a first-time reader sees first)

**§1 Multimodality**.  206 solvents, per-solvent logS means span **7.98
log units** (−6.93 for 1,1-dichloroethane to +1.05).  Dataset-level logS
distribution is a mixture over solvent-specific unimodal distributions.

**§2 Variance decomposition** (between-solvent / total).

| Split | Between-solvent var frac |
|---|---:|
| Cleaned | 11.7 % |
| Train   |  9.7 % |
| Eval    |  9.0 % |
| OOD     | 22.1 % *(thin-tail solvents, denser location shifts)* |
| Gold / Silver / Bronze | 10 / 9.3 / 9.6 % |

**§3 Dummy R² baseline** — solvent-mean-only predictor trained on Train,
evaluated on held-out splits.  This is the R² a model gets **for free**
just by identifying which solvent a point is in, before learning any
solute chemistry:

| Target split | Dummy R² | Dummy RMSE |
|---|---:|---:|
| Eval   | +0.053 | 1.118 |
| OOD    | −0.062 | 1.094 *(solvent means unavailable)* |
| Gold   | +0.008 | 1.168 |
| Silver | −0.003 | 1.136 |
| Bronze | +0.017 | 1.104 |

**§4 Count domination** (cleaned dataset).  Top-1 solvent = 9.5 % of
rows, top-5 = **37.5 %**, top-10 = 62.4 %, top-25 = 84.5 %.  Aggregate
RMSE is a count-weighted average dominated by five solvents.

**§5 MAPE diagnostic**.  5.6 % of rows have |logS| < 0.1 (where
MAPE = |ŷ−y|/|y| diverges); 28.5 % have |logS| < 0.5.  MAPE is
unusable on SC³.

**§6 Heavy-tail label distribution**.  mean |y−μ| / median |y−μ| = 1.19
on cleaned.  Model residuals on label-noise-limited data inherit a
heavier tail; RMSE is tail-sensitive, MedAE is not.

### Metric suite (Phase 8 definitions in `scripts/80_metrics.py`)

| Metric | Formula | Robustness property |
|---|---|---|
| **RMSE**    | `sqrt(mean((ŷ−y)²))` | standard; retained for comparability |
| **MAE**     | `mean(\|ŷ−y\|)` | less tail-sensitive than RMSE |
| **MedAE**   | `median(\|ŷ−y\|)` | **robust to heavy-tailed residuals** |
| **PS-RMSE** | `mean_s(sqrt(mean_{i∈s}((ŷ−y)²)))` | **equal weight per solvent**: strips count-weighting + between-solvent inflation simultaneously |
| **Z-RMSE**  | `sqrt(mean((ŷ−y)/σ)²)` over rows with finite σ | **error in units of the aleatoric floor**: Z = 1 means "matches measurement noise" |
| MAPE        | diagnostic only — diverges for |logS| near zero | unusable here |

### Computational domains per split

| Split  | rows | for RMSE / MAE / MedAE | PS-RMSE groups | for Z-RMSE | Z-RMSE coverage |
|---|---:|---:|---:|---:|---:|
| Train  | 61 403 | 61 403 | 25 solvents | 0 | 0 % (no σ in training pool) |
| Eval   |  6 969 |  6 969 | 25 | 0 | 0 % |
| OOD    | 11 940 | 11 940 | 161 | 0 | 0 % |
| Gold   |  4 507 |  4 507 | 26 | 3 449 | **76.5 %** |
| Silver |  5 475 |  5 475 | 27 | 4 212 | **76.9 %** |
| Bronze |  6 331 |  6 331 | 30 | 4 881 | **77.0 %** |

Papers quoting Z-RMSE on SC³ must state n (σ-defined subset), not the
full tier row count.

**Artifacts**: `scripts/80_metrics.py` (importable metric definitions),
`scripts/81_multimodality.py`, `reports/81_multimodality.json`,
`reports/81_per_solvent_stats.csv`, `reports/81_metric_domains.csv`.

---

## Phase summary — v2 data pipeline complete

| Phase | Role | Output |
|---|---|---|
| 0 | Raw audit | 4 new findings vs v1 (new bad DOI, pentoxifylline misdiagnosis, canon bugs, bit-exact copycats) |
| 1 | Canonicalization (Option D) | 1 525 raw → 1 525 canonical solutes |
| 1.5 | the co-maintainer corrections | 58 rows rescued, 3 DOIs added to bad list |
| 2 | Cleaning waterfall | **101 429** rows, 1 327 solutes, 206 solvents, 1 493 DOIs |
| 3A | Copycat Stage A (bit-exact) | 21 unions, 1 470 prelim groups |
| 4 | Apelblat/van't Hoff fits | 11 063 fits, median R² 0.9993 |
| 3B | Stage B' + C' (interpolated) | 1 415 final groups, **27 HoS DOIs** |
| 5 | Aleatoric limit | **ε_A = 0.11 log S** (primary, 95 % CI [0.09, 0.12]); 0.16 inclusive |
| 6 | Tiers Gold / Silver / Bronze | 335 / 400 / 469 pairs, 4 507 / 5 475 / 6 331 rows |
| 7 | Splits | Train 61 403 / Eval 6 969 / OOD 11 940 |
| 8 | Metric suite + motivation | PS-RMSE, Z-RMSE, MedAE; variance decomp, dummy R², heavy-tail diagnostics |


**Current direction.** Rename v1's `Hard / Medium / Easy` to names that
reflect *ground-truth quality*, not model difficulty, because v1's naming
reverses the intuitive ordering (their "Hard" tier has the *tightest* labels,
i.e. the *lowest* label noise, which is the *easiest* situation for a
model).

Candidate options: `Gold / Silver / Bronze`, `GT-High / GT-Med / GT-Low`,
or `Tier-A / Tier-B / Tier-C`.  Decision deferred to post tier-construction
(Phase 6).

---
