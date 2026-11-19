# R2 QC and handoff - 2026-11-19

**Summary review complete for all 35 retained participants. Two corrected candidates are ready for a release decision; the official v2 FIFs have not been replaced.**

## Review scope and decisions

- Considered all 42 oddball participants. Existing release: 35 retained, 3 a-priori exclusions and 4 target-retention exclusions.
- Visually reviewed retained-epoch PTP maps, Pz condition averages, mean spectra and representative traces of the five highest-amplitude channels for every retained subject. Reviewed nine amplitude/ERP galleries and nine spectrum/trace galleries.
- This is visual summary QC, not exhaustive interactive review of all raw channels. Residual frontal/posterior activity remains; no claim of artifact-free EEG is made.
- Sub-10/OI1h and sub-13/AFF1h showed persistent isolated broadband amplitude elevation. These were marked manually for **candidate-only** reruns; decisions are now in `configs/config.yaml`.
- Other observed residual activity (including sub-07, sub-16, sub-30, sub-31 and sub-37) is recorded in CSV notes without adding an unagreed exclusion threshold.
- Repository `results/preproc_log.csv` now records review scope/date/reviewer and findings. Its scientific counts, exclusion reasons and cohort match the original archive. The archive CSV and all released FIFs remain unchanged.

## Corrected real-data candidates

| Subject / manual channel | Released epochs | Candidate epochs | Old median channel PTP | Candidate median PTP |
|---|---:|---:|---:|---:|
| sub-10 / OI1h | 269 | 591 | 134.53 uV | 38.61 uV |
| sub-13 / AFF1h | 510 | 648 | 106.39 uV | 17.63 uV |

Both candidates were generated from the task-specific original BrainVision data with actual ICA and ICLabel. They passed checks of shape, sampling, finite data, baseline, average reference, PTP, counts and source event class/correctness/RT/block metadata. Both before/after QC panels were inspected. This is a targeted correction, not a complete 42-participant reprocessing run.

Candidate files are in `data/derivatives/preproc/_candidate/`, with log/provenance in that directory and `results/r2_candidate/`. Reviewer evidence: `results/v2_audit/candidate_comparison.json`.

**Do not mix these files silently with v2.** If the team elects to replace both subjects while retaining the other 33, the proposed cohort would still have 35 subjects but 21,060 epochs (16,116 standard, 2,521 target, 2,423 distractor). These are candidate accounting values, not the official release or report results. R3 must rerun affected ERP/F2/P300 outputs after any approved release change; ML/DL must use the same version.

## R3-specific evidence

- `results/v2_audit/montage_sources.json` records the exact OpenNeuro `.bvef` source URLs and SHA-256 for sub-12/sub-24. Saved positions match to < 1e-8 m with MNE default head-size normalization. The other 33 retained subjects use CapTrak according to the release summaries.
- `figures/preprocessing/v2/review/sub-13_task-oddball_matched_raw_clean_psd.{png,pdf}` compares the same first 50 retained event windows before preprocessing and after released preprocessing, using 1-Hz Welch bins. It demonstrates attenuation of line/high-frequency power and persistent in-band AFF1h noise, not removal of every artifact.
- `sub-13_task-oddball_excluded_ics_0.{png,pdf}` shows saved excluded ICA components; excluded indices are recorded in `raw_psd_example.json`. Numerical ICLabel probabilities were not included in the published release and are not invented from the topomaps.
- `sub-10_candidate_comparison.png` and `sub-13_candidate_comparison.png` show the reviewed corrections.

## Methods, accounting and reproducibility

- Report-ready v2 Methods: [r2_methods_v2.md](r2_methods_v2.md), also inserted into `report_template.md` section 3.1.
- Stage-wise and all-42-subject counts: [r2_subject_epoch_table.md](r2_subject_epoch_table.md). The official v2 figures are 35 subjects, 20,600 epochs.
- The tracked v2 implementation is commit `6fccfc2502f3ee64a919de3ac97e86f52a5e3361` (PR #4). This is a verified Git reference, **not proof of the exact generating-job revision**, which the ZIP did not record. Meeting notes explicitly preserve this provenance limitation.
- New modular code: `scripts/r2_preprocessing.py`, `r2_validation.py`, `r2_qc.py`, `r2_checks.py`; CLI: `scripts/02_r2.py`; regression tests: `tests/test_r2.py`.
- New reader preserves raw bytes, decodes Latin-1 sidecars in a temporary view and aligns channel metadata by names to the BrainVision order. Sub-13 has FFT8h/FCC6h sidecar ordering differences; no positional relabeling or raw edits are performed.
- Core code fingerprints use normalized Python syntax so Git newline/format normalization does not invalidate an otherwise identical implementation.

## Verification

The final modular notebook passed all 35 FIF / 42 source-event checks in a real Jupyter kernel. Ruff check and format checks passed. All 11 regression tests passed with network access disabled; synthetic integration does not require the real dataset. Actual `.bvef` source matching is documented separately in the real-data audit. Evidence: `results/v2_audit/test_suite.json`, `notebook_execution.json`, `software_versions.json`.

## Commands

```powershell
# Existing published release: audit only
python scripts/02_r2.py

# Explicit candidate generation; only requested oddball files
python scripts/02_r2.py --process --download -s 10 13

# Automated regression and lint checks
python -m unittest discover -s tests -v
python -m ruff check scripts/r2_*.py scripts/02_r2.py tests/test_r2.py
```

The notebook remains the default review interface and currently contains 24 lines of execution code. It reads the same scripts instead of maintaining a second implementation. Raw sub-10/sub-13 was fetched for this review and is retained unchanged; no other tasks were downloaded.

## Remaining external actions

- Decide whether to keep the disclosed v2 limitations or promote a versioned replacement using the two candidates. No v3 is published by this work.
- The GitHub Issue text is prepared in [data_v2_ready_issue.md](data_v2_ready_issue.md); posting requires a connected GitHub account. No Issue is claimed to have been posted.
- No percentage contribution or team-owner assignment is inferred by this handoff.

Local review package (not an official release): `data/r2_candidate_review_20261119.zip`; SHA-256 `23eaca7486e623ddb7511ca4a356f273272c803d236d20378d0fe9c8974136f8`. Data/figures remain outside Git.
