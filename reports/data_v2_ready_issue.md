# Data v2 ready - validated release and R2 QC handoff

Release: `preproc_v2_20261118.zip`, 35 usable participants from 42 considered.
SHA-256: `c53f040786b6c23a9b5d135a31f0bef966088018434409a1ca44a2e944c7a6ef`.
The existing download entry point remains `python scripts/00_get_data.py`.

- All 35 released FIFs and behavioural summaries for all 42 subjects were checked.
- All 35 retained subjects have visual summary QC annotations in `results/preproc_log.csv`.
- Exact montage source evidence for sub-12/sub-24 and a matched raw/clean PSD plus excluded-IC example are available locally through the R2 handoff.
- Report-ready Methods and complete subject/epoch accounting are in `reports/r2_methods_v2.md` and `reports/r2_subject_epoch_table.md`.
- Tracked v2 source reference: 6fccfc2502f3ee64a919de3ac97e86f52a5e3361 (PR #4). The exact generating-job revision was not recorded in the ZIP; this reference must not be presented as stronger provenance than verified.

## QC decision needed before final results

Persistent broadband channel noise in sub-10/OI1h and sub-13/AFF1h was corrected in separate real-data candidates. Their counts are 591 and 648 epochs versus 269 and 510 in v2. Do not replace files silently or mix versions. If a replacement release is approved, R3 must rerun the affected summaries; all model families must use the same release.

See `reports/r2_qc_and_handoff.md` for review scope, exclusions, findings, comparison figures, reproducible commands and remaining limitations. This text is a prepared Issue body, not evidence that an Issue has been posted.
