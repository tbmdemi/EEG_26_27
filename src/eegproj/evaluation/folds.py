"""Fixed cross-subject fold assignment shared by every model (protocol §4).

The protocol requires one subject-level GroupKFold split, saved to `results/cv_cross_subject_folds.csv`,
so that classical ML (R4) and deep learning (R5) are evaluated on exactly the same test subjects.
The split itself comes from the shared `iter_cross_folds` (R4); this module only freezes it on disk
and re-reads it on later runs.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .splits import format_bids_subject, iter_cross_folds, normalize_subject_id

FOLDS_FILENAME = "cv_cross_subject_folds.csv"
WITHIN_SPLITS = ("equal_segments", "s10_blocks")


def within_cv_metadata(metadata, config):
    """Return a copy of chronological epoch metadata whose `block` column defines the within-subject folds.

    `config['evaluation']['within']['split']`:
      * 'equal_segments' (default): `n_splits` contiguous segments with (almost) equal numbers of trials, in
        recording order — identical to the fallback of the shared `iter_within_folds`.
      * 's10_blocks': keep the blocks between the `S 10` breaks from preprocessing.
    The original block is kept as `s10_block`. Pass the result to `iter_within_folds`.
    """
    within = config["evaluation"]["within"]
    split = within.get("split", "s10_blocks")
    if split not in WITHIN_SPLITS:
        raise ValueError(f"Unknown evaluation.within.split '{split}'. Expected one of {WITHIN_SPLITS}.")
    meta = metadata.reset_index(drop=True).copy()
    if "onset_s" in meta.columns and not meta["onset_s"].is_monotonic_increasing:
        raise ValueError("Trials must be in chronological order to form contiguous folds.")
    if "block" in meta.columns:
        meta["s10_block"] = meta["block"]
    if split == "equal_segments":
        n_splits = int(within["n_splits"])
        segments = np.empty(len(meta), dtype=int)
        for k, idx in enumerate(np.array_split(np.arange(len(meta)), n_splits)):
            segments[idx] = k
        meta["block"] = segments
    return meta


def folds_csv_path(root) -> Path:
    """Return `results/cv_cross_subject_folds.csv` under the repository root."""
    return Path(root) / "results" / FOLDS_FILENAME


def cross_subject_folds(root, config, subjects, allow_fewer_splits: bool = False, use_file: bool = True):
    """Return the fixed cross-subject split as a list of `(fold, train_subjects, test_subjects)`.

    If `results/cv_cross_subject_folds.csv` exists it is used as-is, after checking that it covers exactly
    `subjects`; otherwise the split is produced by `iter_cross_folds` and saved with columns `subject` ('sub-XX')
    and `fold` (0-based test fold). `use_file=False` (quick checks on a subject subset) neither reads nor writes
    the file. Subject IDs in the output are two-digit strings.
    """
    subjects = sorted(normalize_subject_id(s) for s in subjects)
    path = folds_csv_path(root)
    if use_file and path.exists():
        table = pd.read_csv(path, dtype={"subject": str})
        table["subject"] = table["subject"].map(normalize_subject_id)
        if sorted(table["subject"]) != subjects:
            missing = sorted(set(subjects) - set(table["subject"]))
            extra = sorted(set(table["subject"]) - set(subjects))
            raise ValueError(
                f"{path} does not match the requested subjects (missing {missing}, extra {extra}). "
                "Delete it only if the whole team agrees to re-split; all models must use the same folds."
            )
        assignment = dict(zip(table["subject"], table["fold"].astype(int)))
    else:
        assignment = {}
        for fold, _, test_subjects in iter_cross_folds(subjects, config=config, allow_fewer_splits=allow_fewer_splits):
            assignment.update({s: fold for s in test_subjects})
        if use_file:
            path.parent.mkdir(parents=True, exist_ok=True)
            out = pd.DataFrame(
                {"subject": [format_bids_subject(s) for s in subjects], "fold": [assignment[s] for s in subjects]}
            )
            out.to_csv(path, index=False)

    folds = []
    for fold in sorted(set(assignment.values())):
        test = [s for s in subjects if assignment[s] == fold]
        train = [s for s in subjects if assignment[s] != fold]
        if set(train) & set(test):
            raise AssertionError(f"Subject leakage in cross-subject fold {fold}.")
        folds.append((fold, train, test))
    return folds
