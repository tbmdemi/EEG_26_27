"""Cross-validation splits and epoch loading utilities (protocol §2–§4).

Shared by R4 (Classical ML) and R5 (Deep Learning) to guarantee identical,
leakage-free evaluation across within-subject and cross-subject settings.
"""

import warnings
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def normalize_subject_id(subject) -> str:
    """Normalize a subject identifier (e.g. 'sub-01', '01', 1) to a two-digit string '01'."""
    text = str(subject).strip().removeprefix("sub-")
    return text.zfill(2) if text.isdigit() else text


def format_bids_subject(subject) -> str:
    """Format a subject identifier as BIDS 'sub-XX' for CSV output and logs."""
    return f"sub-{normalize_subject_id(subject)}"


def epochs_path(root, config, subject) -> Path:
    """Return the path to a subject's preprocessed epoch FIF file (`data/derivatives/preproc/`)."""
    sub = normalize_subject_id(subject)
    task = config["dataset"]["task"]
    return Path(root) / config["paths"]["derivatives"] / f"sub-{sub}_task-{task}_epo.fif"


def load_valid_subjects(root, config, subjects=None) -> list[str]:
    """Return configured subject IDs ('01', '02', ...) excluding any flagged in preproc_log.csv.

    Parameters
    ----------
    root : Path or str
        Repository root directory.
    config : dict
        Parsed `configs/config.yaml`.
    subjects : sequence of str, optional
        Explicit subset of subjects to filter. Defaults to `config['dataset']['subjects']`.

    Returns
    -------
    list[str]
        Valid two-digit subject codes in sorted order.
    """
    root = Path(root)
    requested = [normalize_subject_id(s) for s in (subjects or config["dataset"]["subjects"])]
    log_path = root / "results" / "preproc_log.csv"
    excluded = set()
    if log_path.exists():
        log_df = pd.read_csv(log_path)
        if "excluded_subject" in log_df.columns and "subject" in log_df.columns:
            mask = log_df["excluded_subject"].astype(str).str.lower().isin({"true", "1"})
            excluded = {normalize_subject_id(s) for s in log_df.loc[mask, "subject"]}
    valid = [s for s in requested if s not in excluded]
    if not valid:
        raise ValueError("No valid subjects remaining after filtering excluded_subject.")
    return valid


def load_task_epochs(root, config, subject, task_name: str, window_name: str) -> tuple[mne.Epochs, np.ndarray]:
    """Load preprocessed epochs for one subject, filtered to a binary task and cropped to a time window.

    Parameters
    ----------
    root : Path or str
        Repository root directory.
    config : dict
        Parsed `configs/config.yaml`.
    subject : str
        Subject ID (e.g. '01' or 'sub-01').
    task_name : str
        Task key in `config['classification']['tasks']` ('tgt_std', 'dis_std', or 'tgt_dis').
    window_name : str
        Window key in `config['classification']['windows']` ('full' [0.0, 0.8] s or 'early' [0.0, 0.45] s).

    Returns
    -------
    epochs : mne.Epochs
        Cropped MNE Epochs object (amplitude in Volts, time in seconds) preserving temporal order.
    y : np.ndarray
        Binary labels of shape `(n_trials,)` with `1` = positive class and `0` = negative class.
    """
    tasks_cfg = config["classification"]["tasks"]
    windows_cfg = config["classification"]["windows"]
    if task_name not in tasks_cfg:
        raise KeyError(f"Unknown task '{task_name}'. Expected one of {sorted(tasks_cfg)}.")
    if window_name not in windows_cfg:
        raise KeyError(f"Unknown window '{window_name}'. Expected one of {sorted(windows_cfg)}.")

    fif_path = epochs_path(root, config, subject)
    if not fif_path.exists():
        raise FileNotFoundError(f"Preprocessed FIF not found: {fif_path}")

    task_spec = tasks_cfg[task_name]
    pos_name, neg_name = task_spec["positive"], task_spec["negative"]
    tmin, tmax = (float(x) for x in windows_cfg[window_name])

    epochs = mne.read_epochs(fif_path, preload=True, verbose=False)
    sub_epochs = epochs[[neg_name, pos_name]].copy()
    sub_epochs.crop(tmin=tmin, tmax=tmax, include_tmax=True)

    if (
        sub_epochs.metadata is not None
        and "onset_s" in sub_epochs.metadata.columns
        and not sub_epochs.metadata["onset_s"].is_monotonic_increasing
    ):
        raise ValueError(f"Epochs for subject {subject} are not in monotonic temporal order.")

    pos_code = sub_epochs.event_id[pos_name]
    neg_code = sub_epochs.event_id[neg_name]
    codes = sub_epochs.events[:, 2]
    if not np.isin(codes, [neg_code, pos_code]).all():
        raise ValueError(f"Unexpected event codes in {fif_path} for task {task_name}.")

    y = (codes == pos_code).astype(np.int64)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError(f"Subject {subject} task {task_name} is missing one of the binary classes.")
    return sub_epochs, y


def iter_within_folds(
    epochs: mne.Epochs, y: np.ndarray | None = None, config: dict | None = None, n_splits: int | None = None
):
    """Yield contiguous time-blocked folds `(fold_idx, train_idx, test_idx)` for within-subject CV.

    Uses `epochs.metadata['block']` (defined by S 10 breaks in preprocessing) when it contains
    exactly `n_splits` contiguous blocks; falls back to splitting trials into `n_splits`
    contiguous time segments without shuffling.

    Parameters
    ----------
    epochs : mne.Epochs
        Subject epochs in chronological order.
    y : np.ndarray, optional
        Binary target vector of shape `(n_trials,)` used to verify both classes exist in each fold.
    config : dict, optional
        Parsed `configs/config.yaml`.
    n_splits : int, optional
        Number of contiguous folds (defaults to `config['evaluation']['within']['n_splits']` = 5).

    Yields
    ------
    fold_idx : int
        Fold index in `0 .. n_splits - 1`.
    train_idx : np.ndarray
        1D integer indices for the training trials.
    test_idx : np.ndarray
        1D integer indices for the test trials.
    """
    if n_splits is None:
        n_splits = int(config["evaluation"]["within"]["n_splits"]) if config else 5
    n_trials = len(epochs)
    if n_trials < n_splits:
        raise ValueError(f"Cannot split {n_trials} trials into {n_splits} folds.")

    if (
        epochs.metadata is not None
        and "onset_s" in epochs.metadata.columns
        and not epochs.metadata["onset_s"].is_monotonic_increasing
    ):
        raise ValueError("Trials must be ordered chronologically for blocked within-subject CV.")

    blocks = None
    if epochs.metadata is not None and "block" in epochs.metadata.columns:
        candidate = epochs.metadata["block"].to_numpy(dtype=int)
        if np.array_equal(np.unique(candidate), np.arange(n_splits)):
            if not np.all(np.diff(candidate) >= 0):
                raise ValueError("Metadata block assignments are not contiguous in time.")
            blocks = candidate

    if blocks is None:
        chunks = np.array_split(np.arange(n_trials), n_splits)
        blocks = np.empty(n_trials, dtype=int)
        for fold_id, idx in enumerate(chunks):
            blocks[idx] = fold_id

    for fold_idx in range(n_splits):
        test_idx = np.flatnonzero(blocks == fold_idx)
        train_idx = np.flatnonzero(blocks != fold_idx)
        if len(train_idx) == 0 or len(test_idx) == 0:
            raise ValueError(f"Empty train or test split in within-subject fold {fold_idx}.")
        if np.intersect1d(train_idx, test_idx).size > 0:
            raise AssertionError(f"Data leakage detected in within-subject fold {fold_idx}.")
        if y is not None and (len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2):
            raise ValueError(f"Fold {fold_idx} is missing a class in train or test.")
        yield fold_idx, train_idx, test_idx


def iter_cross_folds(
    subjects,
    config: dict | None = None,
    n_splits: int | None = None,
    allow_fewer_splits: bool = False,
):
    """Yield subject-level GroupKFold splits `(fold_idx, train_subjects, test_subjects)`.

    Guarantees no subject appears in both `train_subjects` and `test_subjects`.

    Parameters
    ----------
    subjects : sequence of str
        Subject identifiers.
    config : dict, optional
        Parsed `configs/config.yaml`.
    n_splits : int, optional
        Number of folds (defaults to `config['evaluation']['cross']['n_splits']` = 6).
    allow_fewer_splits : bool, default False
        If True and `len(unique_subjects) < n_splits` (e.g. 5-subject exploratory subset),
        reduces `n_splits` to `len(unique_subjects)` with a warning instead of raising an error.

    Yields
    ------
    fold_idx : int
        Fold index in `0 .. actual_splits - 1`.
    train_subjects : list[str]
        Normalized subject IDs in the training set.
    test_subjects : list[str]
        Normalized subject IDs in the test set.
    """
    if n_splits is None:
        n_splits = int(config["evaluation"]["cross"]["n_splits"]) if config else 6
    unique_subjects = sorted(dict.fromkeys(normalize_subject_id(s) for s in subjects))
    n_unique = len(unique_subjects)
    if n_unique < 2:
        raise ValueError(f"Cross-subject CV requires at least 2 subjects, got {n_unique}.")

    actual_splits = n_splits
    if n_unique < n_splits:
        if not allow_fewer_splits:
            raise ValueError(
                f"Cross-subject CV requires n_splits={n_splits} subjects, but only {n_unique} provided. "
                "Pass allow_fewer_splits=True when testing on the exploratory subset."
            )
        actual_splits = n_unique
        warnings.warn(
            f"Reducing cross-subject n_splits from {n_splits} to {actual_splits} for {n_unique}-subject subset.",
            RuntimeWarning,
            stacklevel=2,
        )

    sub_array = np.asarray(unique_subjects, dtype=object)
    gkf = GroupKFold(n_splits=actual_splits)
    for fold_idx, (train_pos, test_pos) in enumerate(gkf.split(sub_array, groups=sub_array)):
        train_subs = sub_array[train_pos].tolist()
        test_subs = sub_array[test_pos].tolist()
        if set(train_subs) & set(test_subs):
            raise AssertionError(f"Subject leakage detected in cross-subject fold {fold_idx}.")
        yield fold_idx, train_subs, test_subs


def iter_cross_trial_splits(
    subject_groups,
    config: dict | None = None,
    n_splits: int | None = None,
    allow_fewer_splits: bool = False,
):
    """Yield trial-level `(fold_idx, train_idx, test_idx)` splits grouped by subject.

    Consistent with `iter_cross_folds`: every trial from a given subject is assigned
    exclusively to either train or test in each fold.
    """
    groups = np.asarray([normalize_subject_id(s) for s in subject_groups], dtype=object)
    for fold_idx, train_subs, test_subs in iter_cross_folds(
        groups, config=config, n_splits=n_splits, allow_fewer_splits=allow_fewer_splits
    ):
        train_idx = np.flatnonzero(np.isin(groups, train_subs))
        test_idx = np.flatnonzero(np.isin(groups, test_subs))
        if np.intersect1d(train_idx, test_idx).size > 0:
            raise AssertionError(f"Trial leakage detected in cross-subject fold {fold_idx}.")
        yield fold_idx, train_idx, test_idx
