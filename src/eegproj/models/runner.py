"""End-to-end experiment runner for classical ML models (protocol §4–§7)."""

import random
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from ..evaluation.folds import cross_subject_folds, within_cv_metadata
from ..evaluation.metrics import (
    VALID_SETTINGS,
    compute_binary_metrics,
    make_metric_row,
    metrics_csv_path,
    save_metrics_csv,
    validate_metrics_df,
)
from ..evaluation.splits import (
    iter_within_folds,
    load_task_epochs,
    load_valid_subjects,
)
from .classical import CLASSICAL_MODELS, build_classical_model


def _evaluate_within(
    root: Path,
    config: dict,
    model_name: str,
    window_name: str,
    task_name: str,
    valid_subjects: list[str],
    seed: int,
) -> list[dict]:
    """Run leakage-free within-subject blocked CV across all valid subjects (epochs in µV, time in s)."""
    rows = []
    for sub in valid_subjects:
        epochs, y = load_task_epochs(root, config, sub, task_name, window_name)
        epochs_cv = epochs.copy()
        if epochs_cv.metadata is not None:
            epochs_cv.metadata = within_cv_metadata(epochs_cv.metadata, config)
        X_uv = np.ascontiguousarray(epochs_cv.get_data(picks="eeg") * 1e6)
        times = epochs_cv.times
        for fold_idx, train_idx, test_idx in iter_within_folds(epochs_cv, y, config=config):
            pipe = build_classical_model(model_name, config, times)
            pipe.fit(X_uv[train_idx], y[train_idx])
            y_score = pipe.predict_proba(X_uv[test_idx])[:, 1]
            y_pred = pipe.predict(X_uv[test_idx])
            metrics = compute_binary_metrics(y[test_idx], y_score, y_pred)
            rows.append(
                make_metric_row(
                    subject=sub,
                    fold=fold_idx,
                    model=model_name,
                    family="classical",
                    setting="within",
                    window=window_name,
                    task=task_name,
                    metrics=metrics,
                    n_train=len(train_idx),
                    n_test=len(test_idx),
                    seed=seed,
                )
            )
    return rows


def _align_eeg_channels(X_uv: np.ndarray, ch_names: list[str], ref_ch_names: list[str], subject: str) -> np.ndarray:
    """Reorder `X_uv` channels by name to match `ref_ch_names` (handles sub-13/sub-42 channel order)."""
    if ch_names == ref_ch_names:
        return np.ascontiguousarray(X_uv)
    if len(ch_names) != len(ref_ch_names) or set(ch_names) != set(ref_ch_names):
        missing = sorted(set(ref_ch_names) - set(ch_names))
        extra = sorted(set(ch_names) - set(ref_ch_names))
        raise ValueError(f"Subject {subject} channel set mismatch (missing={missing}, extra={extra}).")
    order = [ch_names.index(ch) for ch in ref_ch_names]
    return np.ascontiguousarray(X_uv[:, order, :])


def _evaluate_cross(
    root: Path,
    config: dict,
    model_name: str,
    window_name: str,
    task_name: str,
    valid_subjects: list[str],
    allow_fewer_splits: bool,
    seed: int,
    use_folds_file: bool = True,
) -> list[dict]:
    """Run leakage-free cross-subject GroupKFold CV, evaluating per test subject (epochs in µV, time in s)."""
    subject_data = {}
    ref_times = None
    ref_ch_names = None
    for sub in valid_subjects:
        epochs, y = load_task_epochs(root, config, sub, task_name, window_name)
        eeg_picks = mne.pick_types(epochs.info, eeg=True)
        ch_names = [epochs.ch_names[i] for i in eeg_picks]
        X_uv = epochs.get_data(picks="eeg") * 1e6
        if ref_times is None:
            ref_times = epochs.times.copy()
            ref_ch_names = ch_names
            X_uv = np.ascontiguousarray(X_uv)
        else:
            if not np.allclose(epochs.times, ref_times):
                raise ValueError(f"Subject {sub} time grid mismatch in cross-subject evaluation.")
            X_uv = _align_eeg_channels(X_uv, ch_names, ref_ch_names, sub)
        subject_data[sub] = (X_uv, y)

    rows = []
    folds = cross_subject_folds(
        root,
        config,
        valid_subjects,
        allow_fewer_splits=allow_fewer_splits,
        use_file=use_folds_file,
    )
    for fold_idx, train_subs, test_subs in folds:
        X_train = np.concatenate([subject_data[s][0] for s in train_subs], axis=0)
        y_train = np.concatenate([subject_data[s][1] for s in train_subs], axis=0)
        pipe = build_classical_model(model_name, config, ref_times)
        pipe.fit(X_train, y_train)
        n_train = len(y_train)
        for test_sub in test_subs:
            X_test, y_test = subject_data[test_sub]
            y_score = pipe.predict_proba(X_test)[:, 1]
            y_pred = pipe.predict(X_test)
            metrics = compute_binary_metrics(y_test, y_score, y_pred)
            rows.append(
                make_metric_row(
                    subject=test_sub,
                    fold=fold_idx,
                    model=model_name,
                    family="classical",
                    setting="cross",
                    window=window_name,
                    task=task_name,
                    metrics=metrics,
                    n_train=n_train,
                    n_test=len(y_test),
                    seed=seed,
                )
            )
        del X_train, y_train, pipe
    return rows


def run_classical_experiment(
    root: Path | str,
    config: dict,
    model_name: str,
    setting: str,
    window_name: str,
    task_name: str,
    subjects=None,
    allow_fewer_splits: bool = False,
    overwrite: bool = True,
) -> pd.DataFrame:
    """Run one `(model, setting, window, task)` combination and write its protocol §7 CSV.

    Epochs loaded in Volts (V) are converted to microvolts (µV) via `* 1e6` before fitting;
    time windows are defined in seconds (s) in `configs/config.yaml`.

    Parameters
    ----------
    root : Path or str
        Repository root directory.
    config : dict
        Parsed `configs/config.yaml`.
    model_name : str
        Classical model identifier (`'lda_tw'` or `'xdawn_rg'`).
    setting : str
        Evaluation scheme (`'within'` or `'cross'`).
    window_name : str
        Time window key in `config['classification']['windows']` (`'full'` [0.0, 0.8] s or `'early'` [0.0, 0.45] s).
    task_name : str
        Binary classification task key in `config['classification']['tasks']` (`'tgt_std'`, `'dis_std'`, or `'tgt_dis'`).
    subjects : sequence of str, optional
        Explicit subject list; defaults to Pipeline v2 usable subjects from `load_valid_subjects`.
    allow_fewer_splits : bool, default=False
        Allow cross-subject CV when testing on fewer than `n_splits` (5) subjects.
    overwrite : bool, default=True
        Recompute and overwrite the CSV even if it already exists.

    Returns
    -------
    pd.DataFrame
        Validated metrics DataFrame written to `results/metrics/<model>_<setting>_<window>_<task>.csv`.
    """
    valid_windows = set(config["classification"]["windows"])
    valid_tasks = set(config["classification"]["tasks"])
    if model_name not in CLASSICAL_MODELS:
        raise ValueError(f"Unsupported classical model '{model_name}'. Expected {CLASSICAL_MODELS}.")
    if setting not in VALID_SETTINGS:
        raise ValueError(f"Unsupported setting '{setting}'. Expected {sorted(VALID_SETTINGS)}.")
    if window_name not in valid_windows:
        raise ValueError(f"Unsupported window '{window_name}'. Expected {sorted(valid_windows)}.")
    if task_name not in valid_tasks:
        raise ValueError(f"Unsupported task '{task_name}'. Expected {sorted(valid_tasks)}.")

    root_path = Path(root).resolve()
    csv_path = metrics_csv_path(root_path, config, model_name, setting, window_name, task_name)
    if csv_path.exists() and not overwrite:
        return validate_metrics_df(pd.read_csv(csv_path))

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    valid_subjects = load_valid_subjects(root_path, config, subjects=subjects)

    if setting == "within":
        rows = _evaluate_within(root_path, config, model_name, window_name, task_name, valid_subjects, seed)
    else:
        rows = _evaluate_cross(
            root_path,
            config,
            model_name,
            window_name,
            task_name,
            valid_subjects,
            allow_fewer_splits,
            seed,
            use_folds_file=(subjects is None),
        )

    save_metrics_csv(rows, csv_path)
    df = pd.DataFrame(rows)
    mean_bal = df["bal_acc"].mean()
    mean_auc = df["auc"].mean()
    try:
        display_path = csv_path.resolve().relative_to(root_path)
    except ValueError:
        display_path = csv_path
    print(
        f"SAVED {display_path} | rows={len(df)} | bal_acc={mean_bal:.4f} | auc={mean_auc:.4f}",
        flush=True,
    )
    return df


def run_classical_grid(
    root: Path | str,
    config: dict,
    models=None,
    settings=None,
    windows=None,
    tasks=None,
    subjects=None,
    allow_fewer_splits: bool = False,
    overwrite: bool = True,
) -> list[pd.DataFrame]:
    """Run a grid of classical ML experiments across models, settings, windows, and tasks.

    Parameters
    ----------
    root : Path or str
        Repository root directory.
    config : dict
        Parsed `configs/config.yaml`.
    models : sequence of str, optional
        Model names (defaults to `CLASSICAL_MODELS`).
    settings : sequence of str, optional
        CV settings (defaults to `('within', 'cross')`).
    windows : sequence of str, optional
        Time windows in seconds (s) (defaults to keys of `config['classification']['windows']`).
    tasks : sequence of str, optional
        Binary tasks (defaults to keys of `config['classification']['tasks']`).
    subjects : sequence of str, optional
        Explicit subject subset. Defaults to all usable Pipeline v2 subjects.
    allow_fewer_splits : bool, default=False
        Allow cross-subject CV when testing on fewer than `n_splits` (5) subjects.
    overwrite : bool, default=True
        Overwrite existing CSVs if True.

    Returns
    -------
    list[pd.DataFrame]
        Validated metrics DataFrames for all executed grid combinations.
    """
    models = tuple(models or CLASSICAL_MODELS)
    settings = tuple(settings or ("within", "cross"))
    windows = tuple(windows or config["classification"]["windows"])
    tasks = tuple(tasks or config["classification"]["tasks"])

    results = []
    for model_name in models:
        for setting in settings:
            for window_name in windows:
                for task_name in tasks:
                    df = run_classical_experiment(
                        root=root,
                        config=config,
                        model_name=model_name,
                        setting=setting,
                        window_name=window_name,
                        task_name=task_name,
                        subjects=subjects,
                        allow_fewer_splits=allow_fewer_splits,
                        overwrite=overwrite,
                    )
                    results.append(df)
    return results
