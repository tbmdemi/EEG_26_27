"""Evaluation metrics and protocol-compliant CSV writer (protocol §5 & §7).

Shared by R4 (Classical ML) and R5 (Deep Learning).
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from .splits import format_bids_subject

REQUIRED_COLUMNS = [
    "subject",
    "fold",
    "model",
    "family",
    "setting",
    "window",
    "task",
    "bal_acc",
    "auc",
    "f1_target",
    "n_train",
    "n_test",
    "seed",
]

VALID_FAMILIES = {"classical", "dl"}
VALID_SETTINGS = {"within", "cross"}
VALID_WINDOWS = {"full", "early"}
VALID_TASKS = {"tgt_std", "dis_std", "tgt_dis"}


def compute_binary_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: np.ndarray | None = None,
    threshold: float | None = None,
) -> dict[str, float]:
    """Compute balanced accuracy, ROC-AUC, and target-class F1 (unitless scores in [0, 1]).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth binary labels (`1` = positive/target, `0` = negative/standard).
    y_score : array-like of shape (n_samples,)
        Continuous positive-class scores (predicted probabilities in `[0, 1]` or decision values).
    y_pred : array-like of shape (n_samples,), optional
        Predicted binary labels `{0, 1}`. If omitted, inferred from `y_score` using `threshold`.
    threshold : float, optional
        Decision threshold when `y_pred` is None. Defaults to `0.5` if `y_score` is within `[0, 1]`,
        otherwise `0.0`.

    Returns
    -------
    dict[str, float]
        Dictionary with keys `'bal_acc'`, `'auc'`, and `'f1_target'`.
    """
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_score_arr = np.asarray(y_score, dtype=float)
    if y_true_arr.shape != y_score_arr.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true_arr.shape} vs y_score {y_score_arr.shape}.")
    if not np.isfinite(y_score_arr).all():
        raise ValueError("Non-finite values found in y_score.")

    if y_pred is None:
        if threshold is None:
            in_prob_range = (np.min(y_score_arr) >= -1e-6) and (np.max(y_score_arr) <= 1.0 + 1e-6)
            threshold = 0.5 if in_prob_range else 0.0
        y_pred_arr = (y_score_arr >= threshold).astype(np.int64)
    else:
        y_pred_arr = np.asarray(y_pred, dtype=np.int64)

    if len(np.unique(y_true_arr)) < 2:
        warnings.warn(
            "Only one class present in y_true; AUC and balanced accuracy are undefined.", RuntimeWarning, stacklevel=2
        )
        bal_acc = float("nan")
        auc = float("nan")
    else:
        bal_acc = float(balanced_accuracy_score(y_true_arr, y_pred_arr))
        auc = float(roc_auc_score(y_true_arr, y_score_arr))

    f1_target = float(f1_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0))
    return {"bal_acc": bal_acc, "auc": auc, "f1_target": f1_target}


def make_metric_row(
    *,
    subject: str,
    fold: int,
    model: str,
    family: str,
    setting: str,
    window: str,
    task: str,
    metrics: dict[str, float],
    n_train: int,
    n_test: int,
    seed: int,
) -> dict:
    """Build a single validated result dictionary matching protocol §7 schema."""
    row = {
        "subject": format_bids_subject(subject),
        "fold": int(fold),
        "model": str(model),
        "family": str(family),
        "setting": str(setting),
        "window": str(window),
        "task": str(task),
        "bal_acc": float(metrics["bal_acc"]),
        "auc": float(metrics["auc"]),
        "f1_target": float(metrics["f1_target"]),
        "n_train": int(n_train),
        "n_test": int(n_test),
        "seed": int(seed),
    }
    validate_metrics_df(pd.DataFrame([row]))
    return row


def metrics_csv_path(root, config, model: str, setting: str, window: str, task: str) -> Path:
    """Return the canonical CSV path `results/metrics/<model>_<setting>_<window>_<task>.csv`."""
    return Path(root) / config["paths"]["metrics"] / f"{model}_{setting}_{window}_{task}.csv"


def validate_metrics_df(df: pd.DataFrame) -> pd.DataFrame:
    """Validate that a metrics DataFrame strictly conforms to protocol §7."""
    if list(df.columns) != REQUIRED_COLUMNS:
        raise ValueError(f"CSV columns {list(df.columns)} do not match required schema {REQUIRED_COLUMNS}.")
    if df.empty:
        raise ValueError("Metrics DataFrame cannot be empty.")
    if not df["subject"].astype(str).str.startswith("sub-").all():
        raise ValueError("All entries in 'subject' column must use BIDS format 'sub-XX'.")
    if not set(df["family"]).issubset(VALID_FAMILIES):
        raise ValueError(f"Invalid family values: {set(df['family']) - VALID_FAMILIES}.")
    if not set(df["setting"]).issubset(VALID_SETTINGS):
        raise ValueError(f"Invalid setting values: {set(df['setting']) - VALID_SETTINGS}.")
    if not set(df["window"]).issubset(VALID_WINDOWS):
        raise ValueError(f"Invalid window values: {set(df['window']) - VALID_WINDOWS}.")
    if not set(df["task"]).issubset(VALID_TASKS):
        raise ValueError(f"Invalid task values: {set(df['task']) - VALID_TASKS}.")
    if (df["n_train"] <= 0).any() or (df["n_test"] <= 0).any():
        raise ValueError("'n_train' and 'n_test' must be strictly positive integers.")
    return df


def save_metrics_csv(rows, output_path: Path | str) -> Path:
    """Validate and write metric rows to `<model>_<setting>_<window>_<task>.csv`."""
    df = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    df = df[REQUIRED_COLUMNS]
    validate_metrics_df(df)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path
