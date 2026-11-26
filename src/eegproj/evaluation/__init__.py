"""Shared evaluation framework (CV splits, fixed folds, metrics, and RQ2 statistics)."""

from .aggregate import aggregate_f3_subject_scores, plot_f3_balanced_accuracy, summarise_f3_scores
from .folds import (
    FOLDS_FILENAME,
    WITHIN_SPLITS,
    cross_subject_folds,
    folds_csv_path,
    within_cv_metadata,
)
from .metrics import (
    REQUIRED_COLUMNS,
    VALID_FAMILIES,
    VALID_SETTINGS,
    VALID_TASKS,
    VALID_WINDOWS,
    compute_binary_metrics,
    make_metric_row,
    metrics_csv_path,
    save_metrics_csv,
    validate_metrics_df,
)
from .splits import (
    epochs_path,
    format_bids_subject,
    iter_cross_folds,
    iter_cross_trial_splits,
    iter_within_folds,
    load_task_epochs,
    load_valid_subjects,
    normalize_subject_id,
)
from .stats import (
    STATS_COLUMNS,
    aggregate_subject_scores,
    compute_rq2_wilcoxon_holm,
    load_all_metrics,
    plot_f4_paired,
)

__all__ = [
    "FOLDS_FILENAME",
    "REQUIRED_COLUMNS",
    "STATS_COLUMNS",
    "VALID_FAMILIES",
    "VALID_SETTINGS",
    "VALID_TASKS",
    "VALID_WINDOWS",
    "WITHIN_SPLITS",
    "aggregate_f3_subject_scores",
    "aggregate_subject_scores",
    "compute_binary_metrics",
    "compute_rq2_wilcoxon_holm",
    "cross_subject_folds",
    "epochs_path",
    "folds_csv_path",
    "format_bids_subject",
    "iter_cross_folds",
    "iter_cross_trial_splits",
    "iter_within_folds",
    "load_all_metrics",
    "load_task_epochs",
    "load_valid_subjects",
    "make_metric_row",
    "metrics_csv_path",
    "normalize_subject_id",
    "plot_f3_balanced_accuracy",
    "plot_f4_paired",
    "save_metrics_csv",
    "summarise_f3_scores",
    "validate_metrics_df",
    "within_cv_metadata",
]
