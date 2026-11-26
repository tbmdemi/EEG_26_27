"""Gom metric phân loại và tạo hình F3."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from .metrics import VALID_TASKS, validate_metrics_df
from .splits import format_bids_subject
from .stats import aggregate_subject_scores

F3_TASK = "tgt_std"
F3_METRICS = ("bal_acc", "auc", "f1_target")
F3_SUMMARY_COLUMNS = [
    "model",
    "family",
    "setting",
    "window",
    "task",
    "metric",
    "n_subjects",
    "mean",
    "sd",
    "median",
    "q1",
    "q3",
    "iqr",
    "minimum",
    "maximum",
]
_SETTING_ORDER = ("within", "cross")
_WINDOW_ORDER = ("full", "early")


def aggregate_f3_subject_scores(
    metrics: pd.DataFrame,
    expected_subjects: list[str],
    expected_models: dict[str, str],
    within_splits: int,
    task: str = F3_TASK,
) -> pd.DataFrame:
    """Kiểm tra metric theo protocol và rút gọn còn một điểm mỗi subject, điều kiện."""
    if task not in VALID_TASKS:
        raise ValueError(f"Unsupported task {task!r}. Expected one of {sorted(VALID_TASKS)}.")
    if not expected_subjects or len(set(expected_subjects)) != len(expected_subjects):
        raise ValueError("Expected subject IDs must be non-empty and unique.")
    if not expected_models or within_splits < 2:
        raise ValueError("Expected models and at least two within-subject folds are required.")

    validate_metrics_df(metrics)
    selected = metrics.loc[metrics["task"].eq(task)].copy()
    if selected.empty:
        raise ValueError(f"No metric rows found for task {task!r}.")
    score_values = selected[list(F3_METRICS)].to_numpy(dtype=float)
    if not np.isfinite(score_values).all() or ((score_values < 0) | (score_values > 1)).any():
        raise ValueError("All classification metrics must be finite proportions in [0, 1].")

    expected_subjects = {format_bids_subject(subject) for subject in expected_subjects}
    if set(selected["model"]) != set(expected_models):
        raise ValueError(
            f"Metric models {sorted(set(selected['model']))} do not match configured models {sorted(expected_models)}."
        )
    if not selected["family"].eq(selected["model"].map(expected_models)).all():
        raise ValueError("Metric model families do not match the configured model families.")

    row_keys = ["subject", "fold", "model", "setting", "window", "task"]
    if selected.duplicated(row_keys).any():
        raise ValueError(f"Metric rows contain duplicate keys: {row_keys}.")

    cross_folds: dict[str, int] = {}
    for model in expected_models:
        model_metrics = selected.loc[selected["model"].eq(model)]
        for setting in _SETTING_ORDER:
            for window in _WINDOW_ORDER:
                condition = model_metrics.loc[model_metrics["setting"].eq(setting) & model_metrics["window"].eq(window)]
                subjects = set(condition["subject"])
                if subjects != expected_subjects:
                    raise ValueError(f"{model}/{setting}/{window}: subject cohort differs from the expected cohort.")

                for subject, rows in condition.groupby("subject", sort=False):
                    folds = set(rows["fold"].astype(int))
                    if setting == "within":
                        expected_folds = set(range(within_splits))
                        if len(rows) != within_splits or folds != expected_folds:
                            raise ValueError(f"{model}/{setting}/{window}/{subject}: expected each within fold once.")
                    else:
                        if len(rows) != 1 or len(folds) != 1 or not folds <= set(range(within_splits)):
                            raise ValueError(f"{model}/{setting}/{window}/{subject}: expected one cross-test score.")
                        # Dùng cùng ánh xạ subject sang fold cho mọi model và cửa sổ.
                        fold = next(iter(folds))
                        if subject in cross_folds and cross_folds[subject] != fold:
                            raise ValueError(f"{subject}: cross-subject fold differs between model conditions.")
                        cross_folds[subject] = fold

    # Subject là đơn vị thống kê: lấy trung bình fold within, giữ một điểm test cross.
    subject_scores = aggregate_subject_scores(selected)
    return subject_scores.sort_values(["model", "setting", "window", "subject"], kind="stable").reset_index(drop=True)


def summarise_f3_scores(subject_scores: pd.DataFrame) -> pd.DataFrame:
    """Tính thống kê mô tả trên điểm số độc lập theo subject."""
    required = {"model", "family", "setting", "window", "task", "subject", *F3_METRICS}
    missing = required - set(subject_scores.columns)
    if missing:
        raise ValueError(f"Subject scores are missing columns: {sorted(missing)}.")
    if subject_scores.empty:
        raise ValueError("Cannot summarise an empty subject-score table.")

    rows = []
    group_columns = ["model", "family", "setting", "window", "task"]
    for keys, group in subject_scores.groupby(group_columns, sort=False):
        for metric in F3_METRICS:
            values = group[metric].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"{metric} contains non-finite subject scores.")
            q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75], method="linear")
            model, family, setting, window, task = keys
            rows.append(
                {
                    "model": model,
                    "family": family,
                    "setting": setting,
                    "window": window,
                    "task": task,
                    "metric": metric,
                    "n_subjects": group["subject"].nunique(),
                    "mean": values.mean(),
                    "sd": values.std(ddof=1),
                    "median": median,
                    "q1": q1,
                    "q3": q3,
                    "iqr": q3 - q1,
                    "minimum": values.min(),
                    "maximum": values.max(),
                }
            )
    return pd.DataFrame(rows, columns=F3_SUMMARY_COLUMNS)


def plot_f3_balanced_accuracy(
    subject_scores: pd.DataFrame,
    expected_models: dict[str, str],
    window_seconds: dict[str, list[float]],
    output_stem: Path | str,
    dpi: int = 300,
) -> tuple[Path, Path]:
    """Lưu F3 dạng boxplot balanced accuracy và điểm từng subject."""
    task_scores = subject_scores.loc[subject_scores["task"].eq(F3_TASK)]
    if task_scores.empty or dpi <= 0:
        raise ValueError("Figure F3 requires target-standard scores and a positive DPI.")
    if set(task_scores["model"]) != set(expected_models):
        raise ValueError("Figure F3 model set does not match the configured models.")
    for column in ("family", "setting", "window", "subject", "bal_acc"):
        if column not in task_scores:
            raise ValueError(f"Figure F3 is missing required column {column!r}.")
    if not np.isfinite(task_scores["bal_acc"].to_numpy(dtype=float)).all():
        raise ValueError("Figure F3 scores must be finite.")

    stem = Path(output_stem)
    if stem.suffix.lower() in {".png", ".pdf"}:
        stem = stem.with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
    png_path, pdf_path = Path(f"{stem}.png"), Path(f"{stem}.pdf")

    family_colors = {"classical": "#5B9BD5", "dl": "#F4B183"}
    model_order = list(expected_models)
    with plt.rc_context(
        {
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    ):
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True, squeeze=False)
        for row_index, setting in enumerate(_SETTING_ORDER):
            for column_index, window in enumerate(_WINDOW_ORDER):
                ax = axes[row_index, column_index]
                panel_scores = task_scores.loc[task_scores["setting"].eq(setting) & task_scores["window"].eq(window)]
                distributions = []
                scores_by_model: dict[str, np.ndarray] = {}
                for model in model_order:
                    model_scores = panel_scores.loc[panel_scores["model"].eq(model)].sort_values("subject")
                    values = model_scores["bal_acc"].to_numpy(dtype=float)
                    if values.size == 0:
                        raise ValueError(f"Missing F3 scores for {model}/{setting}/{window}.")
                    scores_by_model[model] = values
                    distributions.append(values)

                boxes = ax.boxplot(
                    distributions,
                    positions=np.arange(len(model_order)),
                    widths=0.56,
                    patch_artist=True,
                    showfliers=False,
                    medianprops={"color": "#E87500", "linewidth": 1.8},
                    whiskerprops={"linewidth": 1.2},
                    capprops={"linewidth": 1.2},
                )
                for box, model in zip(boxes["boxes"], model_order):
                    box.set_facecolor(family_colors[expected_models[model]])
                    box.set_alpha(0.7)
                    box.set_edgecolor("#222222")
                    values = scores_by_model[model]
                    jitter = np.linspace(-0.09, 0.09, len(values)) if len(values) > 1 else np.array([0.0])
                    ax.scatter(
                        np.full(len(values), model_order.index(model)) + jitter,
                        values,
                        color="#34495E",
                        alpha=0.55,
                        edgecolors="white",
                        linewidths=0.25,
                        s=18,
                        zorder=3,
                    )
                start, stop = window_seconds[window]
                ax.set(
                    title=f"{setting.title()}, {window} ({start * 1000:g}-{stop * 1000:g} ms)",
                    xticks=np.arange(len(model_order)),
                    xticklabels=model_order,
                    ylim=(0, 1.02),
                )
                ax.tick_params(axis="x", labelrotation=20)
                ax.axhline(0.5, color="#666666", linestyle="--", linewidth=1.0, zorder=0)
                ax.grid(axis="y", alpha=0.25)
                ax.text(
                    0.98,
                    0.04,
                    f"N = {panel_scores['subject'].nunique()} subjects",
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                )
                if column_index == 0:
                    ax.set_ylabel("Balanced accuracy (proportion)")

        fig.legend(
            handles=[
                Patch(facecolor=family_colors[family], edgecolor="#222222", label=family.title())
                for family in ("classical", "dl")
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=2,
            frameon=False,
        )
        fig.suptitle("Figure F3. Target versus standard classification", y=1.08, fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
    return png_path, pdf_path
