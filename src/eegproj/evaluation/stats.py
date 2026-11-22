"""Statistical comparison (RQ2: Wilcoxon signed-rank + Holm) and Figure F4 (protocol §8 & §10)."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
from matplotlib.lines import Line2D

from ..config import load_config
from .metrics import validate_metrics_df

STATS_COLUMNS = [
    "task",
    "setting",
    "window",
    "metric",
    "classical_model",
    "dl_model",
    "n_subjects",
    "classical_median",
    "classical_iqr",
    "dl_median",
    "dl_iqr",
    "median_diff_dl_minus_classical",
    "w_stat",
    "p_raw",
    "p_holm",
    "rank_biserial",
    "significant_holm",
]

_METRIC_LABELS = {
    "bal_acc": "Balanced Accuracy (proportion)",
    "auc": "ROC-AUC (proportion)",
    "f1_target": "Target F1-Score (proportion)",
}
_TASK_ORDER = ("tgt_std", "dis_std", "tgt_dis")
_SETTING_ORDER = ("within", "cross")
_WINDOW_ORDER = ("full", "early")


def _ordered_values(values, preferred_order: tuple[str, ...]) -> list[str]:
    """Return unique string values ordered by `preferred_order`, with any extras sorted afterwards."""
    present = {str(v) for v in values}
    ordered = [item for item in preferred_order if item in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def _select_best_model(family_full_df: pd.DataFrame, selection_metric: str) -> str:
    """Select the top model by median `selection_metric` on `'full'` window (mean as tie-breaker)."""
    summary = (
        family_full_df.groupby("model", as_index=False)[selection_metric]
        .agg(median="median", mean="mean")
        .dropna(subset=["median", "mean"])
    )
    if summary.empty:
        raise ValueError(f"No finite '{selection_metric}' values available on 'full' window for model selection.")
    summary = summary.sort_values(by=["median", "mean", "model"], ascending=[False, False, True])
    return str(summary.iloc[0]["model"])


def _paired_wilcoxon(dl_vals: np.ndarray, cls_vals: np.ndarray) -> tuple[float, float, float]:
    """Run two-sided paired Wilcoxon signed-rank test (`dl - classical`) returning `(W, p_raw, RBC)`."""
    diffs = dl_vals - cls_vals
    if np.allclose(diffs, 0.0):
        return 0.0, 1.0, 0.0
    w_res = pg.wilcoxon(dl_vals, cls_vals, alternative="two-sided")
    w_col = "W_val" if "W_val" in w_res.columns else "W-val"
    p_col = "p_val" if "p_val" in w_res.columns else "p-val"
    return float(w_res[w_col].iloc[0]), float(w_res[p_col].iloc[0]), float(w_res["RBC"].iloc[0])


def load_all_metrics(
    root: Path | str,
    config: dict | None = None,
    task_name: str | None = "tgt_std",
) -> pd.DataFrame:
    """Load and validate protocol §7 CSV files from `results/metrics/` (scores in `[0, 1]`).

    Parameters
    ----------
    root : Path or str
        Repository root directory.
    config : dict, optional
        Parsed `configs/config.yaml`. Loaded from `root` if omitted.
    task_name : str or None, default 'tgt_std'
        Optional task filter (`'tgt_std'`, `'dis_std'`, `'tgt_dis'`). Pass `None` to keep all tasks.

    Returns
    -------
    pd.DataFrame
        Concatenated validated metrics rows (dimensionless proportions in `[0, 1]`).
    """
    root_path = Path(root)
    cfg = load_config(root_path) if config is None else config
    metrics_dir = root_path / cfg["paths"]["metrics"]
    if not metrics_dir.exists():
        raise FileNotFoundError(f"Metrics directory not found: {metrics_dir}.")

    frames = []
    for path in sorted(metrics_dir.glob("*.csv")):
        df = pd.read_csv(path)
        validate_metrics_df(df)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No valid metric CSV files found in {metrics_dir}.")

    all_df = pd.concat(frames, ignore_index=True)
    if task_name is not None:
        all_df = all_df.loc[all_df["task"] == task_name].copy()
        if all_df.empty:
            raise ValueError(f"No metric rows found for task '{task_name}' in {metrics_dir}.")
    return all_df


def aggregate_subject_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Average fold-level metrics per `(family, model, setting, window, task, subject)`.

    Ensures each subject contributes exactly one independent observation (`N = 35` usable subjects)
    prior to paired statistical testing (protocol §8). All metrics are proportions in `[0, 1]`.
    """
    if df.empty:
        raise ValueError("Cannot aggregate an empty metrics DataFrame.")
    group_cols = ["family", "model", "setting", "window", "task", "subject"]
    missing_group = [col for col in group_cols if col not in df.columns]
    if missing_group:
        raise ValueError(f"Metrics DataFrame is missing required grouping columns: {missing_group}.")
    metric_cols = [col for col in ("bal_acc", "auc", "f1_target") if col in df.columns]
    if not metric_cols:
        raise ValueError("Metrics DataFrame must contain at least one metric column ('bal_acc', 'auc', 'f1_target').")
    return df.groupby(group_cols, as_index=False)[metric_cols].mean()


def compute_rq2_wilcoxon_holm(sub_df: pd.DataFrame, metric: str = "bal_acc") -> pd.DataFrame:
    """Compare classical-best vs DL-best per `(task, setting)` via paired Wilcoxon + Holm (protocol §8).

    For each `(task, setting)`, selects `classical-best` and `dl-best` by **median `bal_acc` on the
    `'full'` window** (with mean `bal_acc` as tie-breaker), then evaluates that fixed selected pair
    across `'full'` and `'early'` using two-sided `pingouin.wilcoxon(dl_vals, cls_vals)` and
    Holm step-down correction (`pingouin.multicomp(..., method='holm')`).

    Parameters
    ----------
    sub_df : pd.DataFrame
        Subject-aggregated metrics from `aggregate_subject_scores` (1 row per subject per condition).
    metric : str, default 'bal_acc'
        Metric compared in the paired Wilcoxon test (`'bal_acc'`, `'auc'`, or `'f1_target'`).

    Returns
    -------
    pd.DataFrame
        Summary table with columns `STATS_COLUMNS` (all score units are dimensionless proportions in `[0, 1]`).
    """
    if sub_df.empty:
        raise ValueError("Cannot compute RQ2 statistics on an empty DataFrame.")
    if metric not in _METRIC_LABELS:
        raise ValueError(f"Unsupported metric '{metric}'. Expected one of {sorted(_METRIC_LABELS)}.")
    if metric not in sub_df.columns:
        raise ValueError(f"Metric column '{metric}' not found in sub_df.")

    required_cols = ["family", "model", "setting", "window", "task", "subject"]
    missing = [col for col in required_cols if col not in sub_df.columns]
    if missing:
        raise ValueError(f"sub_df is missing required columns: {missing}.")
    if sub_df.duplicated(subset=required_cols).any():
        raise ValueError(
            "sub_df has duplicate (family, model, setting, window, task, subject) rows; "
            "call aggregate_subject_scores(df) first so each subject provides 1 independent observation."
        )

    selection_metric = "bal_acc" if "bal_acc" in sub_df.columns else metric

    rows = []
    for task in _ordered_values(sub_df["task"], _TASK_ORDER):
        task_df = sub_df.loc[sub_df["task"] == task]
        for setting in _ordered_values(task_df["setting"], _SETTING_ORDER):
            ts_df = task_df.loc[task_df["setting"] == setting]
            full_df = ts_df.loc[ts_df["window"] == "full"].dropna(subset=[selection_metric])
            cls_full = full_df.loc[full_df["family"] == "classical"]
            dl_full = full_df.loc[full_df["family"] == "dl"]
            if cls_full.empty or dl_full.empty:
                continue

            best_cls_model = _select_best_model(cls_full, selection_metric)
            best_dl_model = _select_best_model(dl_full, selection_metric)

            for window in _ordered_values(ts_df["window"], _WINDOW_ORDER):
                win_df = ts_df.loc[ts_df["window"] == window]
                cls_sub = win_df.loc[(win_df["family"] == "classical") & (win_df["model"] == best_cls_model)].set_index(
                    "subject"
                )[metric]
                dl_sub = win_df.loc[(win_df["family"] == "dl") & (win_df["model"] == best_dl_model)].set_index(
                    "subject"
                )[metric]

                paired = pd.concat({"classical": cls_sub, "dl": dl_sub}, axis=1).dropna()
                if len(paired) < 2:
                    continue

                cls_vals = paired["classical"].to_numpy(dtype=float)
                dl_vals = paired["dl"].to_numpy(dtype=float)
                diffs = dl_vals - cls_vals
                w_stat, p_raw, rbc = _paired_wilcoxon(dl_vals, cls_vals)

                rows.append(
                    {
                        "task": task,
                        "setting": setting,
                        "window": window,
                        "metric": metric,
                        "classical_model": best_cls_model,
                        "dl_model": best_dl_model,
                        "n_subjects": len(paired),
                        "classical_median": float(np.median(cls_vals)),
                        "classical_iqr": float(np.percentile(cls_vals, 75) - np.percentile(cls_vals, 25)),
                        "dl_median": float(np.median(dl_vals)),
                        "dl_iqr": float(np.percentile(dl_vals, 75) - np.percentile(dl_vals, 25)),
                        "median_diff_dl_minus_classical": float(np.median(diffs)),
                        "w_stat": w_stat,
                        "p_raw": p_raw,
                        "rank_biserial": rbc,
                    }
                )

    if not rows:
        raise ValueError(
            "Both 'classical' and 'dl' family results (including the 'full' window) are required for RQ2 comparison."
        )

    stats_df = pd.DataFrame(rows)
    reject, p_holm = pg.multicomp(stats_df["p_raw"].to_numpy(dtype=float), method="holm")
    stats_df["p_holm"] = np.asarray(p_holm, dtype=float)
    stats_df["significant_holm"] = np.asarray(reject, dtype=bool)
    return stats_df[STATS_COLUMNS]


def plot_f4_paired(
    sub_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    output_stem: Path | str,
    metric: str = "bal_acc",
    dpi: int = 300,
) -> tuple[Path, Path]:
    """Generate Figure F4 paired subject plot (Classical-best vs DL-best) as PNG (300 dpi) and PDF.

    Parameters
    ----------
    sub_df : pd.DataFrame
        Subject-aggregated metrics from `aggregate_subject_scores`.
    stats_df : pd.DataFrame
        Summary statistics table returned by `compute_rq2_wilcoxon_holm`.
    output_stem : Path or str
        Output stem without extension (e.g. `figures/F4_paired_classical_vs_dl`).
    metric : str, default 'bal_acc'
        Metric plotted on the y-axis (`'bal_acc'`, `'auc'`, or `'f1_target'`).
    dpi : int, default 300
        Resolution in dots per inch for the PNG image.

    Returns
    -------
    tuple[Path, Path]
        Saved `(png_path, pdf_path)`.
    """
    if stats_df.empty:
        raise ValueError("Cannot plot Figure F4 from an empty stats_df.")
    if metric not in _METRIC_LABELS:
        raise ValueError(f"Unsupported metric '{metric}'. Expected one of {sorted(_METRIC_LABELS)}.")

    stem = Path(output_stem)
    if stem.suffix.lower() in {".png", ".pdf"}:
        stem = stem.with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = Path(f"{stem}.png")
    pdf_path = Path(f"{stem}.pdf")

    ylabel = _METRIC_LABELS[metric]
    n_panels = len(stats_df)
    n_cols = min(2, n_panels)  # lưới 2x2 để chữ đọc được khi in ở bề rộng trang
    n_rows = math.ceil(n_panels / n_cols)
    multi_task = stats_df["task"].nunique() > 1

    paired_panels: list[tuple[pd.Series, pd.DataFrame]] = []
    y_values: list[float] = [0.5]
    for _, row in stats_df.reset_index(drop=True).iterrows():
        sub_group = sub_df.loc[
            (sub_df["task"] == row["task"])
            & (sub_df["setting"] == row["setting"])
            & (sub_df["window"] == row["window"])
        ]
        cls_s = sub_group.loc[
            (sub_group["family"] == "classical") & (sub_group["model"] == row["classical_model"])
        ].set_index("subject")[metric]
        dl_s = sub_group.loc[(sub_group["family"] == "dl") & (sub_group["model"] == row["dl_model"])].set_index(
            "subject"
        )[metric]
        paired = pd.concat({"classical": cls_s, "dl": dl_s}, axis=1).dropna()
        paired_panels.append((row, paired))
        if not paired.empty:
            y_values.extend(paired["classical"].tolist())
            y_values.extend(paired["dl"].tolist())

    y_min = max(0.0, min(0.35, float(np.floor(min(y_values) * 20.0) / 20.0 - 0.03)))
    y_max = max(1.02, float(max(y_values) + 0.03))

    rc_params = {
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 11,
    }
    with plt.rc_context(rc_params):
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(4.3 * n_cols, 4.8 * n_rows),
            sharey=True,
            squeeze=False,
        )
        axes_flat = axes.ravel()

        for idx, (row, paired) in enumerate(paired_panels):
            ax = axes_flat[idx]
            for _, vals in paired.iterrows():
                ax.plot(
                    [0, 1],
                    [vals["classical"], vals["dl"]],
                    color="#7f7f7f",
                    alpha=0.45,
                    linewidth=1.0,
                    zorder=1,
                )

            ax.scatter(
                np.zeros(len(paired)),
                paired["classical"],
                color="#1f77b4",
                s=30,
                alpha=0.85,
                edgecolors="white",
                linewidths=0.4,
                zorder=2,
            )
            ax.scatter(
                np.ones(len(paired)),
                paired["dl"],
                color="#d62728",
                s=30,
                alpha=0.85,
                edgecolors="white",
                linewidths=0.4,
                zorder=2,
            )
            ax.plot(
                [0, 1],
                [row["classical_median"], row["dl_median"]],
                color="black",
                linewidth=2.4,
                marker="D",
                markersize=6.5,
                zorder=3,
            )
            ax.axhline(0.5, color="#333333", linestyle="--", linewidth=1.0, alpha=0.7, zorder=0)
            ax.grid(axis="y", alpha=0.25, linestyle=":")
            ax.set_xticks([0, 1])
            ax.set_xticklabels([f"Classical\n({row['classical_model']})", f"DL\n({row['dl_model']})"])
            ax.set_xlim(-0.35, 1.35)
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel("Model Family")

            sig_suffix = "*" if bool(row["significant_holm"]) else ""
            header = (
                f"{row['task']} | {row['setting']} | {row['window']} (N={int(row['n_subjects'])})"
                if multi_task
                else f"{row['setting']} | {row['window']} (N={int(row['n_subjects'])})"
            )
            ax.set_title(
                f"{header}\n"
                rf"$\Delta$med={row['median_diff_dl_minus_classical']:+.3f}, "
                f"r={row['rank_biserial']:+.2f}, "
                rf"p_Holm={row['p_holm']:.3g}{sig_suffix}"
            )
            if idx % n_cols == 0:
                ax.set_ylabel(ylabel)

        for extra_ax in axes_flat[n_panels:]:
            extra_ax.set_visible(False)

        legend_handles = [
            Line2D([0], [0], color="#7f7f7f", alpha=0.65, linewidth=1.2, label="Subject pair"),
            Line2D([0], [0], color="black", linewidth=2.4, marker="D", markersize=6.5, label="Group median"),
            Line2D([0], [0], color="#333333", linestyle="--", linewidth=1.0, label="Chance (0.5)"),
        ]
        if not multi_task:
            task_label = str(stats_df["task"].iloc[0])
            fig.suptitle(f"Figure F4 ({task_label}): Classical-best vs DL-best paired subject comparison", y=0.99)
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=3,
            frameon=True,
            bbox_to_anchor=(0.5, 0.01),
        )
        top_rect = 0.92 if not multi_task else 0.98
        fig.tight_layout(rect=(0.0, 0.08, 1.0, top_rect))
        fig.savefig(png_path, dpi=dpi)
        fig.savefig(pdf_path)
        plt.close(fig)

    return png_path, pdf_path
