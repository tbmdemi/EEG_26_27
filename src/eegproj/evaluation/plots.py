"""Subject-level model comparison plots for F3 (protocol §10)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .aggregate import F3_MODELS, F3_SETTINGS, F3_WINDOWS, summarise_f3


def plot_f3(subject_scores, config, output_stem, *, synthetic_test_only=False):
    """Export 2×2 F3 boxplots plus subject dots as PNG (config DPI >=300) and PDF.

    Input: one row per subject/model/setting/window; scores are proportions.
    Boxes: Q1/median/Q3; whiskers: most extreme observations within 1.5 IQR.
    Dots include every subject, including outliers; dashed line: chance (0.5).
    The synthetic flag visibly labels temporary test figures.
    """
    summarise_f3(subject_scores)  # shared validation rejects raw folds/incomplete cohorts
    dpi = int(config["erp"]["figure_dpi"])
    if dpi < 300:
        raise ValueError("F3 PNG requires at least 300 dpi")
    stem = Path(output_stem)
    with plt.rc_context({"font.size": max(10, config["erp"]["font_size"]), "pdf.fonttype": 42}):
        # Near the report's 16 cm text width, so labels remain >=10 pt in print.
        fig, axes = plt.subplots(2, 2, figsize=(6.4, 5.5), sharex=True, sharey=True, layout="constrained")
        try:
            for row, setting in enumerate(F3_SETTINGS):
                for col, window in enumerate(F3_WINDOWS):
                    ax = axes[row, col]
                    group = subject_scores.loc[
                        subject_scores["setting"].eq(setting) & subject_scores["window"].eq(window)
                    ]
                    data = [
                        group.loc[group["model"].eq(m)].sort_values("subject")["bal_acc"].to_numpy() for m in F3_MODELS
                    ]
                    boxes = ax.boxplot(data, widths=0.55, whis=1.5, showfliers=False, patch_artist=True)
                    for index, (values, patch) in enumerate(zip(data, boxes["boxes"]), start=1):
                        patch.set_facecolor("#9ac3df" if index <= 2 else "#f2c185")
                        # Display offsets preserve scores and the global random state.
                        offsets = np.linspace(-0.17, 0.17, len(values))
                        ax.scatter(index + offsets, values, s=12, c="#253344", alpha=0.55, zorder=3)
                    ax.axhline(0.5, color="#666666", linestyle="--", linewidth=1)
                    start, end = config["classification"]["windows"][window]
                    ax.set_title(f"{setting.capitalize()} | {window}: {start * 1000:g}–{end * 1000:g} ms")
                    ax.text(0.98, 0.04, f"N = {len(data[0])} subjects", transform=ax.transAxes, ha="right")
                    ax.set_xticks(range(1, len(F3_MODELS) + 1), list(F3_MODELS))
                    ax.tick_params(axis="x", labelbottom=True, labelrotation=20)
                    ax.set_ylim(0, 1.02)
                    ax.grid(axis="y", alpha=0.18)
                    if col == 0:
                        ax.set_ylabel("Balanced accuracy (proportion)")
            if synthetic_test_only:
                fig.suptitle("SYNTHETIC TEST ONLY — not scientific results", color="darkred")
            stem.parent.mkdir(parents=True, exist_ok=True)
            png, pdf = stem.with_suffix(".png"), stem.with_suffix(".pdf")
            fig.savefig(png, dpi=dpi)
            fig.savefig(pdf)
        finally:
            plt.close(fig)
    return png, pdf
