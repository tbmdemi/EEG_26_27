"""Subject-level behavioral summaries from pre-rejection v2 logs for R3."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import excluded_subjects


def load_behaviour_subjects(root, config):
    """Return behavioral subject rows and exclusions from the downloaded release log.

    Rates use original stimulus/correct-response counts, before EEG rejection.
    The before-EEG-QC cohort excludes only the configured a-priori exclusions;
    the ERP cohort additionally follows excluded_subject=False in the release.
    Reaction times are the logged per-subject means in ms, before EEG rejection.
    """
    path = Path(root) / config["paths"]["derivatives"] / "preproc_log.csv"
    log = pd.read_csv(path)
    required = {
        "subject",
        "pipeline_version",
        "excluded_subject",
        "exclusion_reason",
        "visual_review",
        "flags",
        "rt_mean_ms",
        "rt_sd_ms",
        "hit_rate",
        "false_alarm_standard",
        "false_alarm_distractor",
    }
    for condition in ("standard", "target", "distractor"):
        required.update((f"{condition}_n", f"{condition}_correct"))
    if missing := required - set(log.columns):
        raise ValueError(f"Missing behavioral log columns: {sorted(missing)}")
    if log["subject"].duplicated().any():
        raise ValueError("Subjects must be unique in the release log")
    expected = {f"sub-{subject}" for subject in config["dataset"]["subjects"]}
    log = log.loc[log["pipeline_version"].eq(2)].copy()
    if set(log["subject"]) != expected:
        raise ValueError("The release log must cover the configured subjects at pipeline v2")
    if not log["excluded_subject"].isin([True, False]).all():
        raise ValueError("Every subject needs an explicit inclusion flag")
    priori = {f"sub-{subject}" for subject in excluded_subjects(config)}
    subjects = log.loc[~log["subject"].isin(priori)].copy()
    subjects["in_erp_cohort"] = subjects["excluded_subject"].eq(False)
    subjects["exclusion_reason"] = subjects["exclusion_reason"].fillna("")
    for condition in ("standard", "target", "distractor"):
        total = subjects[f"{condition}_n"].to_numpy(float)
        correct = subjects[f"{condition}_correct"].to_numpy(float)
        if not np.isfinite(total).all() or not np.isfinite(correct).all() or not (total > 0).all():
            raise ValueError(f"{condition}: missing or invalid behavioral counts")
        if (
            not ((correct >= 0) & (correct <= total)).all()
            or not np.allclose(total, np.round(total))
            or not np.allclose(correct, np.round(correct))
        ):
            raise ValueError(f"{condition}: correct/total counts must be valid integers")
        rate = correct / total if condition == "target" else 1 - correct / total
        column = "hit_rate" if condition == "target" else f"false_alarm_{condition}"
        if not np.allclose(rate, subjects[column], rtol=0, atol=1e-12):
            raise ValueError(f"{condition}: behavioral rates disagree with counts")
    if not np.isfinite(subjects[["rt_mean_ms", "rt_sd_ms"]].to_numpy(float)).all():
        raise ValueError("The behavioral cohort has missing RT summaries")
    if (
        not subjects["rt_mean_ms"]
        .between(config["behaviour"]["rt_min"] * 1000, config["behaviour"]["rt_max"] * 1000)
        .all()
    ):
        raise ValueError("Mean RT falls outside the behavioral scoring interval")
    if not subjects["rt_sd_ms"].ge(0).all():
        raise ValueError("Within-subject RT SD must be nonnegative")
    subjects["hit_rate_pct"] = subjects["hit_rate"] * 100
    subjects["target_omission_pct"] = (1 - subjects["hit_rate"]) * 100
    subjects["false_alarm_standard_pct"] = subjects["false_alarm_standard"] * 100
    subjects["false_alarm_distractor_pct"] = subjects["false_alarm_distractor"] * 100
    exclusions = log.loc[log["excluded_subject"].eq(True), ["subject", "exclusion_reason"]].copy()
    exclusions["stage"] = np.where(exclusions["subject"].isin(priori), "a_priori", "eeg_or_target_criteria")
    return subjects.reset_index(drop=True), exclusions.reset_index(drop=True)


def summarise_behaviour(subjects, config):
    """Return means and sample SDs across people, and descriptive paper comparisons.

    Each person has equal weight. RT SD is calculated across per-subject RT means,
    never by averaging the within-subject rt_sd_ms values. Cohorts overlap and
    comparisons are descriptive, without significance tests or replication claims.
    """
    reference = config["erp"]["behaviour_reference"]
    cohorts = {"before_eeg_qc": subjects, "erp_usable": subjects.loc[subjects["in_erp_cohort"]]}
    rows = []
    for cohort, table in cohorts.items():
        if len(table) < 2 or table["subject"].duplicated().any():
            raise ValueError("Each cohort needs at least two unique subjects")
        for metric, paper in reference["metrics"].items():
            values = table[metric].to_numpy(float)
            if not np.isfinite(values).all():
                raise ValueError(f"{cohort}/{metric}: missing behavioral values")
            rows.append(
                {
                    "cohort": cohort,
                    "metric": metric,
                    "label": paper["label"],
                    "unit": paper["unit"],
                    "n_subjects": len(table),
                    "mean": values.mean(),
                    "sd": values.std(ddof=1),
                    "median": np.median(values),
                    "min": values.min(),
                    "max": values.max(),
                    "paper_n_subjects": reference["n_subjects"],
                    "paper_mean": paper["mean"],
                    "paper_sd": paper["sd"],
                    "mean_difference_from_paper": values.mean() - paper["mean"],
                    "difference_unit": "ms" if paper["unit"] == "ms" else "percentage_points",
                    "paper_value_derived": paper.get("derived_from_omission", False),
                    "source_url": reference["source_url"],
                    "source_table": reference["source_table"],
                }
            )
    return pd.DataFrame(rows)


def plot_behaviour_comparison(subjects, summary, config):
    """Return behavioral subject dots and mean +/- SD for overlapping cohorts and paper."""
    reference = config["erp"]["behaviour_reference"]
    metrics = list(reference["metrics"])
    n_columns = 3
    n_rows = int(np.ceil((len(metrics) + 1) / n_columns))
    plt.rcParams.update({"font.size": config["erp"]["font_size"]})
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(14, 8), layout="constrained", squeeze=False)
    cohorts = {"before_eeg_qc": subjects, "erp_usable": subjects.loc[subjects["in_erp_cohort"]]}
    offsets = dict(zip(subjects["subject"], np.linspace(-0.09, 0.09, len(subjects))))
    tick_labels = [
        f"Before EEG QC\n(n={len(subjects)})",
        f"ERP cohort\n(n={subjects['in_erp_cohort'].sum()})",
        f"Paper\n(n={reference['n_subjects']})",
    ]
    for metric, ax in zip(metrics, axes.ravel()):
        for index, (cohort, table) in enumerate(cohorts.items()):
            row = summary.loc[summary["cohort"].eq(cohort) & summary["metric"].eq(metric)].iloc[0]
            ax.scatter(
                index + table["subject"].map(offsets).to_numpy(),
                table[metric],
                color="gray",
                alpha=0.55,
                s=13,
                zorder=1,
            )
            ax.errorbar(
                index, row["mean"], yerr=row["sd"], fmt="D", color=("tab:blue", "tab:green")[index], capsize=5, zorder=3
            )
        paper = reference["metrics"][metric]
        ax.errorbar(2, paper["mean"], yerr=paper["sd"], fmt="D", color="black", capsize=5, zorder=3)
        ax.set_xticks(range(3), tick_labels, fontsize=10)
        unit = "ms" if paper["unit"] == "ms" else "%"
        ax.set(title=paper["label"], ylabel=f"{paper['label']} ({unit})", xlim=(-0.35, 2.35))
    note_ax = axes.ravel()[len(metrics)]
    note_ax.set_axis_off()
    note_ax.text(
        0,
        0.9,
        "Dots: individual subjects\nDiamonds / bars: mean ± subject SD\n\n"
        "The two local cohorts overlap.\nRates use pre-rejection behavior counts.\n"
        "RT SD is across subject RT means.\nPaper values: Dzianok et al., Table 5.\n"
        "Paper hit rate is derived from omission.\n\n"
        "SD bars describe dispersion and can\nextend beyond the observed rate range.\n"
        "No significance test is performed.",
        transform=note_ax.transAxes,
        va="top",
        fontsize=config["erp"]["font_size"],
    )
    for ax in axes.ravel()[len(metrics) + 1 :]:
        ax.set_visible(False)
    fig.suptitle(
        "Behavioral summary: subject-level observations and published reference",
        fontsize=config["erp"]["font_size"] + 1,
    )
    return fig
