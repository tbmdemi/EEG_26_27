"""Exploratory P300 candidate measurements; no candidate is a confirmed component."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def measure_candidate_peaks(subjects, times, subject_erps, epoch_counts, channels, config):
    """Return per-subject, condition and channel peak candidates (uV and ms).

    Select the maximum signed voltage in the fixed configured interval, without
    rectification, extra smoothing, interpolation or subject-specific windows.
    Preserve negative and boundary maxima but flag them for review. These are
    measurements of subject-averaged ERPs, not single-trial P300 measurements.
    """
    settings = config["erp"]["p300"]
    start, stop = settings["window_s"]
    margin = settings["edge_margin_s"]
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or len(times) < 3 or not np.isfinite(times).all() or not (np.diff(times) > 0).all():
        raise ValueError("A finite, increasing ERP time axis with at least three samples is required")
    if not (times[0] <= start < stop <= times[-1]) or not (0 <= margin < (stop - start) / 2):
        raise ValueError("Invalid peak interval or boundary margin")
    indices = np.flatnonzero((times >= start) & (times <= stop))
    if len(indices) < 3:
        raise ValueError("The peak interval must contain at least three recorded samples")
    if len(set(subjects)) != len(subjects) or len(set(channels)) != len(channels):
        raise ValueError("Subject and channel names must be unique")
    counts = epoch_counts.set_index("subject", verify_integrity=True)
    if set(counts.index) != set(subjects):
        raise ValueError("Epoch-count subjects differ from ERP subjects")
    resolution_ms = np.diff(times) * 1000
    if not np.allclose(resolution_ms, 1000 / config["preprocessing"]["resample_sfreq"]):
        raise ValueError("ERP sampling interval disagrees with config")
    rows = []
    for condition in settings["conditions"]:
        data = np.asarray(subject_erps[condition])
        if data.shape != (len(subjects), len(channels), len(times)) or not np.isfinite(data).all():
            raise ValueError(f"{condition}: invalid subject ERP array")
        for subject_index, subject in enumerate(subjects):
            for channel_index, channel in enumerate(channels):
                wave = data[subject_index, channel_index]
                values = wave[indices]
                local_index = int(np.argmax(values))
                index = indices[local_index]
                amplitude = wave[index]
                distance = min(times[index] - times[indices[0]], times[indices[-1]] - times[index])
                at_boundary = local_index in (0, len(indices) - 1)
                near_boundary = distance <= margin or np.isclose(distance, margin)
                is_local_maximum = bool(
                    0 < index < len(wave) - 1 and wave[index] > wave[index - 1] and wave[index] >= wave[index + 1]
                )
                n_ties = int(np.count_nonzero(values == amplitude))
                flags = []
                if amplitude <= 0:
                    flags.append("nonpositive_maximum")
                if at_boundary:
                    flags.append("boundary_maximum")
                if near_boundary:
                    flags.append("near_window_boundary")
                if not is_local_maximum:
                    flags.append("not_interior_local_maximum")
                if n_ties > 1:
                    flags.append("tied_maximum")
                rows.append(
                    {
                        "subject": subject,
                        "condition": condition,
                        "channel": channel,
                        "n_epochs": int(counts.loc[subject, condition]),
                        "candidate_peak_uv": amplitude * 1e6,
                        "candidate_latency_ms": times[index] * 1000,
                        "mean_window_uv": values.mean() * 1e6,
                        "sample_resolution_ms": float(np.median(resolution_ms)),
                        "window_start_s": start,
                        "window_stop_s": stop,
                        "first_sample_s": times[indices[0]],
                        "last_sample_s": times[indices[-1]],
                        "n_window_samples": len(indices),
                        "edge_distance_ms": distance * 1000,
                        "positive_maximum": bool(amplitude > 0),
                        "boundary_maximum": at_boundary,
                        "near_window_boundary": bool(near_boundary),
                        "interior_local_maximum": is_local_maximum,
                        "peak_needs_review": bool(flags),
                        "peak_flags": ";".join(flags),
                        "visual_review": counts.loc[subject, "visual_review"],
                        "measurement_status": settings["status"],
                    }
                )
    return pd.DataFrame(rows)


def summarise_candidate_peaks(candidates):
    """Return descriptive summaries of all candidates, keeping flagged rows included.

    No inferred component identity or automatic subject exclusions are applied.
    Summaries are grouped by electrode so one person is never counted three times
    in a single distribution.
    """
    return (
        candidates.groupby(["condition", "channel"], sort=False)
        .agg(
            n_subjects=("subject", "nunique"),
            n_peak_review=("peak_needs_review", "sum"),
            n_boundary=("boundary_maximum", "sum"),
            peak_median_uv=("candidate_peak_uv", "median"),
            peak_q25_uv=("candidate_peak_uv", lambda values: values.quantile(0.25)),
            peak_q75_uv=("candidate_peak_uv", lambda values: values.quantile(0.75)),
            latency_median_ms=("candidate_latency_ms", "median"),
            latency_q25_ms=("candidate_latency_ms", lambda values: values.quantile(0.25)),
            latency_q75_ms=("candidate_latency_ms", lambda values: values.quantile(0.75)),
            mean_window_median_uv=("mean_window_uv", "median"),
        )
        .reset_index()
    )


def plot_candidate_distribution(candidates, config):
    """Return paired subject plots of candidate amplitude (uV) and latency (ms) at Pz."""
    settings = config["erp"]["p300"]
    primary = candidates.loc[candidates["channel"].eq(settings["primary_channel"])]
    conditions = settings["conditions"]
    subjects = primary["subject"].drop_duplicates().tolist()
    plt.rcParams.update({"font.size": config["erp"]["font_size"]})
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
    offsets = np.linspace(-0.06, 0.06, len(subjects))
    for ax, metric, label in zip(
        axes,
        ("candidate_peak_uv", "candidate_latency_ms"),
        ("Candidate peak amplitude (µV)", "Candidate peak latency (ms)"),
    ):
        for subject, offset in zip(subjects, offsets):
            rows = primary.loc[primary["subject"].eq(subject)].set_index("condition").loc[conditions]
            x = np.arange(len(conditions)) + offset
            ax.plot(x, rows[metric], color="lightgray", linewidth=0.7, zorder=1)
            for condition_index, row in enumerate(rows.itertuples()):
                ax.scatter(
                    x[condition_index],
                    getattr(row, metric),
                    marker="x" if row.peak_needs_review else "o",
                    color="tab:orange" if row.peak_needs_review else "tab:blue",
                    s=24,
                    zorder=2,
                )
        ax.scatter([], [], marker="o", color="tab:blue", label="No peak-rule flag")
        ax.scatter([], [], marker="x", color="tab:orange", label="Peak needs review")
        ax.set_xticks(range(len(conditions)), conditions)
        ax.set(ylabel=label, title=label.split(" (")[0])
        ax.legend(fontsize=10)
    start, stop = settings["window_s"]
    fig.suptitle(
        f"Exploratory P300 candidates at {settings['primary_channel']}: {len(subjects)} paired subjects\n"
        f"Maximum in {start * 1000:g}–{stop * 1000:g} ms; flagged points retained; visual QC pending",
        fontsize=config["erp"]["font_size"] + 1,
    )
    return fig


def plot_candidate_review(subjects, times, subject_erps, candidates, channels, config):
    """Return a grid of individual Pz ERPs with candidate markers for manual review."""
    settings = config["erp"]["p300"]
    primary_channel = settings["primary_channel"]
    channel_index = channels.index(primary_channel)
    conditions = settings["conditions"]
    n_columns = 5
    n_rows = int(np.ceil(len(subjects) / n_columns))
    plt.rcParams.update({"font.size": config["erp"]["font_size"]})
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(16, 2.7 * n_rows), layout="constrained", squeeze=False)
    start, stop = settings["window_s"]
    for subject_index, (subject, ax) in enumerate(zip(subjects, axes.ravel())):
        for condition, color in zip(conditions, ("tab:orange", "tab:green")):
            row = candidates.loc[
                candidates["subject"].eq(subject)
                & candidates["channel"].eq(primary_channel)
                & candidates["condition"].eq(condition)
            ].iloc[0]
            ax.plot(
                times * 1000, subject_erps[condition][subject_index, channel_index] * 1e6, color=color, label=condition
            )
            ax.scatter(
                row["candidate_latency_ms"],
                row["candidate_peak_uv"],
                color=color,
                marker="x" if row["peak_needs_review"] else "o",
                zorder=3,
            )
        needs_review = candidates.loc[
            candidates["subject"].eq(subject) & candidates["channel"].eq(primary_channel), "peak_needs_review"
        ].any()
        ax.axvspan(start * 1000, stop * 1000, color="gray", alpha=0.10)
        ax.axhline(0, color="gray", linewidth=0.6)
        ax.axvline(0, color="gray", linewidth=0.6)
        ax.set(title=f"{subject}{' *' if needs_review else ''}", xlabel="Time (ms)", ylabel="Amplitude (µV)")
        ax.set_xlim(times[0] * 1000, times[-1] * 1000)
    for ax in axes.ravel()[len(subjects) :]:
        ax.set_visible(False)
    axes[0, 0].legend(fontsize=10)
    fig.suptitle(
        f"{primary_channel}: individual ERP candidate review (exploratory; visual QC pending)\n"
        "Circle = no peak-rule flag; cross / subject * = inspect candidate. Each panel has its own amplitude scale.",
        fontsize=config["erp"]["font_size"] + 1,
    )
    return fig
