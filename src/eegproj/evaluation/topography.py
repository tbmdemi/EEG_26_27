"""Group scalp maps of subject ERPs for R3; all amplitudes originate in volts."""

from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from .erp import _usable_release_log, collect_subject_erps


def collect_scalp_erps(root, config):
    """Return scalp ERPs and a mean head-coordinate montage in a result dictionary.

    Align all subjects by electrode name before averaging. Sensor positions in
    metres are averaged across subjects in their common head coordinate system
    for group visualization only. The returned positions table preserves each
    subject's geometry and the montage source recorded in the release log.
    """
    data_dir = Path(root) / config["paths"]["derivatives"]
    log = _usable_release_log(root, config)
    first_path = data_dir / f"{log.iloc[0]['subject']}_task-{config['dataset']['task']}_epo.fif"
    reference = mne.read_epochs(first_path, preload=False, verbose="error")
    channels = reference.ch_names.copy()
    del reference
    subjects, times, erps, counts = collect_subject_erps(root, config, channels)
    coordinates = []
    fiducials = {key: [] for key in ("nasion", "lpa", "rpa")}
    position_rows = []
    reordered_subjects = []
    for row in log.to_dict("records"):
        subject = row["subject"]
        path = data_dir / f"{subject}_task-{config['dataset']['task']}_epo.fif"
        epochs = mne.read_epochs(path, preload=False, verbose="error")
        if set(epochs.ch_names) != set(channels):
            raise ValueError(f"{subject}: electrode names differ from the reference")
        if epochs.ch_names != channels:
            reordered_subjects.append(subject)
        montage = epochs.get_montage()
        if montage is None:
            raise ValueError(f"{subject}: no electrode positions")
        positions = montage.get_positions()
        if positions["coord_frame"] != "head":
            raise ValueError(f"{subject}: expected head coordinates")
        xyz = np.array([positions["ch_pos"][channel] for channel in channels])
        if not np.isfinite(xyz).all() or np.unique(xyz, axis=0).shape[0] != len(channels):
            raise ValueError(f"{subject}: invalid or duplicate electrode positions")
        coordinates.append(xyz)
        for key, values in fiducials.items():
            if positions[key] is not None:
                values.append(positions[key])
        position_rows.append(
            pd.DataFrame(
                {
                    "subject": subject,
                    "channel": channels,
                    "x_m": xyz[:, 0],
                    "y_m": xyz[:, 1],
                    "z_m": xyz[:, 2],
                    "montage_source": row.get("montage", "unknown"),
                }
            )
        )
    mean_positions = np.mean(coordinates, axis=0)
    if np.unique(mean_positions, axis=0).shape[0] != len(channels):
        raise ValueError("The mean montage contains duplicate electrode positions")
    mean_fiducials = {key: np.mean(values, axis=0) for key, values in fiducials.items() if len(values) == len(subjects)}
    montage = mne.channels.make_dig_montage(
        ch_pos=dict(zip(channels, mean_positions)), coord_frame="head", **mean_fiducials
    )
    info = mne.create_info(channels, config["preprocessing"]["resample_sfreq"], ch_types="eeg")
    info.set_montage(montage)
    return {
        "subjects": subjects,
        "channels": channels,
        "times": times,
        "erps": erps,
        "epoch_counts": counts,
        "info": info,
        "positions": pd.concat(position_rows, ignore_index=True),
        "reordered_subjects": reordered_subjects,
    }


def summarise_scalp_maps(result, window_s, snapshot_times_s):
    """Return window-mean and time-snapshot tables in uV with between-subject SEM.

    Window means include the available samples within the requested interval.
    Snapshot maps use the nearest recorded sample, whose actual time is saved.
    Target-minus-standard is calculated within each subject before the SEM.
    """
    times = result["times"]
    start, stop = window_s
    if not (times[0] <= start < stop <= times[-1]):
        raise ValueError("The topomap interval must lie inside the saved epoch")
    mask = (times >= start) & (times <= stop)
    if not mask.any():
        raise ValueError("No recorded samples in the topomap interval")
    snapshots = np.asarray(snapshot_times_s, dtype=float)
    if snapshots.ndim != 1 or not len(snapshots) or not np.isfinite(snapshots).all():
        raise ValueError("Specify a non-empty list of finite snapshot times")
    if (snapshots < start).any() or (snapshots > stop).any():
        raise ValueError("Snapshot times must lie inside the topomap interval")
    indices = np.array([np.argmin(np.abs(times - value)) for value in snapshots])
    if len(np.unique(indices)) != len(indices):
        raise ValueError("Snapshot times map to duplicate recorded samples")
    waves = dict(result["erps"])
    waves["target_minus_standard"] = waves["target"] - waves["standard"]
    window_rows, snapshot_rows = [], []
    n_subjects = len(result["subjects"])
    if n_subjects < 2:
        raise ValueError("At least two subjects are required for a between-subject SEM")
    for condition, data in waves.items():
        expected_shape = (n_subjects, len(result["channels"]), len(times))
        if data.shape != expected_shape or not np.isfinite(data).all():
            raise ValueError(f"{condition}: invalid subject ERP array")
        per_subject = data[:, :, mask].mean(axis=-1) * 1e6
        window_rows.append(
            pd.DataFrame(
                {
                    "condition": condition,
                    "channel": result["channels"],
                    "mean_uv": per_subject.mean(axis=0),
                    "sem_uv": per_subject.std(axis=0, ddof=1) / np.sqrt(n_subjects),
                    "n_subjects": n_subjects,
                    "window_start_s": start,
                    "window_stop_s": stop,
                    "first_sample_s": times[mask][0],
                    "last_sample_s": times[mask][-1],
                    "n_samples": int(mask.sum()),
                }
            )
        )
        for requested, index in zip(snapshots, indices):
            per_subject = data[:, :, index] * 1e6
            snapshot_rows.append(
                pd.DataFrame(
                    {
                        "condition": condition,
                        "channel": result["channels"],
                        "requested_time_s": requested,
                        "actual_time_s": times[index],
                        "mean_uv": per_subject.mean(axis=0),
                        "sem_uv": per_subject.std(axis=0, ddof=1) / np.sqrt(n_subjects),
                        "n_subjects": n_subjects,
                    }
                )
            )
    return pd.concat(window_rows, ignore_index=True), pd.concat(snapshot_rows, ignore_index=True)


def plot_erp_topomap(result, window_table, config):
    """Return draft F2 with the Pz ERP (uV, subject SEM) and window-mean scalp maps."""
    conditions = ("standard", "target", "distractor", "target_minus_standard")
    labels = ("Standard", "Target", "Distractor", "Target - standard")
    colors = {"standard": "tab:blue", "target": "tab:orange", "distractor": "tab:green"}
    plt.rcParams.update({"font.size": config["erp"]["font_size"]})
    fig = plt.figure(figsize=(13, 8), layout="constrained")
    grid = fig.add_gridspec(2, len(conditions))
    erp_ax = fig.add_subplot(grid[0, :])
    pz_index = result["channels"].index("Pz")
    n_subjects = len(result["subjects"])
    for condition in conditions[:-1]:
        values = result["erps"][condition][:, pz_index, :] * 1e6
        mean = values.mean(axis=0)
        sem = values.std(axis=0, ddof=1) / np.sqrt(n_subjects)
        time_ms = result["times"] * 1000
        erp_ax.plot(time_ms, mean, label=condition, color=colors[condition])
        erp_ax.fill_between(time_ms, mean - sem, mean + sem, color=colors[condition], alpha=0.18)
    erp_ax.axhline(0, color="gray", linewidth=0.6)
    erp_ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    start, stop = config["erp"]["topomap_window_s"]
    erp_ax.axvspan(start * 1000, stop * 1000, color="gray", alpha=0.08, label="topomap interval")
    erp_ax.set(
        xlabel="Time from stimulus onset (ms)", ylabel="Pz amplitude (µV)", title="ERP: mean ± SEM across subjects"
    )
    erp_ax.set_xlim(result["times"][0] * 1000, result["times"][-1] * 1000)
    erp_ax.legend(fontsize=10, loc="upper left")
    axes = []
    limit = max(window_table["mean_uv"].abs().max(), np.finfo(float).eps)
    for index, (condition, label) in enumerate(zip(conditions, labels)):
        ax = fig.add_subplot(grid[1, index])
        values = window_table.loc[window_table["condition"].eq(condition)].set_index("channel")
        image, _ = mne.viz.plot_topomap(
            values.loc[result["channels"], "mean_uv"].to_numpy(),
            result["info"],
            axes=ax,
            show=False,
            cmap="RdBu_r",
            vlim=(-limit, limit),
        )
        ax.set_title(f"{label}\nMean {start * 1000:g}–{stop * 1000:g} ms", fontsize=config["erp"]["font_size"])
        axes.append(ax)
    fig.colorbar(image, ax=axes, shrink=0.7, label="Amplitude (µV)")
    fig.suptitle(
        f"F2 draft: {n_subjects} subjects (visual QC pending)\n"
        "Scalp maps use mean electrode positions in head coordinates",
        fontsize=config["erp"]["font_size"] + 1,
    )
    return fig


def plot_topomap_timecourse(result, snapshot_table, config):
    """Return scalp maps at recorded samples near the configured times, using one scale."""
    conditions = ("standard", "target", "distractor", "target_minus_standard")
    labels = ("Standard", "Target", "Distractor", "Target - standard")
    requested_times = config["erp"]["topomap_times_s"]
    plt.rcParams.update({"font.size": config["erp"]["font_size"]})
    fig, axes = plt.subplots(
        len(conditions), len(requested_times), figsize=(13, 11), layout="constrained", squeeze=False
    )
    limit = max(snapshot_table["mean_uv"].abs().max(), np.finfo(float).eps)
    for row, (condition, label) in enumerate(zip(conditions, labels)):
        for column, requested in enumerate(requested_times):
            values = snapshot_table.loc[
                snapshot_table["condition"].eq(condition) & snapshot_table["requested_time_s"].eq(requested)
            ].set_index("channel")
            image, _ = mne.viz.plot_topomap(
                values.loc[result["channels"], "mean_uv"].to_numpy(),
                result["info"],
                axes=axes[row, column],
                show=False,
                cmap="RdBu_r",
                vlim=(-limit, limit),
            )
            actual_ms = values["actual_time_s"].iloc[0] * 1000
            axes[row, column].set_title(f"{label}\n{actual_ms:g} ms", fontsize=config["erp"]["font_size"])
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.7, label="Amplitude (µV)")
    fig.suptitle(
        f"Scalp ERP time course: {len(result['subjects'])} subjects (visual QC pending)\n"
        "Common amplitude scale; mean electrode positions in head coordinates",
        fontsize=config["erp"]["font_size"] + 1,
    )
    return fig
