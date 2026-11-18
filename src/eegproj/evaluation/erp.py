"""Subject-level ERP averages and group summaries for the R3 notebooks."""

from pathlib import Path

import mne
import numpy as np
import pandas as pd


def _usable_release_log(root, config):
    """Read the included pipeline v2 subjects from the downloaded release log."""
    data_dir = Path(root) / config["paths"]["derivatives"]
    log = pd.read_csv(data_dir / "preproc_log.csv")
    usable = log.loc[log["pipeline_version"].eq(2) & log["excluded_subject"].eq(False)].copy()
    if usable.empty or usable["subject"].duplicated().any():
        raise ValueError("The release log must contain unique, usable pipeline v2 subjects.")
    return usable


def collect_subject_erps(root, config, channels=("Pz", "CPz", "Cz")):
    """Return subjects, times (s), ERPs (V; subject/channel/time) and epoch counts.

    Read the release log and average trials separately within each subject and
    condition. Only pipeline v2 subjects marked as included are used. Subjects
    receive equal weight in subsequent group summaries, regardless of trial count.
    This checks file structure and counts, not visual signal quality.
    """
    data_dir = Path(root) / config["paths"]["derivatives"]
    usable = _usable_release_log(root, config)
    conditions = ("standard", "target", "distractor")
    pre = config["preprocessing"]
    arrays = {condition: [] for condition in conditions}
    rows = []
    times = None
    for row in usable.to_dict("records"):
        subject = row["subject"]
        path = data_dir / f"{subject}_task-{config['dataset']['task']}_epo.fif"
        epochs = mne.read_epochs(path, preload=False, verbose="error")
        if not np.isclose(epochs.info["sfreq"], pre["resample_sfreq"]):
            raise ValueError(f"{subject}: unexpected sampling rate")
        if not np.allclose([epochs.tmin, epochs.tmax], [pre["epoch"]["tmin"], pre["epoch"]["tmax"]]):
            raise ValueError(f"{subject}: unexpected epoch window")
        if epochs.baseline is None or not np.allclose(epochs.baseline, pre["epoch"]["baseline"]):
            raise ValueError(f"{subject}: unexpected baseline")
        if set(epochs.event_id) != set(conditions):
            raise ValueError(f"{subject}: unexpected event names")
        if epochs.metadata is None or not {"subject", "onset_s", "block", "rt"} <= set(epochs.metadata):
            raise ValueError(f"{subject}: missing pipeline v2 metadata")
        if len(epochs.ch_names) != int(row["n_channels"]) or not set(channels) <= set(epochs.ch_names):
            raise ValueError(f"{subject}: unexpected channels")
        if times is None:
            times = epochs.times.copy()
        elif times.shape != epochs.times.shape or not np.allclose(times, epochs.times):
            raise ValueError(f"{subject}: time axis differs between subjects")
        epochs.load_data().pick(list(channels))
        counts = {}
        for condition in conditions:
            count = len(epochs[condition])
            if count == 0 or count != int(row[f"{condition}_after"]):
                raise ValueError(f"{subject}/{condition}: epoch count disagrees with release log")
            counts[condition] = count
            data = epochs[condition].average().get_data(picks=list(channels))
            if not np.isfinite(data).all():
                raise ValueError(f"{subject}/{condition}: non-finite ERP")
            arrays[condition].append(data)
        rows.append({"subject": subject, **counts, "visual_review": row.get("visual_review", "unknown")})
        print(f"{subject}: {counts}", flush=True)
        del epochs
    return (
        usable["subject"].tolist(),
        times,
        {key: np.stack(value) for key, value in arrays.items()},
        pd.DataFrame(rows),
    )


def summarise_erps(subject_erps, times, channels):
    """Return group mean and SEM (uV), with each subject contributing one ERP.

    Compute target-minus-standard within each subject before calculating its SEM.
    SEM uses the sample standard deviation (ddof=1) divided by sqrt(n_subjects).
    """
    waves = dict(subject_erps)
    waves["target_minus_standard"] = subject_erps["target"] - subject_erps["standard"]
    rows = []
    for condition, data in waves.items():
        n_subjects = data.shape[0]
        if n_subjects < 2:
            raise ValueError("At least two subjects are required for a between-subject SEM.")
        if data.shape[1:] != (len(channels), len(times)):
            raise ValueError("ERP dimensions do not match the channels and times.")
        mean = data.mean(axis=0) * 1e6
        sem = data.std(axis=0, ddof=1) / np.sqrt(n_subjects) * 1e6
        for index, channel in enumerate(channels):
            rows.append(
                pd.DataFrame(
                    {
                        "condition": condition,
                        "channel": channel,
                        "time_s": times,
                        "mean_uv": mean[index],
                        "sem_uv": sem[index],
                        "n_subjects": n_subjects,
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)
