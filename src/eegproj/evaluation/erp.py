"""Subject-level ERP averages and group summaries for the R3 notebooks."""

from pathlib import Path

import mne
import numpy as np
import pandas as pd


def _usable_release_log(root, config):
    """Validate the complete v2 release and match the committed handoff cohort."""
    root = Path(root)
    data_dir = Path(root) / config["paths"]["derivatives"]
    log = pd.read_csv(data_dir / "preproc_log.csv")
    handoff = pd.read_csv(root / "results" / "preproc_log.csv")
    expected = {f"sub-{subject}" for subject in config["dataset"]["subjects"]}
    excluded = {f"sub-{subject}" for subject in config["dataset"].get("exclude", {})}
    for label, table in (("release", log), ("handoff", handoff)):
        if not {"subject", "pipeline_version", "excluded_subject"} <= set(table.columns):
            raise ValueError(f"{label}: missing subject/version/inclusion columns")
        if table["subject"].duplicated().any() or set(table["subject"]) != expected:
            raise ValueError(f"{label}: log must cover all configured subjects exactly once")
        if not table["pipeline_version"].eq(2).all():
            raise ValueError(f"{label}: all log rows must use pipeline v2")
        if not table["excluded_subject"].isin([True, False]).all():
            raise ValueError(f"{label}: missing or invalid subject inclusion flag")
        if table.loc[table["subject"].isin(excluded), "excluded_subject"].eq(False).any():
            raise ValueError(f"{label}: an a-priori excluded subject was marked usable")
    usable = log.loc[log["excluded_subject"].eq(False)].copy()
    handoff_subjects = set(handoff.loc[handoff["excluded_subject"].eq(False), "subject"])
    if usable.empty or set(usable["subject"]) != handoff_subjects:
        raise ValueError("Release cohort differs from results/preproc_log.csv; check the data release")
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
        metadata = epochs.metadata
        metadata_subjects = metadata["subject"].astype(str).str.removeprefix("sub-").str.zfill(2)
        if not metadata_subjects.eq(subject.removeprefix("sub-")).all():
            raise ValueError(f"{subject}: metadata belongs to another subject")
        if not np.isfinite(metadata["onset_s"]).all() or not metadata["onset_s"].is_monotonic_increasing:
            raise ValueError(f"{subject}: invalid or unordered stimulus onsets")
        if not metadata["block"].isin(range(config["evaluation"]["within"]["n_splits"])).all():
            raise ValueError(f"{subject}: invalid block metadata")
        is_target = epochs.events[:, 2] == epochs.event_id["target"]
        if (
            not metadata.loc[is_target, "rt"]
            .between(config["behaviour"]["rt_min"], config["behaviour"]["rt_max"])
            .all()
        ):
            raise ValueError(f"{subject}: target RT metadata is missing or outside the behavioral interval")
        if metadata.loc[~is_target, "rt"].notna().any():
            raise ValueError(f"{subject}: non-target trials contain a response time")
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
