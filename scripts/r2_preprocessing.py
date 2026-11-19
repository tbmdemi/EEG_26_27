"""R2 candidate preprocessing; released v2 data remain read-only."""

import os
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "4")
ROOT = Path(__file__).resolve().parents[1]
"""Reproducible oddball preprocessing, with original event timing and QC logs."""
import gc
import hashlib
import json
import logging
import re
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import yaml
from mne_bids import BIDSPath, read_raw_bids
from mne_icalabel.iclabel import iclabel_label_components
from scipy.signal import welch
from threadpoolctl import threadpool_limits

CONFIG = yaml.safe_load((ROOT / "configs/config.yaml").read_text(encoding="utf-8"))


import copy
import tempfile
import urllib.error
import urllib.request
from urllib.parse import urlencode

import defusedxml.ElementTree as ET

IMPLEMENTATION_REVISION = "r2-notebook-v2.1"
SOURCE_URL = "https://s3.amazonaws.com/openneuro.org/ds004621/"


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, np.generic):
            return clean(x.item())
        if isinstance(x, float) and not np.isfinite(x):
            return None
        return x

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(value), indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


import ast

IMPLEMENTATION_SHA256 = hashlib.sha256(
    ast.dump(ast.parse(Path(__file__).read_text(encoding="utf-8")), include_attributes=False).encode()
).hexdigest()


def config_fingerprint(config):
    scientific = {k: config[k] for k in ["seed", "dataset", "behaviour", "preprocessing"]}
    return hashlib.sha256(json.dumps(scientific, sort_keys=True).encode()).hexdigest()


def candidate_config(root, config):
    result = copy.deepcopy(config)
    result["paths"]["derivatives"] = str(Path(config["paths"]["derivatives"]) / "_candidate")
    result["_r2"] = {
        "qc": "figures/preprocessing/candidate",
        "log": "results/r2_candidate/preproc_log.csv",
        "raw_root": str((Path(root) / config["paths"]["raw"]).resolve()),
    }
    return result


def require_candidate(root, config):
    root = Path(root).resolve()
    published = yaml.safe_load((root / "configs/config.yaml").read_text(encoding="utf-8"))
    allowed = (root / published["paths"]["derivatives"] / "_candidate").resolve()
    out = (root / config["paths"]["derivatives"]).resolve()
    if out != allowed:
        raise ValueError("Preprocessing writes only to the configured derivatives/_candidate directory")
    return out


def has_recording(rawroot, subject, task):
    prefix = f"sub-{subject}_task-{task}"
    folder = Path(rawroot) / f"sub-{subject}" / "eeg"
    return all(
        (folder / f"{prefix}{suffix}").is_file()
        for suffix in ["_eeg.eeg", "_eeg.vhdr", "_eeg.vmrk", "_events.tsv", "_channels.tsv"]
    )


def fetch_file(relative, rawroot):
    destination = Path(rawroot) / relative
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    urllib.request.urlretrieve(SOURCE_URL + relative, temporary)
    temporary.replace(destination)
    return destination


def download_subject(rawroot, subject, task):
    # Exact oddball prefix, never another task or another participant.
    prefix = f"ds004621/sub-{subject}/eeg/sub-{subject}_task-{task}_"
    query = urlencode({"list-type": "2", "prefix": prefix})
    listing = ET.fromstring(
        urllib.request.urlopen("https://s3.amazonaws.com/openneuro.org?" + query, timeout=60).read()
    )
    if any(e.text == "true" for e in listing.iter() if e.tag.endswith("}IsTruncated")):
        raise RuntimeError("Unexpected truncated subject listing")
    keys = [e.text for e in listing.iter() if e.tag.endswith("}Key")]
    if not keys:
        raise FileNotFoundError(prefix)
    for key in keys:
        if not key.startswith(prefix):
            raise ValueError("Unexpected download key")
        if Path(key).suffix in {".eeg", ".vhdr", ".vmrk", ".tsv", ".json"}:
            fetch_file(key.removeprefix("ds004621/"), rawroot)
    for name in [
        "dataset_description.json",
        "participants.tsv",
        "participants.json",
        f"task-{task}_events.json",
        f"task-{task}_eeg.json",
    ]:
        try:
            fetch_file(name, rawroot)
        except urllib.error.HTTPError as error:
            if error.code != 404 or name == "dataset_description.json":
                raise
    if not has_recording(rawroot, subject, task):
        raise FileNotFoundError(f"Incomplete oddball files: sub-{subject}")


def ensure_coordinates(rawroot, subject, download=False):
    base = f"sourcedata/sub-{subject}/coords/sub-{subject}_captrak"
    for suffix in [".bvct", ".bvef"]:
        path = Path(rawroot) / (base + suffix)
        if path.is_file():
            return path
    if download:
        for suffix in [".bvct", ".bvef"]:
            try:
                return fetch_file(base + suffix, rawroot)
            except urllib.error.HTTPError as error:
                if error.code != 404:
                    raise
    raise FileNotFoundError(f"No subject coordinates (.bvct or .bvef): sub-{subject}")


def read_subject_montage(path):
    path = Path(path)
    if path.suffix.lower() == ".bvct":
        return mne.channels.read_dig_captrak(path), "subject CapTrak"
    if path.suffix.lower() == ".bvef":
        # MNE default head-size normalization matches the existing v2 release.
        montage = mne.channels.read_custom_montage(path, coord_frame="head")
        return montage, "subject BrainVision electrodes (.bvef); MNE default head-size normalization"
    raise ValueError(f"Unsupported coordinate file: {path}")


def read_bids_readonly(root, rawroot, subject, task, channel_aliases=None):
    # Decode Latin-1 sidecars in a temporary BIDS view, never in the original raw tree.
    with tempfile.TemporaryDirectory(prefix="r2-bids-") as temporary:
        mirror = Path(temporary)
        paths = list(Path(rawroot).glob("*.json")) + list(Path(rawroot).glob("participants*.tsv"))
        paths += list((Path(rawroot) / f"sub-{subject}" / "eeg").glob(f"sub-{subject}_task-{task}_*"))
        for source in paths:
            if not source.is_file():
                continue
            target = mirror / source.relative_to(rawroot)
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.suffix.lower() == ".tsv":
                data = source.read_bytes()
                try:
                    text = data.decode("utf-8-sig")
                except UnicodeDecodeError:
                    text = data.decode("latin-1")
                target.write_text(text, encoding="utf-8", newline="")
            else:
                # A filesystem link avoids copying a ~0.7 GB recording. No writes to links.
                try:
                    os.link(source, target)
                except OSError:
                    shutil.copy2(source, target)
        prefix = f"sub-{subject}_task-{task}"
        header = mirror / f"sub-{subject}" / "eeg" / f"{prefix}_eeg.vhdr"
        channel_table = header.with_name(f"{prefix}_channels.tsv")
        header_names = mne.io.read_raw_brainvision(header, preload=False, verbose="error").ch_names
        table = pd.read_csv(channel_table, sep="\t")
        aliases = channel_aliases or {}
        canonical_header = [aliases.get(ch, ch) for ch in header_names]
        canonical_table = [aliases.get(ch, ch) for ch in table["name"]]
        if len(set(canonical_table)) != len(canonical_table) or set(canonical_table) != set(canonical_header):
            raise ValueError("channels.tsv and BrainVision contain different/ambiguous channel identities")
        aligned = table["name"].tolist() != header_names
        if aligned:
            table = table.set_axis(canonical_table).loc[canonical_header].reset_index(drop=True)
            table["name"] = header_names
            table.to_csv(channel_table, sep="\t", index=False)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Unable to map the following column")
            raw = read_raw_bids(
                BIDSPath(root=mirror, subject=subject, task=task, datatype="eeg", suffix="eeg", extension=".vhdr"),
                verbose=False,
            )
            raw.load_data()
            # Loaded data must not retain references to the temporary mirror.
            raw.filenames = [None for _ in raw.filenames]
            raw._r2_sidecar_aligned = aligned
        return raw


def raw_fingerprint(rawroot, subject, task, coordinates):
    prefix = f"sub-{subject}_task-{task}"
    paths = list((Path(rawroot) / f"sub-{subject}" / "eeg").glob(prefix + "_*"))
    paths += list(Path(rawroot).glob("*.json")) + [Path(coordinates)]
    return {p.relative_to(rawroot).as_posix(): file_sha256(p) for p in sorted(set(paths)) if p.is_file()}


def complete_summary(out, prefix, config):
    path = Path(out) / f"{prefix}_summary.json"
    if not path.is_file():
        return None
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
        if (
            row.get("processing_status") != "complete"
            or row.get("pipeline_version") != PIPELINE_VERSION
            or row.get("implementation_revision") != IMPLEMENTATION_REVISION
            or row.get("implementation_sha256") != IMPLEMENTATION_SHA256
            or row.get("config_fingerprint") != config_fingerprint(config)
        ):
            return None
        artifacts = row.get("artifacts", {})
        if row.get("excluded_subject") is False and not artifacts:
            return None
        if row.get("excluded_subject") is False and not all(
            f"{prefix}_{suffix}.fif" in artifacts for suffix in ["epo", "ica"]
        ):
            return None
        rawroot = Path(config["_r2"]["raw_root"])
        for name, expected in row.get("source_sha256", {}).items():
            source = rawroot / name
            if source.exists() and file_sha256(source) != expected:
                return None
        for name, expected in artifacts.items():
            target = (Path(out) / name).resolve()
            if not target.is_relative_to(Path(out).resolve()):
                return None
            if not target.is_file() or file_sha256(target) != expected:
                return None
        return row
    except (ValueError, OSError):
        return None


class ICAProgress(logging.Handler):
    """Read true iteration progress (Infomax n_iter can equal max on convergence)."""

    def __init__(self):
        super().__init__()
        self.step = 0
        self.change = float("inf")

    def emit(self, record):
        match = re.search(r"step (\d+) - .*wchange\s+([\d.eE+-]+)", record.getMessage())
        if match:
            self.step, self.change = int(match[1]), float(match[2])


def bad_channel_scores(raw, cfg):
    """Flag flat channels or extreme log standard deviation (robust MAD z)."""
    data = raw.get_data()[:, :: cfg["sample_step"]]
    std = np.std(data, axis=1)
    logs = np.log(np.maximum(std, np.finfo(float).tiny))
    mad = np.median(np.abs(logs - np.median(logs)))
    z = (logs - np.median(logs)) / max(1.4826 * mad, np.finfo(float).eps)
    frame = pd.DataFrame({"channel": raw.ch_names, "std_uv": std * 1e6, "robust_z": z})
    frame["auto_bad"] = (np.abs(z) > cfg["robust_z"]) | (std * 1e6 < cfg["flat_std_uv"])
    return frame


def qc_plot(before, raw, scores, output, cfg):
    """Save signal, spectrum, and channel amplitude checks for visual review."""
    sfreq, original = before
    cleaned = raw.get_data(start=0, stop=int(cfg["duration_s"] * raw.info["sfreq"]))
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for data, rate, label in [(original, sfreq, "Raw"), (cleaned, raw.info["sfreq"], "Cleaned")]:
        f, power = welch(data, fs=rate, nperseg=min(int(rate * 2), data.shape[1]), axis=-1)
        axes[0, 0].plot(f, 10 * np.log10(np.maximum(np.median(power, axis=0) * 1e12, 1e-30)), label=label)
    axes[0, 0].set(xlim=(0, 100), xlabel="Frequency (Hz)", ylabel="Median PSD (dB uV²/Hz)")
    axes[0, 0].legend()
    axes[0, 1].bar(np.arange(len(scores)), scores["std_uv"], color=np.where(scores["auto_bad"], "red", "steelblue"))
    axes[0, 1].set(yscale="log", xlabel="Channel index", ylabel="Raw standard deviation (uV)")
    picks = [raw.ch_names.index(n) for n in ["Fp1", "C3", "Cz", "Pz", "O1"]]
    for ax, data, rate, title in [
        (axes[1, 0], original, sfreq, "Before"),
        (axes[1, 1], cleaned, raw.info["sfreq"], "After"),
    ]:
        count = min(int(rate * cfg["trace_duration_s"]), data.shape[1])
        for i in picks:
            ax.plot(np.arange(count) / rate, data[i, :count] * 1e6, label=raw.ch_names[i], linewidth=0.6)
        ax.set(title=title, xlabel="Time (s)", ylabel="Amplitude (uV)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=cfg["dpi"])
    plt.close(fig)


def excluded_subjects(config):
    return {str(k): v for k, v in (config["dataset"].get("exclude") or {}).items()}


"""Behavioural scoring of the oddball task from the BIDS *_events.tsv (protocol §1).

A response (event_type 'S  1') counts for a stimulus when it falls in [rt_min, rt_max] seconds
after the stimulus onset (Dzianok et al. 2022: 100 ms to 200 ms + 1200 ms shortest ISI).
    target      correct = at least one response in the window; RT = first response
    standard    correct = no response in the window
    distractor  correct = no response in the window
"""

STIMULUS_CODES = {
    CONFIG["dataset"]["events"][k].removeprefix("Stimulus/"): k for k in ["standard", "target", "distractor"]
}
RESPONSE_CODE = CONFIG["dataset"]["events"]["response"].removeprefix("Stimulus/")


def score_trials(table, rt_min, rt_max):
    """Return one row per stimulus: onset (s), condition, correct (bool), rt (s, NaN if none).

    `table` is the events.tsv DataFrame with columns `onset` and `event_type`.
    """
    if not {"event_type", "onset"} <= set(table.columns):
        raise ValueError("events.tsv needs event_type and onset columns")
    if not np.isfinite(table["onset"].to_numpy(float)).all():
        raise ValueError("Non-finite event onsets")
    codes = table["event_type"].astype(str)
    responses = np.sort(table.loc[codes == RESPONSE_CODE, "onset"].to_numpy(float))
    stim = table.loc[codes.isin(STIMULUS_CODES)].sort_values("onset")
    rows = []
    for onset, code in zip(stim["onset"].to_numpy(float), codes[stim.index]):
        latency = responses - onset
        hits = latency[(latency >= rt_min - 1e-9) & (latency <= rt_max + 1e-9)]
        condition = STIMULUS_CODES[code]
        responded = hits.size > 0
        rows.append(
            {
                "onset_s": onset,
                "condition": condition,
                "correct": responded if condition == "target" else not responded,
                "rt": float(hits.min()) if responded else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=["onset_s", "condition", "correct", "rt"])


def summarise(trials):
    """Behavioural summary for the log: counts, hit rate, false-alarm rates, RT of correct targets."""
    out = {"n_stimuli": len(trials)}
    for condition in STIMULUS_CODES.values():
        sub = trials[trials.condition == condition]
        out[f"{condition}_n"] = len(sub)
        out[f"{condition}_correct"] = int(sub.correct.sum())
    target = trials[trials.condition == "target"]
    out["hit_rate"] = float(target.correct.mean()) if len(target) else np.nan
    for condition in ("standard", "distractor"):
        sub = trials[trials.condition == condition]
        out[f"false_alarm_{condition}"] = float(1 - sub.correct.mean()) if len(sub) else np.nan
    rts = target.loc[target.correct, "rt"]
    out["rt_mean_ms"] = float(rts.mean() * 1000) if len(rts) else np.nan
    out["rt_sd_ms"] = float(rts.std() * 1000) if len(rts) > 1 else np.nan
    return out


def match_onsets(event_onsets, trial_onsets, tol=0.002):
    """Index into `trial_onsets` for each event onset (nearest, must be within `tol` seconds)."""
    trial_onsets = np.asarray(trial_onsets, float)
    event_onsets = np.asarray(event_onsets, float)
    if not len(event_onsets):
        return np.array([], dtype=int)
    if not len(trial_onsets):
        raise ValueError("No trial onsets available")
    order = np.argsort(trial_onsets)
    sorted_onsets = trial_onsets[order]
    pos = np.clip(np.searchsorted(sorted_onsets, event_onsets), 1, len(sorted_onsets) - 1)
    left, right = sorted_onsets[pos - 1], sorted_onsets[pos]
    nearest = np.where(np.abs(event_onsets - left) <= np.abs(event_onsets - right), pos - 1, pos)
    if np.max(np.abs(sorted_onsets[nearest] - event_onsets)) > tol:
        raise ValueError("Epoch events do not match events.tsv onsets")
    return order[nearest]


"""Single-subject preprocessing pipeline and batch runner (protocol section 1).

Reproducible oddball preprocessing, with original event timing and QC logs.
Only behaviourally correct trials are epoched (protocol §1, config `behaviour`).
"""
import hashlib
import shutil
from pathlib import Path

import yaml

# Bump when the content of the saved epochs changes; older summaries are then re-processed.
# v1: all trials (2026-11-13). v2: correct trials only, rt in metadata, discontinuities marked BAD.
PIPELINE_VERSION = 2


def _prefix(subject, config):
    return f"sub-{subject}_task-{config['dataset']['task']}"


def _discontinuities(vmrk_path):
    """Read every positive New Segment sample, including the first marker if not at t=0."""
    vmrk_path = Path(vmrk_path)
    if not vmrk_path.exists():
        return np.array([])
    header = vmrk_path.with_suffix(".vhdr").read_text(encoding="utf-8", errors="replace")
    match = re.search(r"^SamplingInterval=([0-9.eE+-]+)", header, re.MULTILINE | re.IGNORECASE)
    if match is None or float(match.group(1)) <= 0:
        raise ValueError("Missing/invalid BrainVision SamplingInterval")
    interval_s = float(match.group(1)) * 1e-6
    markers = vmrk_path.read_text(encoding="utf-8", errors="replace")
    positions = [
        int(p) for p in re.findall(r"^Mk[0-9]+=New Segment,[^,]*,([0-9]+),", markers, re.MULTILINE | re.IGNORECASE)
    ]
    return np.array(sorted({(p - 1) * interval_s for p in positions if p > 1}))


def preprocess(root, subject, config):
    """Process one full recording and save the correct trials of all three stimulus classes in FIF."""
    start = time.monotonic()
    root = Path(root)
    pre = config["preprocessing"]
    beh_cfg = config["behaviour"]
    task = config["dataset"]["task"]
    rawroot = root / config["paths"]["raw"]
    out = require_candidate(root, config)
    qc = root / config["_r2"]["qc"]
    out.mkdir(parents=True, exist_ok=True)
    qc.mkdir(parents=True, exist_ok=True)
    prefix = _prefix(subject, config)
    print(f"BEGIN {prefix}", flush=True)

    # Behaviour first: cheap, and decides whether the subject is usable at all.
    table = pd.read_csv(rawroot / f"sub-{subject}" / "eeg" / f"{prefix}_events.tsv", sep="\t")
    trials = score_trials(table, beh_cfg["rt_min"], beh_cfg["rt_max"])
    behaviour = summarise(trials)
    flags = []
    if subject in (config["dataset"].get("flag") or {}):
        flags.append(config["dataset"]["flag"][subject])
    if behaviour["n_stimuli"] != config["dataset"]["expected_stimuli"]:
        flags.append(f"n_stimuli={behaviour['n_stimuli']}")
    if behaviour["hit_rate"] < beh_cfg["flag_hit_rate_below"]:
        flags.append(f"hit_rate={behaviour['hit_rate']:.2f}")
    row = {
        "subject": f"sub-{subject}",
        "pipeline_version": PIPELINE_VERSION,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "implementation_sha256": IMPLEMENTATION_SHA256,
        "config_fingerprint": config_fingerprint(config),
        "processing_status": "running",
        "source_sha256": {
            (rawroot / f"sub-{subject}" / "eeg" / f"{prefix}_events.tsv").relative_to(rawroot).as_posix(): file_sha256(
                rawroot / f"sub-{subject}" / "eeg" / f"{prefix}_events.tsv"
            )
        },
        **behaviour,
    }
    write_json(out / f"{prefix}_summary.json", row)
    if behaviour["target_correct"] < beh_cfg["min_correct_targets"]:
        _set_aside(out, prefix)  # never leave an older epoch file of an excluded subject in the shared folder
        row.update(
            processing_status="complete",
            artifacts={},
            excluded_subject=True,
            exclusion_reason=f"correct_targets={behaviour['target_correct']}",
            flags="; ".join(flags),
            seconds=round(time.monotonic() - start, 1),
        )
        write_json(out / f"{prefix}_summary.json", row)
        print("EXCLUDED " + json.dumps(row), flush=True)
        return row

    raw = read_bids_readonly(root, rawroot, subject, task, pre["channel_aliases"])
    if raw._r2_sidecar_aligned:
        flags.append("channels.tsv aligned by name to BrainVision order in temporary BIDS view")
    raw.pick("eeg")
    raw.rename_channels({k: v for k, v in pre["channel_aliases"].items() if k in raw.ch_names})
    if len(raw.ch_names) != 127:
        raise ValueError("Expected 127 recorded EEG channels")
    coordinates = ensure_coordinates(rawroot, subject)
    montage, montage_source = read_subject_montage(coordinates)
    raw.set_montage(montage, on_missing="raise")
    if not all(np.isfinite(ch["loc"][:3]).all() for ch in raw.info["chs"]):
        raise ValueError("Non-finite electrode coordinates")
    source_hashes = raw_fingerprint(rawroot, subject, task, coordinates)
    descriptions = ["Stimulus/" + v for v in table["event_type"]]
    annotations = mne.Annotations(table["onset"].to_numpy(), np.zeros(len(table)), descriptions)
    # Joins between concatenated files (sub-37, sub-38): skip in filtering, ICA fit and epoching.
    joins = _discontinuities(rawroot / f"sub-{subject}" / "eeg" / f"{prefix}_eeg.vmrk")
    pad = pre["discontinuity_pad_s"]
    if joins.size:
        onsets = np.clip(joins - pad, 0, None)
        annotations += mne.Annotations(onsets, joins + pad - onsets, ["BAD_ACQ_SKIP"] * joins.size)
        flags.append(f"discontinuities={joins.size}")
    raw.set_annotations(annotations)
    raw.load_data()
    if not np.isfinite(raw.get_data()).all():
        raise ValueError("Non-finite raw samples")
    before = (raw.info["sfreq"], raw.get_data(start=0, stop=int(pre["qc"]["duration_s"] * raw.info["sfreq"])))
    scores = bad_channel_scores(raw, pre["bad_channels"])
    auto = scores.loc[scores.auto_bad, "channel"].tolist()
    bads = sorted(set(raw.info["bads"] + auto + pre["bad_channels"]["manual"].get(subject, [])))
    raw.info["bads"] = bads
    scores.to_csv(qc / f"{prefix}_channels.csv", index=False)
    raw.filter(
        pre["l_freq"],
        pre["h_freq"],
        method="fir",
        phase="zero",
        skip_by_annotation=("edge", "BAD_ACQ_SKIP"),
        verbose=False,
    )
    # With this protocol, 50 Hz lies outside the 40 Hz passband.
    notch_applied = False
    if pre.get("notch") and pre["notch"] < pre["h_freq"]:
        raw.notch_filter(pre["notch"], verbose=False)
        notch_applied = True
    if bads:
        raw.interpolate_bads(reset_bads=True, mode="accurate", method={"eeg": "spline"}, verbose=False)
    raw.set_eeg_reference(pre["reference"], projection=False, verbose=False)
    fit_raw = raw.copy().filter(
        pre["ica"]["fit_l_freq"], None, skip_by_annotation=("edge", "BAD_ACQ_SKIP"), verbose=False
    )
    ica = mne.preprocessing.ICA(
        n_components=pre["ica"]["n_components"],
        method=pre["ica"]["method"],
        fit_params={"extended": pre["ica"]["extended"], "w_change": pre["ica"]["w_change"], "verbose": True},
        max_iter=pre["ica"]["max_iter"],
        random_state=config["seed"],
    )
    print(f"{prefix}: ICA fit, bad channels={bads}", flush=True)
    progress = ICAProgress()
    mne.utils.logger.addHandler(progress)
    try:
        ica.fit(fit_raw, decim=pre["ica"]["decim"], reject_by_annotation=True, verbose=True)
    finally:
        mne.utils.logger.removeHandler(progress)
    if not progress.step or (progress.step >= pre["ica"]["max_iter"] and progress.change >= pre["ica"]["w_change"]):
        raise RuntimeError("ICA convergence not verified from iteration log")
    probabilities = iclabel_label_components(fit_raw, ica, backend="onnx")
    names = ["brain", "muscle artifact", "eye blink", "heart beat", "line noise", "channel noise", "other"]
    labels = pd.DataFrame(probabilities, columns=names)
    labels.index.name = "component"
    reject_cols = pre["ica"]["iclabel_reject"]
    labels["excluded"] = labels[reject_cols].max(axis=1) >= pre["ica"]["iclabel_threshold"]
    ica.exclude = labels.index[labels.excluded].tolist()
    labels.to_csv(qc / f"{prefix}_iclabel.csv")
    ica.save(out / f"{prefix}_ica.fif", overwrite=True, verbose=False)
    if ica.exclude:
        figs = ica.plot_components(picks=ica.exclude, show=False)
        if not isinstance(figs, list):
            figs = [figs]
        for i, fig in enumerate(figs):
            fig.savefig(qc / f"{prefix}_excluded_ics_{i}.png", dpi=pre["qc"]["dpi"])
            plt.close(fig)
    del fit_raw
    gc.collect()
    ica.apply(raw, verbose=False)
    event_id = {name: i + 1 for i, name in enumerate(["standard", "target", "distractor"])}
    mapped = {config["dataset"]["events"][k]: v for k, v in event_id.items()}
    events, _ = mne.events_from_annotations(raw, event_id=mapped, verbose=False)
    original_onsets = events[:, 0] / raw.info["sfreq"]
    # Attach behaviour to every stimulus event and keep only correct trials.
    index = match_onsets(original_onsets, trials["onset_s"].to_numpy())
    names_by_code = {v: k for k, v in event_id.items()}
    if not (trials["condition"].to_numpy()[index] == [names_by_code[c] for c in events[:, 2]]).all():
        raise ValueError("Stimulus classes in annotations and events.tsv disagree")
    correct = trials["correct"].to_numpy(bool)[index]
    rts = trials["rt"].to_numpy(float)[index]
    break_code = config["dataset"]["events"]["break"].removeprefix("Stimulus/")
    breaks = np.sort(table.loc[table.event_type == break_code, "onset"].to_numpy())
    blocks = np.searchsorted(breaks, original_onsets, side="right")
    counts_before = {k: int(np.sum(events[:, 2] == v)) for k, v in event_id.items()}
    events, original_onsets, blocks, rts = events[correct], original_onsets[correct], blocks[correct], rts[correct]
    counts_correct = {k: int(np.sum(events[:, 2] == v)) for k, v in event_id.items()}
    raw, events = raw.resample(pre["resample_sfreq"], events=events, verbose=False)
    qc_plot(before, raw, scores, qc / f"{prefix}_qc.png", pre["qc"])
    ep = pre["epoch"]
    epochs = mne.Epochs(
        raw,
        events,
        event_id,
        tmin=ep["tmin"],
        tmax=ep["tmax"],
        baseline=tuple(ep["baseline"]),
        reject={"eeg": ep["reject_ptp_uv"] * 1e-6},
        reject_by_annotation=True,
        preload=True,
        metadata=pd.DataFrame({"subject": subject, "onset_s": original_onsets, "block": blocks, "rt": rts}),
        on_missing="ignore",
        verbose=False,
    )
    epochs.drop_bad(verbose=False)
    counts_after = {k: int(np.sum(epochs.events[:, 2] == code)) for k, code in event_id.items()}
    target_retained = counts_after["target"] / counts_correct["target"] if counts_correct["target"] else 0.0
    excluded = target_retained < pre["min_target_retained"]
    if excluded:  # keep the file for inspection, but outside the folder the team loads from
        _set_aside(out, prefix)
        (out / "_excluded").mkdir(exist_ok=True)
    epochs.save((out / "_excluded" if excluded else out) / f"{prefix}_epo.fif", overwrite=True, verbose=False)
    if not np.isfinite(epochs.get_data()).all():
        raise ValueError("Non-finite epochs")
    row.update(
        {
            "n_channels": len(raw.ch_names),
            "sfreq": epochs.info["sfreq"],
            "bad_channels": ";".join(bads),
            "n_bad_channels": len(bads),
            "n_ica": int(ica.n_components_),
            "ica_iterations": progress.step,
            "ica_final_weight_change": progress.change,
            "n_ic_removed": len(ica.exclude),
            "excluded_subject": excluded,
            "exclusion_reason": "target_retention_below_threshold" if excluded else "",
            "flags": "; ".join(flags),
            "montage": montage_source,
            "notch_applied": notch_applied,
            "visual_review": "pending",
            "source_sha256": source_hashes,
            "processing_status": "complete",
            "seconds": round(time.monotonic() - start, 1),
        }
    )
    for k in event_id:
        row.update(
            {
                f"{k}_before": counts_before[k],
                f"{k}_correct_trials": counts_correct[k],
                f"{k}_after": counts_after[k],
                f"{k}_retained": counts_after[k] / counts_correct[k] if counts_correct[k] else np.nan,
            }
        )
    artifact_paths = list(out.glob(f"{prefix}_*.fif")) + list((out / "_excluded").glob(f"{prefix}_*.fif"))
    row["artifacts"] = {p.relative_to(out).as_posix(): file_sha256(p) for p in artifact_paths}
    write_json(out / f"{prefix}_summary.json", row)
    print("DONE " + json.dumps(row), flush=True)
    return row


def _set_aside(out, prefix):
    """Move candidate FIF files only; keep current summary in the manifest directory."""
    moved = []
    for path in Path(out).glob(f"{prefix}_*.fif"):
        target = Path(out) / "_excluded" / path.name
        target.parent.mkdir(exist_ok=True)
        if target.exists():
            archive = Path(out) / "_history" / str(time.time_ns())
            archive.mkdir(parents=True)
            target.rename(archive / target.name)
        path.rename(target)
        moved.append(path.name)
    return moved


def write_log(root, config):
    out = require_candidate(root, config)
    rows = []
    for subject in config["dataset"]["subjects"]:
        prefix = _prefix(subject, config)
        path = out / f"{prefix}_summary.json"
        row = {
            "subject": f"sub-{subject}",
            "pipeline_version": PIPELINE_VERSION,
            "processing_status": "pending",
            "excluded_subject": None,
            "exclusion_reason": "",
        }
        if path.exists():
            row.update(json.loads(path.read_text(encoding="utf-8")))
        rows.append(row)
    log = pd.DataFrame(rows)
    destination = Path(root) / config["_r2"]["log"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    log.drop(columns=["artifacts", "source_sha256"], errors="ignore").to_csv(destination, index=False)
    return log


def usable_subjects(root, config):
    out = Path(root) / config["paths"]["derivatives"]
    return [
        subject
        for subject in config["dataset"]["subjects"]
        if (row := complete_summary(out, _prefix(subject, config), config)) is not None
        and row.get("excluded_subject") is False
    ]


def archive_candidate_subject(out, prefix):
    files = list(out.glob(f"{prefix}_*")) + list((out / "_excluded").glob(f"{prefix}_*"))
    files = [p for p in files if p.is_file()]
    if files:
        history = out / "_history" / str(time.time_ns())
        history.mkdir(parents=True)
        for path in files:
            relative = path.relative_to(out)
            destination = history / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            path.rename(destination)


def run(root, subjects=None, overwrite=False, download=False, review_bads=None):
    """Explicit candidate processing. Default considers all configured participants; [] means none.

    download=True fetches only each requested non-excluded subject's oddball files.
    It never deletes raw or modifies the published release/log.
    """
    root = Path(root).resolve()
    config = candidate_config(root, yaml.safe_load((root / "configs/config.yaml").read_text(encoding="utf-8")))
    if review_bads is not None:
        if any(s not in config["dataset"]["subjects"] for s in review_bads):
            raise ValueError("Manual review contains an unconfigured subject")
        for subject, channels in review_bads.items():
            if not isinstance(channels, list) or not all(isinstance(ch, str) for ch in channels):
                raise ValueError("Manual bad channels must be lists of channel names")
            existing = config["preprocessing"]["bad_channels"]["manual"].get(subject, [])
            config["preprocessing"]["bad_channels"]["manual"][subject] = sorted(set(existing + channels))
    out = require_candidate(root, config)
    requested = list(config["dataset"]["subjects"] if subjects is None else subjects)
    if not requested:
        return pd.DataFrame()
    if len(set(requested)) != len(requested) or any(s not in config["dataset"]["subjects"] for s in requested):
        raise ValueError("Subjects must be unique configured two-digit IDs")
    out.mkdir(parents=True, exist_ok=True)
    rawroot = root / config["paths"]["raw"]
    for subject in requested:
        prefix = _prefix(subject, config)
        current = complete_summary(out, prefix, config)
        if current is not None and not overwrite:
            print(f"RESUME {prefix}: configuration, implementation, source and artifact hashes checked")
            continue
        archive_candidate_subject(out, prefix)
        reason = excluded_subjects(config).get(subject)
        if reason:
            write_json(
                out / f"{prefix}_summary.json",
                {
                    "subject": f"sub-{subject}",
                    "pipeline_version": PIPELINE_VERSION,
                    "implementation_revision": IMPLEMENTATION_REVISION,
                    "implementation_sha256": IMPLEMENTATION_SHA256,
                    "config_fingerprint": config_fingerprint(config),
                    "processing_status": "complete",
                    "excluded_subject": True,
                    "exclusion_reason": reason,
                    "artifacts": {},
                },
            )
            print(f"SKIP {prefix}: {reason}")
            write_log(root, config)
            continue
        try:
            if download:
                download_subject(rawroot, subject, config["dataset"]["task"])
            # Behaviour-only exclusions do not require downloading coordinates or reading EEG.
            table_path = rawroot / f"sub-{subject}" / "eeg" / f"{prefix}_events.tsv"
            if not table_path.exists():
                raise FileNotFoundError(
                    f"{table_path}; raw was removed. Use download=True explicitly to fetch oddball files."
                )
            trials = score_trials(
                pd.read_csv(table_path, sep="\t"), config["behaviour"]["rt_min"], config["behaviour"]["rt_max"]
            )
            if (
                int(trials.loc[trials.condition == "target", "correct"].sum())
                >= config["behaviour"]["min_correct_targets"]
            ):
                if not has_recording(rawroot, subject, config["dataset"]["task"]):
                    raise FileNotFoundError(f"Incomplete oddball recording: sub-{subject}")
                ensure_coordinates(rawroot, subject, download=download)
            with threadpool_limits(limits=4):
                preprocess(root, subject, config)
        except Exception as error:
            write_json(
                out / f"{prefix}_summary.json",
                {
                    "subject": f"sub-{subject}",
                    "pipeline_version": PIPELINE_VERSION,
                    "processing_status": "failed",
                    "excluded_subject": None,
                    "exclusion_reason": "",
                    "processing_error": f"{type(error).__name__}: {error}",
                },
            )
            write_log(root, config)
            raise
        write_log(root, config)
        gc.collect()
    provenance = {
        "implementation_revision": IMPLEMENTATION_REVISION,
        "implementation_sha256": IMPLEMENTATION_SHA256,
        "pipeline_version": PIPELINE_VERSION,
        "config_fingerprint": config_fingerprint(config),
        "config": config,
        "mne_version": mne.__version__,
        "subjects_requested": requested,
        "usable_subjects": usable_subjects(root, config),
        "notes": [
            "Candidate only; not the team release",
            "ICA fit on full unlabelled recording",
            "ICLabel training band differs from 1-40 Hz input",
            "Interactive visual review pending",
        ],
    }
    write_json(out / "provenance.json", provenance)
    return write_log(root, config)
