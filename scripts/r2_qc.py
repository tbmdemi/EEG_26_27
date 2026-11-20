"""Matched raw/released PSD and coordinate-source evidence for R2."""

import json

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from r2_preprocessing import (
    CONFIG,
    ROOT,
    fetch_file,
    file_sha256,
    match_onsets,
    read_subject_montage,
    score_trials,
    write_json,
)


def compare_candidates():
    """Validate two manually reviewed candidates and quantify changes without promoting them."""
    records = []
    pre = CONFIG["preprocessing"]
    beh = CONFIG["behaviour"]
    folder = ROOT / "figures/preprocessing/v2/review"
    for sid, channel in (("10", "OI1h"), ("13", "AFF1h")):
        prefix = f"sub-{sid}_task-oddball"
        base = ROOT / CONFIG["paths"]["derivatives"]
        old = mne.read_epochs(base / f"{prefix}_epo.fif", preload=True, verbose="error")
        new = mne.read_epochs(base / "_candidate" / f"{prefix}_epo.fif", preload=True, verbose="error")
        summary = json.loads((base / "_candidate" / f"{prefix}_summary.json").read_text(encoding="utf-8"))
        data = new.get_data()
        assert data.shape[1:] == (127, round((pre["epoch"]["tmax"] - pre["epoch"]["tmin"]) * pre["resample_sfreq"]) + 1)
        assert new.info["sfreq"] == pre["resample_sfreq"]
        assert np.isfinite(data).all() and not new.info["bads"]
        assert np.max(np.ptp(data, axis=-1)) <= pre["epoch"]["reject_ptp_uv"] * 1e-6 * 1.00001
        assert np.max(np.abs(data.mean(axis=1))) < 1e-9
        baseline = (new.times >= pre["epoch"]["baseline"][0]) & (new.times <= pre["epoch"]["baseline"][1])
        assert np.max(np.abs(data[:, :, baseline].mean(axis=-1))) < 1e-10
        assert list(new.metadata.columns) == ["subject", "onset_s", "block", "rt"]
        target = new.events[:, 2] == new.event_id["target"]
        assert new.metadata.loc[target, "rt"].between(beh["rt_min"] - 1e-9, beh["rt_max"] + 1e-9).all()
        assert new.metadata.loc[~target, "rt"].isna().all()
        for condition in new.event_id:
            assert len(new[condition]) == summary[condition + "_after"]
        table = pd.read_csv(ROOT / "data/validation/ds004621" / f"sub-{sid}" / "eeg" / f"{prefix}_events.tsv", sep="\t")
        trials = score_trials(table, beh["rt_min"], beh["rt_max"])
        index = match_onsets(new.metadata.onset_s.to_numpy(), trials.onset_s.to_numpy())
        assert trials.iloc[index].correct.all()
        assert np.allclose(trials.iloc[index].rt.to_numpy(), new.metadata.rt.to_numpy(), equal_nan=True)
        for condition, code in new.event_id.items():
            assert (trials.iloc[index].condition.to_numpy()[new.events[:, 2] == code] == condition).all()
        break_code = CONFIG["dataset"]["events"]["break"].removeprefix("Stimulus/")
        breaks = np.sort(table.loc[table.event_type == break_code, "onset"].to_numpy())
        assert np.array_equal(np.searchsorted(breaks, new.metadata.onset_s, side="right"), new.metadata.block)
        assert channel in summary["bad_channels"].split(";")
        old_ptp = float(np.median(np.ptp(old.get_data(picks=[channel]), axis=-1)) * 1e6)
        new_ptp = float(np.median(np.ptp(new.get_data(picks=[channel]), axis=-1)) * 1e6)
        assert new_ptp < old_ptp
        fig, axes = plt.subplots(2, 2, figsize=(15, 8))
        for row, (label, ep) in enumerate((("Released v2", old), ("Candidate: manual interpolation", new))):
            image = axes[row, 0].imshow(
                np.ptp(ep.get_data() * 1e6, axis=-1).T, origin="lower", aspect="auto", vmin=0, vmax=150
            )
            axes[row, 0].set(title=f"{label}: {len(ep)} epochs", xlabel="Retained epoch", ylabel="Channel index")
            fig.colorbar(image, ax=axes[row, 0], label="PTP (uV)")
            for condition in ("standard", "target", "distractor"):
                axes[row, 1].plot(
                    ep.times * 1000, ep[condition].average().get_data(picks=["Pz"])[0] * 1e6, label=condition
                )
            axes[row, 1].set(xlabel="Time (ms)", ylabel="Pz (uV)", title="QC; versions are not interchangeable")
            axes[row, 1].legend()
        fig.tight_layout()
        fig.savefig(folder / f"sub-{sid}_candidate_comparison.png", dpi=160)
        plt.close(fig)
        records.append(
            {
                "subject": sid,
                "manual_channel": channel,
                "old_epochs": len(old),
                "candidate_epochs": len(new),
                "old_counts": {k: len(old[k]) for k in old.event_id},
                "candidate_counts": {k: len(new[k]) for k in new.event_id},
                "old_channel_median_ptp_uv": old_ptp,
                "candidate_channel_median_ptp_uv": new_ptp,
                "automatic_candidate_checks_passed": True,
                "promoted_to_release": False,
            }
        )
    write_json(ROOT / "results/v2_audit/candidate_comparison.json", records)
    return records


def build_review_pack():
    """Render whole-cohort amplitude/spectral panels; review decisions are recorded separately."""
    folder = ROOT / "figures/preprocessing/v2/review"
    folder.mkdir(parents=True, exist_ok=True)
    log = pd.read_csv(ROOT / "results/preproc_log.csv")
    subjects = log.loc[~log.excluded_subject, "subject"].tolist()
    records = []
    for subject in subjects:
        ep = mne.read_epochs(
            ROOT / CONFIG["paths"]["derivatives"] / f"{subject}_task-oddball_epo.fif", preload=True, verbose="error"
        )
        data = ep.get_data() * 1e6
        ptp = np.ptp(data, axis=-1)
        med = np.median(ptp, axis=0)
        freq, power = welch(data, fs=ep.info["sfreq"], nperseg=round(ep.info["sfreq"]), axis=-1)
        spectrum = power.mean(axis=0)
        for i, ch in enumerate(ep.ch_names):
            records.append(
                {
                    "subject": subject,
                    "channel": ch,
                    "median_ptp_uv": med[i],
                    "max_ptp_uv": ptp[:, i].max(),
                    "relative_median_ptp": med[i] / np.median(med),
                }
            )
        picks = np.argsort(med)[-5:][::-1]
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        axes[0].plot(freq, 10 * np.log10(np.maximum(np.median(spectrum, axis=0), 1e-30)), label="Median channel")
        for i in picks:
            axes[0].plot(freq, 10 * np.log10(np.maximum(spectrum[i], 1e-30)), label=ep.ch_names[i], alpha=0.8)
        axes[0].set(xlim=(0, 100), xlabel="Hz", ylabel="PSD (dB uV^2/Hz)", title=subject + " retained-epoch spectra")
        axes[0].legend(fontsize=7)
        for offset, i in enumerate(picks):
            axes[1].plot(ep.times * 1000, data[0, i] + offset * 100, label=ep.ch_names[i], lw=0.7)
        axes[1].set(xlabel="ms", ylabel="uV, offsets 100 uV", title="First retained epoch: high-amplitude channels")
        axes[1].legend(fontsize=7)
        axes[2].barh([ep.ch_names[i] for i in picks][::-1], med[picks][::-1])
        axes[2].set(xlabel="Median epoch PTP (uV)", title="Highest-amplitude channels")
        fig.tight_layout()
        fig.savefig(folder / f"{subject}_spectra_traces.png", dpi=130)
        plt.close(fig)
    pd.DataFrame(records).to_csv(ROOT / "results/v2_audit/channel_qc_metrics.csv", index=False)
    for start in range(0, len(subjects), 4):
        batch = subjects[start : start + 4]
        for kind in ("epoch", "spectrum"):
            fig, axes = plt.subplots(2, 2, figsize=(24, 8 if kind == "epoch" else 7))
            for ax, subject in zip(axes.flat, batch):
                path = (
                    folder.parent / f"{subject}_task-oddball_epoch_qc.png"
                    if kind == "epoch"
                    else folder / f"{subject}_spectra_traces.png"
                )
                ax.imshow(plt.imread(path))
                ax.axis("off")
            for ax in list(axes.flat)[len(batch) :]:
                ax.axis("off")
            fig.tight_layout()
            fig.savefig(folder / f"{kind}_gallery_{start // 4 + 1:02}.png", dpi=125)
            plt.close(fig)
    return pd.DataFrame(records)


from scipy.signal import welch


def montage_evidence():
    records = []
    for sid in ("12", "24"):
        relative = f"sourcedata/sub-{sid}/coords/sub-{sid}_captrak.bvef"
        fixture = fetch_file(relative, ROOT / "data/validation/ds004621")
        montage, description = read_subject_montage(fixture)
        epochs = mne.read_epochs(
            ROOT / CONFIG["paths"]["derivatives"] / f"sub-{sid}_task-oddball_epo.fif",
            preload=False,
            verbose="error",
        )
        error = max(
            np.linalg.norm(montage.get_positions()["ch_pos"][ch] - epochs.info["chs"][i]["loc"][:3])
            for i, ch in enumerate(epochs.ch_names)
        )
        assert error < 1e-8
        records.append(
            {
                "subject": sid,
                "source_url": "https://s3.amazonaws.com/openneuro.org/ds004621/" + relative,
                "sha256": file_sha256(fixture),
                "reader": description,
                "max_position_error_m": error,
            }
        )
    write_json(ROOT / "results/v2_audit/montage_sources.json", records)
    return records


def matched_raw_psd(sid="13", n_epochs=50):
    """Compare identical retained event windows, not unrelated raw/clean segments."""
    folder = ROOT / "figures/preprocessing/v2/review"
    folder.mkdir(parents=True, exist_ok=True)
    prefix = f"sub-{sid}_task-oddball"
    header = ROOT / CONFIG["paths"]["raw"] / f"sub-{sid}" / "eeg" / f"{prefix}_eeg.vhdr"
    raw = mne.io.read_raw_brainvision(header, preload=False, verbose="error")
    raw.rename_channels({k: v for k, v in CONFIG["preprocessing"]["channel_aliases"].items() if k in raw.ch_names})
    epochs = mne.read_epochs(ROOT / CONFIG["paths"]["derivatives"] / f"{prefix}_epo.fif", preload=True, verbose="error")
    n = min(n_epochs, len(epochs))
    picks = [raw.ch_names.index(ch) for ch in epochs.ch_names]
    before = []
    for onset in epochs.metadata.onset_s.iloc[:n]:
        start = round((onset + epochs.tmin) * raw.info["sfreq"])
        count = round((epochs.tmax - epochs.tmin) * raw.info["sfreq"]) + 1
        data = raw.get_data(picks=picks, start=start, stop=start + count) * 1e6
        fb, power = welch(data, fs=raw.info["sfreq"], nperseg=round(raw.info["sfreq"]), axis=-1)
        before.append(power)
    fc, power = welch(
        epochs.get_data()[:n] * 1e6, fs=epochs.info["sfreq"], nperseg=round(epochs.info["sfreq"]), axis=-1
    )
    pb, pa = np.mean(before, axis=0), power.mean(axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    channel = "AFF1h" if sid == "13" else "Pz"
    for ax, index, label in ((axes[0], None, "Median channel"), (axes[1], epochs.ch_names.index(channel), channel)):
        b = np.median(pb, axis=0) if index is None else pb[index]
        a = np.median(pa, axis=0) if index is None else pa[index]
        ax.plot(fb, 10 * np.log10(np.maximum(b, 1e-30)), label="Raw, matched retained windows")
        ax.plot(fc, 10 * np.log10(np.maximum(a, 1e-30)), label="Released cleaned epochs")
        ax.set(xlim=(0, 100), xlabel="Frequency (Hz)", ylabel="PSD (dB uV^2/Hz)", title=label)
        ax.legend(fontsize=8)
    fig.suptitle(f"{prefix}: same first {n} retained windows; Welch 1 Hz bins")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(folder / f"{prefix}_matched_raw_clean_psd.{extension}", dpi=300)
    plt.close(fig)
    ica = mne.preprocessing.read_ica(ROOT / CONFIG["paths"]["derivatives"] / f"{prefix}_ica.fif", verbose="error")
    figures = ica.plot_components(picks=ica.exclude, show=False)
    if not isinstance(figures, list):
        figures = [figures]
    for i, figure in enumerate(figures):
        for extension in ("png", "pdf"):
            figure.savefig(folder / f"{prefix}_excluded_ics_{i}.{extension}", dpi=300)
        plt.close(figure)
    record = {
        "subject": sid,
        "n_matched_windows": n,
        "raw_eeg_sha256": file_sha256(header.with_suffix(".eeg")),
        "excluded_ica_indices": ica.exclude,
        "interpretation": "Before/after example, not proof that all artifacts were removed",
    }
    write_json(ROOT / "results/v2_audit/raw_psd_example.json", record)
    return record
