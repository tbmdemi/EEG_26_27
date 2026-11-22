"""R2 release verification and QC plots, independent of raw reprocessing."""

import gc
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import yaml
from r2_preprocessing import ROOT, fetch_file, match_onsets, score_trials, summarise

QC_COLUMNS = {"visual_review", "visual_review_by", "visual_review_date", "visual_review_scope", "qc_findings"}


def compare_release_logs(released, tracked):
    """Human QC annotations may evolve; scientific release fields must remain unchanged."""
    original = released.drop(columns=list(QC_COLUMNS), errors="ignore")
    current = tracked.drop(columns=list(QC_COLUMNS), errors="ignore")
    pd.testing.assert_frame_equal(original, current)


def audit_release(root=ROOT):
    ROOT = Path(root)
    CONFIG = yaml.safe_load((ROOT / "configs/config.yaml").read_text(encoding="utf-8"))
    OUT = ROOT / CONFIG["paths"]["derivatives"]
    LOG = pd.read_csv(OUT / "preproc_log.csv")
    RELEASE_CONFIG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
    assert LOG.subject.is_unique
    assert set(LOG.subject) == {f"sub-{s}" for s in CONFIG["dataset"]["subjects"]}
    # Tải events.tsv gốc của file nào còn thiếu (khoảng 20 KB mỗi file), để kiểm tra đủ 42 người trên mọi máy.
    for subject in LOG.subject:
        fetch_file(f"{subject}/eeg/{subject}_task-oddball_events.tsv", ROOT / "data/validation/ds004621")
    assert (LOG.pipeline_version == 2).all()
    assert LOG.excluded_subject.notna().all()
    assert set(LOG.excluded_subject.unique()) <= {True, False}
    USABLE = LOG.loc[~LOG.excluded_subject].copy()
    EXCLUDED = LOG.loc[LOG.excluded_subject].copy()
    for s in CONFIG["dataset"]["exclude"]:
        assert bool(LOG.set_index("subject").loc[f"sub-{s}", "excluded_subject"])
    assert EXCLUDED.exclusion_reason.notna().all()
    assert len(USABLE) == 35
    print("Considered:", len(LOG), "Usable:", len(USABLE), "Excluded:", len(EXCLUDED))
    print(EXCLUDED[["subject", "exclusion_reason"]].to_string(index=False))

    # Compare numerical/method parameters with the immutable release's flat config.
    for key in ["seed", "behaviour"]:
        assert RELEASE_CONFIG[key] == CONFIG[key], key
    for key in [
        "channel_aliases",
        "bad_channels",
        "l_freq",
        "h_freq",
        "resample_sfreq",
        "discontinuity_pad_s",
        "ica",
        "epoch",
        "min_target_retained",
        "qc",
    ]:
        if key == "bad_channels":
            # Manual QC decisions are for future candidates; retain the released numeric detector thresholds.
            assert {k: v for k, v in RELEASE_CONFIG[key].items() if k != "manual"} == {
                k: v for k, v in CONFIG["preprocessing"][key].items() if k != "manual"
            }, key
        else:
            assert RELEASE_CONFIG[key] == CONFIG["preprocessing"][key], key
    assert RELEASE_CONFIG["exclude"] == CONFIG["dataset"]["exclude"]
    assert RELEASE_CONFIG["events"] == {k: CONFIG["dataset"]["events"][k] for k in ["standard", "target", "distractor"]}
    assert RELEASE_CONFIG["task"] == CONFIG["dataset"]["task"]
    # Per-subject counts are the release source of truth; the Git log must match too.
    compare_release_logs(LOG, pd.read_csv(ROOT / "results/preproc_log.csv"))

    AUDIT_DIR = ROOT / "results" / "v2_audit"
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    pre = CONFIG["preprocessing"]
    checks = []
    for row in USABLE.to_dict("records"):
        subject = row["subject"]
        sid = subject.removeprefix("sub-")
        prefix = f"{subject}_task-{CONFIG['dataset']['task']}"
        path = OUT / f"{prefix}_epo.fif"
        epochs = mne.read_epochs(path, preload=True, verbose="error")
        data = epochs.get_data()
        meta = epochs.metadata
        assert len(epochs.ch_names) == 127 and not epochs.info["bads"], subject
        assert all(np.isfinite(ch["loc"][:3]).all() for ch in epochs.info["chs"]), subject
        assert np.max(np.abs(data.mean(axis=1))) < 1e-9, subject
        assert epochs.info["sfreq"] == pre["resample_sfreq"], subject
        assert epochs.event_id == {"standard": 1, "target": 2, "distractor": 3}, subject
        assert np.allclose([epochs.tmin, epochs.tmax], [pre["epoch"]["tmin"], pre["epoch"]["tmax"]]), subject
        assert data.shape[-1] == 251 and np.isfinite(data).all(), subject
        assert epochs.baseline is not None and np.allclose(epochs.baseline, pre["epoch"]["baseline"]), subject
        peak_uv = float(np.ptp(data, axis=-1).max() * 1e6)
        assert peak_uv <= pre["epoch"]["reject_ptp_uv"] * 1.00001, subject
        base = (epochs.times >= pre["epoch"]["baseline"][0]) & (epochs.times <= pre["epoch"]["baseline"][1])
        baseline_error = float(np.abs(data[:, :, base].mean(axis=-1)).max())
        assert baseline_error < 1e-10, subject
        assert list(meta.columns) == ["subject", "onset_s", "block", "rt"], subject
        assert set(meta.subject.astype(str).str.replace("sub-", "", regex=False).str.zfill(2)) == {sid}, subject
        assert meta.onset_s.is_monotonic_increasing and meta.onset_s.is_unique, subject
        assert meta.onset_s.notna().all() and np.isfinite(meta.onset_s).all(), subject
        assert meta.block.notna().all() and set(meta.block) <= set(range(CONFIG["evaluation"]["within"]["n_splits"])), (
            subject
        )
        target = epochs.events[:, 2] == epochs.event_id["target"]
        assert meta.loc[~target, "rt"].isna().all(), subject
        assert (
            meta.loc[target, "rt"]
            .between(CONFIG["behaviour"]["rt_min"] - 1e-9, CONFIG["behaviour"]["rt_max"] + 1e-9)
            .all()
        ), subject
        counts = {k: len(epochs[k]) for k in epochs.event_id}
        for k, count in counts.items():
            assert count == int(row[f"{k}_after"]), (subject, k, count, row[f"{k}_after"])
            assert count <= row[f"{k}_correct_trials"] <= row[f"{k}_before"], subject
        assert row["target_correct"] >= CONFIG["behaviour"]["min_correct_targets"], subject
        assert counts["target"] / row["target_correct_trials"] >= pre["min_target_retained"], subject
        summary = json.loads((OUT / f"{prefix}_summary.json").read_text(encoding="utf-8"))
        assert summary["pipeline_version"] == 2 and not summary["excluded_subject"], subject
        for k, count in counts.items():
            assert summary[f"{k}_after"] == count, subject
        raw_status = "not_checked_raw_events_absent"
        events_path = ROOT / CONFIG["paths"]["raw"] / subject / "eeg" / f"{prefix}_events.tsv"
        validation_path = ROOT / "data/validation/ds004621" / subject / "eeg" / f"{prefix}_events.tsv"
        if validation_path.exists():
            events_path = validation_path
        if events_path.exists():
            table = pd.read_csv(events_path, sep="\t")
            trials = score_trials(table, CONFIG["behaviour"]["rt_min"], CONFIG["behaviour"]["rt_max"])
            index = match_onsets(meta.onset_s.to_numpy(), trials.onset_s.to_numpy())
            matched = trials.iloc[index]
            assert matched.correct.all(), subject
            expected = np.array([{1: "standard", 2: "target", 3: "distractor"}[c] for c in epochs.events[:, 2]])
            assert (matched.condition.to_numpy() == expected).all(), subject
            assert np.allclose(matched.rt.to_numpy(), meta.rt.to_numpy(), equal_nan=True, atol=1e-9), subject
            breaks = table.loc[table.event_type == "S 10", "onset"].to_numpy()
            assert np.array_equal(np.searchsorted(breaks, meta.onset_s, side="right"), meta.block), subject
            behaviour = summarise(trials)
            for k in ["n_stimuli", "standard_correct", "target_correct", "distractor_correct"]:
                assert behaviour[k] == row[k], (subject, k)
            raw_status = "matched_source_events" if events_path == validation_path else "matched_existing_raw_events"
        checks.append(
            {
                "subject": subject,
                "epochs": len(epochs),
                "counts": counts,
                "channels": len(epochs.ch_names),
                "samples": data.shape[-1],
                "max_ptp_uv": peak_uv,
                "max_baseline_error_v": baseline_error,
                "blocks": sorted(meta.block.unique().tolist()),
                "raw_behaviour_check": raw_status,
                "automatic_checks_passed": True,
                "visual_review": "not_assessed_by_automatic_checks",
            }
        )
        print(subject, len(epochs), raw_status, flush=True)
        del data, epochs
        gc.collect()
    (AUDIT_DIR / "verification.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print("Verified", len(checks), "subjects;", sum(r["epochs"] for r in checks), "epochs")

    behaviour_checks = []
    for row in LOG.to_dict("records"):
        subject = row["subject"]
        path = ROOT / "data/validation/ds004621" / subject / "eeg" / f"{subject}_task-oddball_events.tsv"
        table = pd.read_csv(path, sep="\t")
        stats = summarise(score_trials(table, CONFIG["behaviour"]["rt_min"], CONFIG["behaviour"]["rt_max"]))
        for key, value in stats.items():
            if key in row and pd.notna(row[key]):
                assert np.isclose(value, row[key], equal_nan=True, atol=1e-6), (subject, key, value, row[key])
        if subject.removeprefix("sub-") not in CONFIG["dataset"]["exclude"]:
            too_few = stats["target_correct"] < CONFIG["behaviour"]["min_correct_targets"]
            low_retention = row["target_retained"] < CONFIG["preprocessing"]["min_target_retained"]
            assert bool(row["excluded_subject"]) == (too_few or low_retention), subject
        behaviour_checks.append(dict(subject=subject, **stats, excluded=bool(row["excluded_subject"]), verified=True))
    (AUDIT_DIR / "behaviour_42.json").write_text(json.dumps(behaviour_checks, indent=2), encoding="utf-8")
    print("Behaviour/exclusion checks passed for", len(behaviour_checks), "participants")

    return {"verification": checks, "behaviour": behaviour_checks}


def make_epoch_qc(root=ROOT):
    ROOT = Path(root)
    CONFIG = yaml.safe_load((ROOT / "configs/config.yaml").read_text(encoding="utf-8"))
    LOG = pd.read_csv(ROOT / CONFIG["paths"]["derivatives"] / "preproc_log.csv")
    USABLE = LOG.loc[~LOG.excluded_subject]
    # Kiểm tra epoch trên toàn bộ thời gian thí nghiệm, không chỉ đoạn đầu recording.
    qc_dir = ROOT / "figures/preprocessing/v2"
    qc_dir.mkdir(parents=True, exist_ok=True)
    for subject in [s.removeprefix("sub-") for s in USABLE.subject]:
        prefix = f"sub-{subject}_task-oddball"
        epochs = mne.read_epochs(
            ROOT / CONFIG["paths"]["derivatives"] / f"{prefix}_epo.fif", preload=True, verbose=False
        )
        amplitudes = np.ptp(epochs.get_data(), axis=-1) * 1e6
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        image = axes[0].imshow(
            amplitudes.T, origin="lower", aspect="auto", vmin=0, vmax=CONFIG["preprocessing"]["epoch"]["reject_ptp_uv"]
        )
        axes[0].set(xlabel="Retained trial index", ylabel="EEG channel index", title=f"{prefix}: peak-to-peak")
        fig.colorbar(image, ax=axes[0], label="Amplitude (uV)")
        for condition in ["standard", "target", "distractor"]:
            if len(epochs[condition]):
                evoked = epochs[condition].average()
                axes[1].plot(evoked.times * 1000, evoked.get_data(picks=["Pz"])[0] * 1e6, label=condition)
        axes[1].axvline(0, color="gray", linewidth=0.6)
        axes[1].set(xlabel="Time (ms)", ylabel="Amplitude at Pz (uV)", title="ERP quality check (exploratory)")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(qc_dir / f"{prefix}_epoch_qc.png", dpi=CONFIG["preprocessing"]["qc"]["dpi"])
        plt.close(fig)
        del epochs

    return sorted(qc_dir.glob("*_epoch_qc.png"))
