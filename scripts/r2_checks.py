"""Synthetic integration checks; pybv is required for fixture export only."""

import copy
import json
import tempfile
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import yaml
from mne_bids import BIDSPath
from r2_preprocessing import (
    IMPLEMENTATION_REVISION,
    IMPLEMENTATION_SHA256,
    ROOT,
    _discontinuities,
    candidate_config,
    complete_summary,
    file_sha256,
    match_onsets,
    read_bids_readonly,
    read_subject_montage,
    require_candidate,
    run,
    score_trials,
    usable_subjects,
    write_json,
)


def test_r2():
    """Real synthetic BrainVision -> BIDS -> filter/ICA/ICLabel -> FIF, with isolation checks.

    Requires pybv for test-input export only. No real EEG or external coordinate fixture is needed.
    """
    from mne_bids import write_raw_bids

    original_config = yaml.safe_load((ROOT / "configs/config.yaml").read_text(encoding="utf-8"))
    # Inclusive RT boundaries, empty data, shuffled trial-table order and onset matching.
    table = pd.DataFrame(
        {
            "onset": [0, 0.1, 3, 4.4, 6, 7.40001, 9, 9.09999],
            "event_type": ["S  6", "S  1", "S  5", "S  1", "S  7", "S  1", "S  6", "S  1"],
        }
    )
    assert score_trials(table, 0.1, 1.4).correct.tolist() == [True, False, True, False]
    assert score_trials(table.iloc[::-1], 0.1, 1.4).correct.tolist() == [True, False, True, False]
    assert list(score_trials(table.iloc[:0], 0.1, 1.4).columns) == ["onset_s", "condition", "correct", "rt"]
    assert match_onsets(np.array([]), np.array([])).size == 0
    assert match_onsets(np.array([1.0]), np.array([1.0])).tolist() == [0]
    try:
        match_onsets(np.array([1.0]), np.array([]))
    except ValueError:
        pass
    else:
        raise AssertionError("Empty source onsets must fail")
    result = {"boundary_and_onset_tests": True}
    with tempfile.TemporaryDirectory(prefix="r2-synthetic-") as directory:
        testroot = Path(directory)
        (testroot / "configs").mkdir()
        cfg = copy.deepcopy(original_config)
        cfg["dataset"]["subjects"] = ["01", "02", "03", "04"]
        # Small synthetic study only: real config thresholds remain untouched.
        cfg["behaviour"]["min_correct_targets"] = 2
        (testroot / "configs/config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
        rawroot = testroot / cfg["paths"]["raw"]
        coord = rawroot / "sourcedata/sub-01/coords/sub-01_captrak.bvef"
        coord.parent.mkdir(parents=True)
        # Fully synthetic coordinate fixture: no untracked release or network is required.
        import inspect
        import xml.etree.ElementTree as XML

        standard = mne.channels.make_standard_montage("standard_1005")
        required = ["Fp1", "C3", "Cz", "Pz", "O1"]
        names = required + [name for name in standard.ch_names if name not in required][:122]
        positions = np.array([standard.get_positions()["ch_pos"][name] for name in names])
        radii = np.linalg.norm(positions, axis=1)
        xmlroot = XML.Element("Electrodes")
        for name, position, radius in zip(names, positions, radii):
            electrode = XML.SubElement(xmlroot, "Electrode")
            for key, value in {
                "Name": name,
                "Theta": np.degrees(np.arccos(position[2] / radius)),
                "Phi": np.degrees(np.arctan2(position[1], position[0])),
                "Radius": radius * 1000,
            }.items():
                XML.SubElement(electrode, key).text = str(value)
        XML.ElementTree(xmlroot).write(coord, encoding="utf-8", xml_declaration=True)
        montage, _ = read_subject_montage(coord)
        head_size = inspect.signature(mne.channels.read_custom_montage).parameters["head_size"].default
        expected_positions = positions * head_size / np.median(radii)
        assert (
            max(
                np.linalg.norm(montage.get_positions()["ch_pos"][name] - expected_positions[i])
                for i, name in enumerate(names)
            )
            < 1e-10
        )
        rng = np.random.default_rng(42)
        samples = 30000
        sources = rng.laplace(size=(3, samples))
        signal = (rng.normal(size=(127, 3)) @ sources + rng.normal(scale=0.01, size=(127, samples))) * 1e-6
        raw = mne.io.RawArray(signal, mne.create_info(names, 1000, "eeg"), verbose="error")
        raw.set_montage(montage)
        raw.info["line_freq"] = 50
        raw.set_meas_date(None)
        bids = BIDSPath(root=rawroot, subject="01", task="oddball", datatype="eeg")
        write_raw_bids(raw, bids, format="BrainVision", allow_preload=True, overwrite=True, verbose="error")
        rows = []
        for i, onset in enumerate(np.arange(1.0, 28.0, 2.0)):
            event = ["S  5", "S  6", "S  7"][i % 3]
            rows.append({"onset": onset, "event_type": event, "duration": 0, "trial_type": "stimulus"})
            if event == "S  6":
                rows.append({"onset": onset + 0.4, "event_type": "S  1", "duration": 0, "trial_type": "response"})
        for onset in [10.0, 20.0]:
            rows.append({"onset": onset, "event_type": "S 10", "duration": 0, "trial_type": "break"})
        events = pd.DataFrame(rows).sort_values("onset")
        eeg = rawroot / "sub-01/eeg"
        events.to_csv(eeg / "sub-01_task-oddball_events.tsv", sep="\t", index=False)
        # A true acquisition join overlaps one target epoch at t=21 s.
        vmrk = eeg / "sub-01_task-oddball_eeg.vmrk"
        with vmrk.open("a", encoding="utf-8") as stream:
            stream.write("\nMk2=New Segment,,21001,1,0,20200101000000000000\n")
        assert _discontinuities(vmrk).size == 1 and np.allclose(_discontinuities(vmrk), [21.0])
        channels = eeg / "sub-01_task-oddball_channels.tsv"
        shuffled = pd.read_csv(channels, sep="\t")
        shuffled = shuffled.iloc[[1, 0] + list(range(2, len(shuffled)))].copy()
        shuffled["units"] = chr(181) + "V"
        text = shuffled.to_csv(sep="\t", index=False)
        channels.write_bytes(text.encode("latin-1"))
        assert bytes([181]) in channels.read_bytes()
        source_before = {str(f.relative_to(rawroot)): file_sha256(f) for f in rawroot.rglob("*") if f.is_file()}
        loaded = read_bids_readonly(testroot, rawroot, "01", "oddball", cfg["preprocessing"]["channel_aliases"])
        assert loaded.ch_names == names
        assert np.allclose(loaded.get_data(), signal, atol=1e-12, rtol=1e-5)
        del loaded

        # The writer will create its own candidate tree, never a published output.
        assert run(testroot, subjects=[], review_bads={"01": ["Fp1"]}).empty
        try:
            run(testroot, subjects=[], review_bads={"99": ["Fp1"]})
        except ValueError:
            pass
        else:
            raise AssertionError("Unconfigured manual-review subject must fail")
        assert run(testroot, subjects=[]).empty
        assert not (testroot / cfg["paths"]["derivatives"] / "_candidate").exists()
        log = run(testroot, subjects=["01", "02"])
        ccfg = candidate_config(testroot, cfg)
        out = testroot / ccfg["paths"]["derivatives"]
        summary = complete_summary(out, "sub-01_task-oddball", ccfg)
        assert summary is not None and not summary["excluded_subject"]
        assert summary["target_after"] == summary["target_correct_trials"] - 1
        assert log.set_index("subject").loc["sub-02", "excluded_subject"] == True
        assert log.set_index("subject").loc["sub-03", "processing_status"] == "pending"
        epo = out / "sub-01_task-oddball_epo.fif"
        saved = mne.read_epochs(epo, preload=True, verbose="error")
        assert saved.info["sfreq"] == cfg["preprocessing"]["resample_sfreq"]
        assert saved.get_data().shape[1:] == (127, 251)
        assert list(saved.metadata.columns) == ["subject", "onset_s", "block", "rt"]
        assert np.isfinite(saved.get_data()).all()
        assert not ((saved.metadata.onset_s - 0.2 <= 21.5) & (saved.metadata.onset_s + 0.8 >= 20.5)).any()
        assert source_before == {str(f.relative_to(rawroot)): file_sha256(f) for f in rawroot.rglob("*") if f.is_file()}
        assert not (testroot / "results/preproc_log.csv").exists()
        before = file_sha256(epo)
        run(testroot, subjects=["01"])
        assert file_sha256(epo) == before
        assert usable_subjects(testroot, ccfg) == ["01"]
        assert complete_summary(out, "sub-01_task-oddball", ccfg) is not None
        changed = copy.deepcopy(ccfg)
        changed["preprocessing"]["l_freq"] = 0.2
        assert complete_summary(out, "sub-01_task-oddball", changed) is None
        with epo.open("ab") as stream:
            stream.write(b"changed")
        assert complete_summary(out, "sub-01_task-oddball", ccfg) is None
        # A retention failure moves all candidate FIFs outside the usable top-level directory.
        stricter = copy.deepcopy(cfg)
        stricter["preprocessing"]["min_target_retained"] = 0.9
        (testroot / "configs/config.yaml").write_text(yaml.safe_dump(stricter), encoding="utf-8")
        run(testroot, subjects=["01"])
        retention = json.loads((out / "sub-01_task-oddball_summary.json").read_text())
        assert retention["excluded_subject"] and retention["exclusion_reason"] == "target_retention_below_threshold"
        assert not epo.exists() and (out / "_excluded" / epo.name).exists()
        assert (out / "_excluded/sub-01_task-oddball_ica.fif").exists()
        assert complete_summary(out, "sub-01_task-oddball", candidate_config(testroot, stricter)) is not None
        # Too few correct targets is a scientific exclusion, without loading EEG.
        small = rawroot / "sub-03/eeg"
        small.mkdir(parents=True)
        pd.DataFrame({"onset": [1.0], "event_type": ["S  6"]}).to_csv(
            small / "sub-03_task-oddball_events.tsv", sep="\t", index=False
        )
        run(testroot, subjects=["03"])
        excluded = json.loads((out / "sub-03_task-oddball_summary.json").read_text())
        assert excluded["excluded_subject"] and excluded["exclusion_reason"] == "correct_targets=0"
        # Missing files are processing failures, never scientific exclusions or resumable results.
        try:
            run(testroot, subjects=["04"])
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("Missing raw should fail explicitly")
        failed = json.loads((out / "sub-04_task-oddball_summary.json").read_text())
        assert failed["processing_status"] == "failed" and failed["excluded_subject"] is None
        try:
            require_candidate(testroot, cfg)
        except ValueError:
            pass
        else:
            raise AssertionError("Published output path must be rejected")
        result.update(
            real_brainvision_bids_pipeline=True,
            ica_and_iclabel_executed=True,
            readonly_latin1_input=True,
            bvef_geometry_roundtrip=True,
            bad_join_rejection=True,
            config_and_artifact_resume=True,
            retention_exclusion_paths=True,
            excluded_and_failed_statuses=True,
            published_output_protected=True,
        )
    evidence = ROOT / "results/v2_audit/r2_code_tests.json"
    write_json(
        evidence,
        {
            "implementation_revision": IMPLEMENTATION_REVISION,
            "implementation_sha256": IMPLEMENTATION_SHA256,
            "synthetic_test_config_only": True,
            **result,
        },
    )
    print("PASS R2 synthetic integration:", result)
    return result
