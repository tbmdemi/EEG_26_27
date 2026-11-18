"""Small numerical and input-validation regressions for R3 (standard unittest)."""

import copy
import unittest
from unittest.mock import patch

import mne
import numpy as np
import pandas as pd

from eegproj.config import find_root, load_config
from eegproj.evaluation.behaviour_summary import summarise_behaviour
from eegproj.evaluation.erp import _usable_release_log, collect_subject_erps, summarise_erps
from eegproj.evaluation.p300 import measure_candidate_peaks
from eegproj.evaluation.topography import summarise_scalp_maps


class ReleaseValidationTests(unittest.TestCase):
    def setUp(self):
        self.root = find_root()
        self.config = copy.deepcopy(load_config(self.root))
        self.config["dataset"]["subjects"] = ["01", "02", "03"]
        self.config["dataset"]["exclude"] = {"02": "a-priori exclusion"}
        self.handoff = pd.DataFrame(
            {
                "subject": ["sub-01", "sub-02", "sub-03"],
                "pipeline_version": [2, 2, 2],
                "excluded_subject": [False, True, False],
            }
        )

    def test_complete_release_preserves_subject_order(self):
        release = self.handoff.iloc[::-1]
        with patch("eegproj.evaluation.erp.pd.read_csv", side_effect=[release, self.handoff]):
            usable = _usable_release_log(self.root, self.config)
        self.assertEqual(usable.subject.tolist(), ["sub-03", "sub-01"])

    def test_incomplete_old_or_different_cohort_is_rejected(self):
        variants = [
            self.handoff.iloc[:-1],
            self.handoff.assign(pipeline_version=[2, 2, 1]),
            self.handoff.assign(excluded_subject=[False, True, True]),
            self.handoff.assign(excluded_subject=[False, False, False]),
        ]
        for release in variants:
            with (
                self.subTest(release=release.to_dict("list")),
                patch("eegproj.evaluation.erp.pd.read_csv", side_effect=[release, self.handoff]),
                self.assertRaises(ValueError),
            ):
                _usable_release_log(self.root, self.config)


class MetadataValidationTests(unittest.TestCase):
    def setUp(self):
        self.root = find_root()
        self.config = load_config(self.root)
        self.log = pd.DataFrame(
            {
                "subject": ["sub-01"],
                "n_channels": [1],
                "standard_after": [1],
                "target_after": [1],
                "distractor_after": [1],
                "visual_review": ["pending"],
            }
        )
        self.metadata = pd.DataFrame(
            {"subject": ["01"] * 3, "onset_s": [1.0, 2.0, 3.0], "block": [0, 0, 0], "rt": [np.nan, 0.5, np.nan]}
        )

    def load(self, metadata):
        pre = self.config["preprocessing"]
        samples = round((pre["epoch"]["tmax"] - pre["epoch"]["tmin"]) * pre["resample_sfreq"]) + 1
        epochs = mne.EpochsArray(
            np.zeros((3, 1, samples)),
            mne.create_info(["Pz"], pre["resample_sfreq"], "eeg"),
            events=np.array([[100, 0, 1], [400, 0, 2], [700, 0, 3]]),
            event_id={"standard": 1, "target": 2, "distractor": 3},
            tmin=pre["epoch"]["tmin"],
            baseline=tuple(pre["epoch"]["baseline"]),
            metadata=metadata,
            verbose=False,
        )
        with (
            patch("eegproj.evaluation.erp._usable_release_log", return_value=self.log),
            patch("eegproj.evaluation.erp.mne.read_epochs", return_value=epochs),
        ):
            return collect_subject_erps(self.root, self.config, ["Pz"])

    def test_correct_metadata_and_channel_order_are_accepted(self):
        subjects, _, erps, counts = self.load(self.metadata)
        self.assertEqual(subjects, ["sub-01"])
        self.assertEqual(erps["target"].shape[0:2], (1, 1))
        self.assertEqual(counts.loc[0, "target"], 1)

    def test_wrong_subject_or_behavioral_rt_is_rejected(self):
        for metadata in [
            self.metadata.assign(subject="03"),
            self.metadata.assign(rt=[np.nan, np.nan, np.nan]),
            self.metadata.assign(rt=[0.3, 0.5, np.nan]),
            self.metadata.assign(onset_s=[2.0, 1.0, 3.0]),
        ]:
            with self.subTest(metadata=metadata.to_dict("list")), self.assertRaises(ValueError):
                self.load(metadata)


class NumericalTests(unittest.TestCase):
    def test_subject_sem_and_paired_difference_units(self):
        standard = np.array([[[1.0, 1.0]], [[3.0, 3.0]]]) * 1e-6
        waves = {"standard": standard, "target": standard + 1e-6, "distractor": standard.copy()}
        summary = summarise_erps(waves, np.array([0.0, 0.004]), ["Pz"])
        np.testing.assert_allclose(summary.loc[summary.condition.eq("standard"), "mean_uv"], 2.0)
        np.testing.assert_allclose(summary.loc[summary.condition.eq("standard"), "sem_uv"], 1.0)
        np.testing.assert_allclose(summary.loc[summary.condition.eq("target_minus_standard"), "mean_uv"], 1.0)
        np.testing.assert_allclose(
            summary.loc[summary.condition.eq("target_minus_standard"), "sem_uv"], 0.0, atol=1e-14
        )

    def test_topomap_window_and_paired_sem(self):
        standard = np.array([[[1.0, 1.0, 1.0]], [[3.0, 3.0, 3.0]]]) * 1e-6
        result = {
            "subjects": ["a", "b"],
            "channels": ["Pz"],
            "times": np.array([0.0, 0.4, 0.8]),
            "erps": {"standard": standard, "target": standard + 1e-6, "distractor": standard.copy()},
        }
        window, snapshots = summarise_scalp_maps(result, [0.25, 0.6], [0.4])
        self.assertTrue(window.n_samples.eq(1).all())
        np.testing.assert_allclose(window.first_sample_s, 0.4)
        np.testing.assert_allclose(window.loc[window.condition.eq("standard"), "mean_uv"], 2.0)
        np.testing.assert_allclose(
            snapshots.loc[snapshots.condition.eq("target_minus_standard"), "sem_uv"], 0.0, atol=1e-14
        )

    def test_signed_peak_tie_and_boundary_flags(self):
        config = load_config(find_root())
        times = np.arange(201) / config["preprocessing"]["resample_sfreq"]
        data = np.zeros((3, 1, len(times)))
        index = lambda time: int(np.argmin(abs(times - time)))
        data[0, 0, index(0.4)] = -100e-6
        data[0, 0, index(0.5)] = 2e-6
        data[1, 0] = times * 1e-6
        data[2, 0, index(0.448) : index(0.448) + 2] = 3e-6
        subjects = ["signed", "boundary", "tie"]
        counts = pd.DataFrame({"subject": subjects, "target": 10, "distractor": 10, "visual_review": "pending"})
        rows = measure_candidate_peaks(subjects, times, {"target": data, "distractor": data}, counts, ["Pz"], config)
        target = rows.loc[rows.condition.eq("target")].set_index("subject")
        self.assertAlmostEqual(target.loc["signed", "candidate_peak_uv"], 2.0)
        self.assertAlmostEqual(target.loc["signed", "candidate_latency_ms"], 500.0)
        self.assertTrue(target.loc["boundary", "boundary_maximum"])
        self.assertIn("tied_maximum", target.loc["tie", "peak_flags"])
        self.assertAlmostEqual(target.loc["tie", "candidate_latency_ms"], 448.0)

    def test_behavior_equal_subject_weights_and_rt_sd(self):
        subjects = pd.DataFrame(
            {
                "subject": ["a", "b"],
                "in_erp_cohort": [True, True],
                "rt_mean_ms": [100.0, 300.0],
                "rt_sd_ms": [1.0, 1.0],
                "hit_rate_pct": [50.0, 100.0],
                "target_omission_pct": [50.0, 0.0],
                "false_alarm_distractor_pct": [0.0, 20.0],
                "false_alarm_standard_pct": [0.0, 10.0],
            }
        )
        summary = summarise_behaviour(subjects, load_config(find_root()))
        rows = summary.loc[summary.cohort.eq("before_eeg_qc")].set_index("metric")
        self.assertAlmostEqual(rows.loc["hit_rate_pct", "mean"], 75.0)
        self.assertAlmostEqual(rows.loc["rt_mean_ms", "sd"], np.sqrt(20000.0))


if __name__ == "__main__":
    unittest.main()
