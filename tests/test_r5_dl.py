"""Tests for the R5 deep-learning code on small synthetic data in the v2 epoch format.

python -m unittest tests.test_r5_dl -v
"""

import copy
import shutil
import tempfile
import unittest
import warnings
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import torch
import yaml

from eegproj.config import find_root, load_config
from eegproj.evaluation.folds import cross_subject_folds, folds_csv_path, within_cv_metadata
from eegproj.evaluation.metrics import REQUIRED_COLUMNS, validate_metrics_df
from eegproj.evaluation.splits import iter_within_folds
from eegproj.models.dl import NETWORKS, TorchClassifier, channel_stats, load_checkpoint, make_dl
from eegproj.models.dl_runner import (
    _TrialTable,
    align_channels,
    cross_validation_subjects,
    load_dl_subject,
    run_dl_grid,
    summarise_dl,
    window_mask,
    within_validation_mask,
)

REPO = find_root(Path(__file__).resolve().parent)
CHANNELS = ["Fz", "FC1", "FC2", "C3", "Cz", "C4", "CP1", "CP2", "P3", "Pz", "P4", "Oz"]


def synthetic_epochs(subject, rng, n_std=150, n_tgt=30, n_dis=30, sfreq=250.0):
    """v2-like epochs: -0.2..0.8 s, 3 classes, 5 blocks, metadata subject/onset_s/block/rt, P300 on targets."""
    labels = np.array([1] * n_std + [2] * n_tgt + [3] * n_dis)
    rng.shuffle(labels)
    n = len(labels)
    times = np.arange(-0.2, 0.8 + 1e-9, 1 / sfreq)
    data = rng.normal(0, 5e-6, (n, len(CHANNELS), len(times)))
    p300 = np.exp(-0.5 * ((times - 0.4) / 0.08) ** 2)
    topo = np.array([0.3, 0.4, 0.4, 0.5, 0.8, 0.5, 0.9, 0.9, 0.8, 1.0, 0.8, 0.4])
    for cls, amp in ((2, 6e-6), (3, 3e-6)):
        data[labels == cls] += amp * topo[None, :, None] * p300[None, None, :]
    info = mne.create_info(CHANNELS, sfreq, "eeg")
    onsets = np.cumsum(rng.uniform(1.4, 1.8, n))
    events = np.c_[np.round(onsets * sfreq).astype(int), np.zeros(n, int), labels]
    metadata = pd.DataFrame(
        {
            "subject": f"sub-{subject}",
            "onset_s": onsets,
            "block": np.minimum(np.arange(n) * 5 // n, 4),
            "rt": np.where(labels == 2, rng.uniform(0.4, 0.7, n), np.nan),
        }
    )
    return mne.EpochsArray(
        data,
        info,
        events=events,
        tmin=-0.2,
        event_id={"standard": 1, "target": 2, "distractor": 3},
        metadata=metadata,
        verbose="error",
    )


def fast_config(subjects):
    config = copy.deepcopy(load_config(REPO))
    config["dataset"]["subjects"] = list(subjects)
    config["dataset"]["exclude"] = {}
    config["models"]["dl"].update(max_epochs=6, early_stopping_patience=3, batch_size=32)
    return config


class SyntheticRepo:
    """Temporary repository root with config, preproc_log.csv and synthetic epoch files."""

    def __init__(self, subjects, seed=0):
        self.root = Path(tempfile.mkdtemp(prefix="r5_test_"))
        self.config = fast_config(subjects)
        (self.root / "configs").mkdir()
        (self.root / "configs" / "config.yaml").write_text(yaml.safe_dump(self.config), encoding="utf-8")
        deriv = self.root / self.config["paths"]["derivatives"]
        deriv.mkdir(parents=True)
        rng = np.random.default_rng(seed)
        for s in subjects:
            synthetic_epochs(s, rng).save(deriv / f"sub-{s}_task-oddball_epo.fif", verbose="error")
        (self.root / "results").mkdir()
        pd.DataFrame(
            {"subject": [f"sub-{s}" for s in subjects], "pipeline_version": 2, "excluded_subject": False}
        ).to_csv(self.root / "results" / "preproc_log.csv", index=False)

    def cleanup(self):
        shutil.rmtree(self.root, ignore_errors=True)


class TestNetworks(unittest.TestCase):
    def test_output_shapes_at_dl_rate(self):
        """Both networks accept 127 channels for the full (103 samples) and early (58 samples) windows at 128 Hz."""
        config = load_config(REPO)
        for name, net_cls in NETWORKS.items():
            for n_samples in (103, 58):
                net = net_cls(127, n_samples, 128.0, **config["models"]["dl"][name]).eval()
                with torch.no_grad():
                    out = net(torch.zeros(3, 127, n_samples))
                self.assertEqual(tuple(out.shape), (3, 2), f"{name} {n_samples}")

    def test_eegnet_maxnorm(self):
        net = NETWORKS["eegnet"](8, 103, 128.0)
        with torch.no_grad():
            net.spatial.weight.mul_(100)
            net.classifier.weight.mul_(100)
        net.constrain()
        self.assertLessEqual(net.spatial.weight.flatten(1).norm(dim=1).max().item(), 1.0 + 1e-5)
        self.assertLessEqual(net.classifier.weight.norm(dim=1).max().item(), 0.25 + 1e-5)


class TestTrainingHelpers(unittest.TestCase):
    def test_channel_stats_match_numpy(self):
        X = np.random.default_rng(1).normal(3, 2, (50, 4, 20)).astype(np.float32)
        idx = np.arange(0, 50, 3)
        mean, std = channel_stats(X, idx, chunk=4)
        np.testing.assert_allclose(mean[0, :, 0], X[idx].mean(axis=(0, 2)), rtol=1e-5)
        np.testing.assert_allclose(std[0, :, 0], X[idx].std(axis=(0, 2)), rtol=1e-4)

    def test_within_validation_is_tail_of_each_training_block(self):
        meta = pd.DataFrame({"block": np.repeat(np.arange(5), 20)})
        y = np.tile([0, 0, 0, 1], 25)
        train_idx = np.flatnonzero(meta.block != 2)
        mask = within_validation_mask(meta, train_idx, y, 0.2)
        val = train_idx[mask]
        self.assertEqual(len(val), 16)
        for b in (0, 1, 3, 4):
            np.testing.assert_array_equal(val[meta.block.to_numpy()[val] == b], np.arange(b * 20 + 16, b * 20 + 20))
        self.assertFalse(np.isin(val, np.flatnonzero(meta.block == 2)).any())

    def test_cross_validation_subjects_from_train_only(self):
        train = [f"{i:02d}" for i in range(1, 29)]
        val = cross_validation_subjects(train, 0.2, seed=42)
        self.assertEqual(len(val), 6)
        self.assertTrue(set(val) <= set(train))
        self.assertEqual(val, cross_validation_subjects(train, 0.2, seed=42))

    def test_fit_uses_fit_part_only_and_checkpoint_roundtrip(self):
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (120, 6, 103)).astype(np.float32)
        y = np.r_[np.zeros(100, int), np.ones(20, int)]
        X[y == 1, :, 40:60] += 1.5
        X[:60] += 10.0  # shift the first half; statistics must come from fit trials only
        val = np.zeros(120, bool)
        val[np.r_[80:100, 110:120]] = True
        clf = TorchClassifier("eegnet", 128, seed=0, max_epochs=3, patience=2, batch_size=32, device="cpu")
        clf.fit(X, y, val_mask=val)
        np.testing.assert_allclose(clf.mean_[0, :, 0], X[~val].mean(axis=(0, 2)), rtol=1e-4)
        self.assertEqual((clf.n_fit_, clf.n_val_), (90, 30))
        with tempfile.TemporaryDirectory() as tmp:
            path = clf.save(Path(tmp) / "m.pt", ch_names=list("abcdef"))
            clf2, info = load_checkpoint(path)
        np.testing.assert_allclose(clf.decision_function(X), clf2.decision_function(X), rtol=1e-5, atol=1e-5)
        self.assertEqual(info["extra"]["ch_names"], list("abcdef"))
        proba = clf2.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)
        np.testing.assert_array_equal(clf2.predict(X), (proba[:, 1] > 0.5).astype(int))


class TestWithinSplit(unittest.TestCase):
    def test_equal_segments_handle_short_last_block(self):
        """Like sub-13: last S10 block has no target -> equal contiguous segments still give 5 valid folds."""
        config = load_config(REPO)
        n = 300
        meta = pd.DataFrame({"onset_s": np.arange(n) * 1.5, "block": np.r_[np.repeat([0, 1, 2, 3], 70), [4] * 20]})
        y = np.zeros(n, int)
        y[np.arange(3, 280, 7)] = 1  # no target in the short last block
        cfg = copy.deepcopy(config)
        cfg["evaluation"]["within"]["split"] = "s10_blocks"
        with self.assertRaises(ValueError):
            list(iter_within_folds(_TrialTable(within_cv_metadata(meta, cfg)), y, config=cfg))
        cfg["evaluation"]["within"]["split"] = "equal_segments"
        cv_meta = within_cv_metadata(meta, cfg)
        folds = list(iter_within_folds(_TrialTable(cv_meta), y, config=cfg))
        self.assertEqual(len(folds), 5)
        self.assertEqual([len(t) for _, _, t in folds], [60] * 5)
        for _, _, test in folds:
            self.assertTrue(np.all(np.diff(test) == 1))  # contiguous in time
        np.testing.assert_array_equal(cv_meta["s10_block"], meta["block"])


class TestChannelAlignment(unittest.TestCase):
    def test_reorder_by_name(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(4, 3, 5)).astype(np.float32)
        d = align_channels({"X": X[:, [2, 0, 1]], "ch_names": ["c", "a", "b"]}, ["a", "b", "c"])
        np.testing.assert_array_equal(d["X"], X)
        self.assertEqual(d["ch_names"], ["a", "b", "c"])
        with self.assertRaises(ValueError):
            align_channels({"X": X, "ch_names": ["a", "b", "x"]}, ["a", "b", "c"])


class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.subjects = ["01", "03", "04", "05", "06", "07"]
        cls.repo = SyntheticRepo(cls.subjects)
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        cls.repo.cleanup()

    def test_loader_resamples_then_crops(self):
        d = load_dl_subject(self.repo.root, self.repo.config, "01", "tgt_std")
        self.assertEqual(d["X"].dtype, np.float32)
        self.assertEqual(len(d["y"]), 180)
        self.assertEqual(int(d["y"].sum()), 30)
        self.assertTrue(d["metadata"]["onset_s"].is_monotonic_increasing)
        self.assertEqual(window_mask(d["times"], self.repo.config, "full").sum(), 103)
        self.assertEqual(window_mask(d["times"], self.repo.config, "early").sum(), 58)
        self.assertGreater(np.abs(d["X"]).max(), 1.0)  # µV, not V
        folds = list(iter_within_folds(_TrialTable(d["metadata"]), d["y"], config=self.repo.config))
        self.assertEqual(len(folds), 5)

    def test_grid_outputs_and_resume(self):
        root, config = self.repo.root, self.repo.config
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            written = run_dl_grid(root, config, "eegnet", windows=("full", "early"), device="cpu", log=lambda m: None)
        self.assertEqual(len(written), 4)
        for path in written:
            df = validate_metrics_df(pd.read_csv(path))
            self.assertEqual(list(df.columns), REQUIRED_COLUMNS)
            self.assertTrue((df.family == "dl").all() and (df.model == "eegnet").all())
            if "_within_" in path.name:
                self.assertEqual(len(df), len(self.subjects) * 5)
            else:
                self.assertEqual(sorted(df.subject), [f"sub-{s}" for s in self.subjects])  # one row per subject
        full_within = pd.read_csv(root / "results/metrics/eegnet_within_full_tgt_std.csv")
        self.assertGreater(full_within.auc.mean(), 0.6)  # synthetic P300 is learnable

        folds_file = folds_csv_path(root)
        self.assertTrue(folds_file.exists())
        self.assertEqual(
            cross_subject_folds(root, config, self.subjects), cross_subject_folds(root, config, self.subjects)
        )
        ckpts = sorted((root / "results/checkpoints/eegnet/cross_full_tgt_std").glob("fold-*.pt"))
        self.assertEqual(len(ckpts), 5)
        _, info = load_checkpoint(ckpts[0])
        self.assertEqual(len(info["extra"]["ch_names"]), len(CHANNELS))
        self.assertFalse(set(info["extra"]["test_subjects"]) & set(info["extra"]["train_subjects"]))
        pred = pd.read_csv(root / "results/predictions/eegnet_cross_full_tgt_std.csv.gz")
        self.assertEqual(len(pred), len(self.subjects) * 180)
        self.assertFalse(list(root.glob("results/metrics/*.partial.json")))

        mtime = written[0].stat().st_mtime
        run_dl_grid(root, config, "eegnet", settings=("within",), windows=("full",), device="cpu", log=lambda m: None)
        self.assertEqual(written[0].stat().st_mtime, mtime)  # finished results are skipped, not recomputed
        table = summarise_dl(root, config, models=("eegnet",))
        self.assertEqual(len(table), 4)
        self.assertTrue((table.n_subjects == len(self.subjects)).all())

    def test_subset_goes_to_dev_folder(self):
        root, config = self.repo.root, self.repo.config
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            run_dl_grid(
                root,
                config,
                "shallow",
                settings=("cross",),
                windows=("early",),
                subjects=["01", "03"],
                device="cpu",
                allow_fewer_splits=True,
                save_checkpoints=False,
                log=lambda m: None,
            )
        self.assertTrue((root / "results/dev/metrics/shallow_cross_early_tgt_std.csv").exists())
        self.assertFalse((root / "results/metrics/shallow_cross_early_tgt_std.csv").exists())


class TestConfig(unittest.TestCase):
    def test_factory_reads_config(self):
        config = load_config(REPO)
        clf = make_dl(config, "shallow", device="cpu")(7)
        self.assertEqual(clf.sfreq, config["models"]["dl"]["sfreq"])
        self.assertEqual(clf.patience, config["models"]["dl"]["early_stopping_patience"])
        self.assertEqual(clf.net_kwargs["n_filters"], config["models"]["dl"]["shallow"]["n_filters"])


if __name__ == "__main__":
    unittest.main()
