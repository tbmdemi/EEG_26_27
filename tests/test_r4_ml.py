"""Unit and integration tests for R4 classical ML, feature extraction, channel alignment, and RQ2 stats."""

import copy
import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import yaml

from eegproj.config import find_root, load_config
from eegproj.evaluation import (
    REQUIRED_COLUMNS,
    aggregate_subject_scores,
    compute_rq2_wilcoxon_holm,
    cross_subject_folds,
    folds_csv_path,
    plot_f4_paired,
    validate_metrics_df,
)
from eegproj.features import TimeWindowAverager, compute_bin_masks
from eegproj.models import (
    build_lda_tw,
    build_xdawn_rg,
    extract_lda_haufe_patterns,
    run_classical_experiment,
)
from eegproj.models.classical import _RankSafeXdawnEstimator

REPO = find_root(Path(__file__).resolve().parent)
CHANNELS = ["Fz", "FC1", "FC2", "C3", "Cz", "C4", "CP1", "CP2", "P3", "Pz", "P4", "Oz"]


def _load_stats_script():
    spec = importlib.util.spec_from_file_location("stats_rq2_cli", REPO / "scripts" / "03_stats_rq2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _synthetic_epochs(subject: str, rng: np.random.Generator, ch_names=None, n_std=100, n_tgt=25, n_dis=25):
    ch_names = list(ch_names or CHANNELS)
    sfreq = 250.0
    labels = np.array([1] * n_std + [2] * n_tgt + [3] * n_dis)
    rng.shuffle(labels)
    n_trials = len(labels)
    times = np.arange(-0.2, 0.8 + 1e-9, 1.0 / sfreq)
    data = rng.normal(0.0, 4e-6, (n_trials, len(ch_names), len(times)))
    p300 = np.exp(-0.5 * ((times - 0.4) / 0.08) ** 2)
    topo_map = {
        "Fz": 0.3,
        "FC1": 0.4,
        "FC2": 0.4,
        "C3": 0.5,
        "Cz": 0.8,
        "C4": 0.5,
        "CP1": 0.9,
        "CP2": 0.9,
        "P3": 0.8,
        "Pz": 1.0,
        "P4": 0.8,
        "Oz": 0.4,
    }
    topo = np.array([topo_map[ch] for ch in ch_names])
    for cls, amp in ((2, 7e-6), (3, 3.5e-6)):
        data[labels == cls] += amp * topo[None, :, None] * p300[None, None, :]
    info = mne.create_info(ch_names, sfreq, "eeg")
    onsets = np.cumsum(rng.uniform(1.4, 1.8, n_trials))
    events = np.c_[np.round(onsets * sfreq).astype(int), np.zeros(n_trials, int), labels]
    metadata = pd.DataFrame(
        {
            "subject": f"sub-{subject}",
            "onset_s": onsets,
            "block": np.minimum(np.arange(n_trials) * 5 // n_trials, 4),
            "rt": np.where(labels == 2, rng.uniform(0.4, 0.7, n_trials), np.nan),
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


class _SyntheticClassicalRepo:
    def __init__(self, subjects, permute_channels_for=None, seed=123):
        self.root = Path(tempfile.mkdtemp(prefix="r4_test_"))
        self.config = copy.deepcopy(load_config(REPO))
        self.config["dataset"]["subjects"] = list(subjects)
        self.config["dataset"]["exclude"] = {}
        (self.root / "configs").mkdir()
        (self.root / "configs" / "config.yaml").write_text(yaml.safe_dump(self.config), encoding="utf-8")
        deriv = self.root / self.config["paths"]["derivatives"]
        deriv.mkdir(parents=True)
        (self.root / "results").mkdir()
        permute_set = set(permute_channels_for or [])
        rng = np.random.default_rng(seed)
        for s in subjects:
            ch_order = list(reversed(CHANNELS)) if s in permute_set else CHANNELS
            _synthetic_epochs(s, rng, ch_names=ch_order).save(deriv / f"sub-{s}_task-oddball_epo.fif", verbose="error")
        pd.DataFrame(
            {"subject": [f"sub-{s}" for s in subjects], "pipeline_version": 2, "excluded_subject": False}
        ).to_csv(self.root / "results" / "preproc_log.csv", index=False)

    def cleanup(self):
        shutil.rmtree(self.root, ignore_errors=True)


class TestFeaturesAndModels(unittest.TestCase):
    def test_time_window_averager_bin_counts(self):
        times_full = np.arange(0.0, 0.8 + 1e-9, 1.0 / 250.0)
        times_early = np.arange(0.0, 0.45 + 1e-9, 1.0 / 250.0)
        edges_full, masks_full = compute_bin_masks(times_full, 50.0)
        edges_early, masks_early = compute_bin_masks(times_early, 50.0)
        self.assertEqual(len(masks_full), 16)
        self.assertEqual(len(masks_early), 9)
        self.assertEqual(len(edges_full), 17)
        self.assertEqual(len(edges_early), 10)

        X = np.ones((4, 12, len(times_full))) * 5.0
        tw = TimeWindowAverager(times=times_full, bin_ms=50.0)
        Z = tw.fit_transform(X)
        self.assertEqual(Z.shape, (4, 12 * 16))
        np.testing.assert_allclose(Z, 5.0)

    def test_rank_safe_xdawn_and_haufe_patterns(self):
        cfg = load_config(REPO)
        rng = np.random.default_rng(7)
        times = np.arange(0.0, 0.8 + 1e-9, 1.0 / 250.0)
        # Create rank-deficient EEG (average-referenced, rank <= 11 out of 12)
        X = rng.normal(0.0, 2.0, (40, 12, len(times)))
        y = np.r_[np.zeros(30, int), np.ones(10, int)]
        X[y == 1, 9, 75:125] += 4.0
        X -= X.mean(axis=1, keepdims=True)

        est = _RankSafeXdawnEstimator(estimator="lwf")
        cov_p = est(X.mean(axis=0))
        cov_x = est(X.transpose(1, 0, 2).reshape(12, -1))
        self.assertEqual(cov_p.shape, (12, 12))
        self.assertEqual(cov_x.shape, (12, 12))
        self.assertTrue(np.isfinite(cov_p).all() and np.isfinite(cov_x).all())

        lda_pipe = build_lda_tw(times, cfg)
        self.assertEqual(lda_pipe.named_steps["clf"].priors, [0.5, 0.5])
        lda_pipe.fit(X, y)
        haufe = extract_lda_haufe_patterns(lda_pipe, X)
        self.assertEqual(haufe["pattern_uv"].shape, (12, 16))
        self.assertEqual(haufe["weights"].shape, (12, 16))
        self.assertTrue(np.isfinite(haufe["pattern_uv"]).all())

        xdawn_pipe = build_xdawn_rg(cfg)
        self.assertEqual(xdawn_pipe.named_steps["clf"].class_weight, "balanced")
        xdawn_pipe.fit(X, y)
        probs = xdawn_pipe.predict_proba(X)
        self.assertEqual(probs.shape, (40, 2))
        self.assertTrue(np.isfinite(probs).all())


class TestRunnerAndChannelAlignment(unittest.TestCase):
    def test_cross_subject_channel_reordering_gives_identical_scores(self):
        subjects = ["01", "03", "04", "05", "06"]
        repo_std = _SyntheticClassicalRepo(subjects, permute_channels_for=None, seed=99)
        repo_perm = _SyntheticClassicalRepo(subjects, permute_channels_for={"03", "05"}, seed=99)
        try:
            df_std = run_classical_experiment(repo_std.root, repo_std.config, "lda_tw", "cross", "full", "tgt_std")
            df_perm = run_classical_experiment(repo_perm.root, repo_perm.config, "lda_tw", "cross", "full", "tgt_std")
            validate_metrics_df(df_std)
            self.assertEqual(list(df_std.columns), REQUIRED_COLUMNS)
            self.assertTrue(folds_csv_path(repo_std.root).exists())
            pd.testing.assert_frame_equal(
                df_std.sort_values("subject", ignore_index=True),
                df_perm.sort_values("subject", ignore_index=True),
                atol=1e-10,
            )
            self.assertEqual(
                cross_subject_folds(repo_std.root, repo_std.config, subjects),
                cross_subject_folds(repo_perm.root, repo_perm.config, subjects),
            )
        finally:
            repo_std.cleanup()
            repo_perm.cleanup()

    def test_within_equal_segments_with_runner(self):
        subjects = ["01", "03"]
        repo = _SyntheticClassicalRepo(subjects, seed=42)
        try:
            df = run_classical_experiment(repo.root, repo.config, "xdawn_rg", "within", "early", "tgt_std")
            validate_metrics_df(df)
            self.assertEqual(len(df), 2 * 5)
            self.assertTrue((df["window"] == "early").all())
            self.assertGreater(df["bal_acc"].mean(), 0.7)
        finally:
            repo.cleanup()


class TestRQ2StatsAndSafety(unittest.TestCase):
    def test_median_selection_on_full_window_and_plot_f4(self):
        rng = np.random.default_rng(42)
        subs = [f"sub-{i:02d}" for i in range(1, 21)]
        rows = []
        # Make cls_b beat cls_a in median on full (even if outlier pulls cls_a mean up)
        for sub in subs:
            rows.append(
                {
                    "family": "classical",
                    "model": "cls_a",
                    "setting": "within",
                    "window": "full",
                    "task": "tgt_std",
                    "subject": sub,
                    "bal_acc": 0.75,
                    "auc": 0.80,
                    "f1_target": 0.60,
                }
            )
            rows.append(
                {
                    "family": "classical",
                    "model": "cls_b",
                    "setting": "within",
                    "window": "full",
                    "task": "tgt_std",
                    "subject": sub,
                    "bal_acc": 0.82,
                    "auc": 0.85,
                    "f1_target": 0.65,
                }
            )
            rows.append(
                {
                    "family": "classical",
                    "model": "cls_a",
                    "setting": "within",
                    "window": "early",
                    "task": "tgt_std",
                    "subject": sub,
                    "bal_acc": 0.79,
                    "auc": 0.82,
                    "f1_target": 0.61,
                }
            )
            rows.append(
                {
                    "family": "classical",
                    "model": "cls_b",
                    "setting": "within",
                    "window": "early",
                    "task": "tgt_std",
                    "subject": sub,
                    "bal_acc": 0.76,
                    "auc": 0.81,
                    "f1_target": 0.59,
                }
            )
            rows.append(
                {
                    "family": "dl",
                    "model": "eegnet",
                    "setting": "within",
                    "window": "full",
                    "task": "tgt_std",
                    "subject": sub,
                    "bal_acc": float(0.88 + rng.normal(0, 0.02)),
                    "auc": 0.92,
                    "f1_target": 0.75,
                }
            )
            rows.append(
                {
                    "family": "dl",
                    "model": "eegnet",
                    "setting": "within",
                    "window": "early",
                    "task": "tgt_std",
                    "subject": sub,
                    "bal_acc": float(0.81 + rng.normal(0, 0.02)),
                    "auc": 0.86,
                    "f1_target": 0.68,
                }
            )
        sub_df = aggregate_subject_scores(pd.DataFrame(rows))
        stats_df = compute_rq2_wilcoxon_holm(sub_df, metric="bal_acc")
        self.assertTrue((stats_df["classical_model"] == "cls_b").all())
        self.assertTrue((stats_df["dl_model"] == "eegnet").all())
        self.assertTrue(stats_df["p_holm"].between(0.0, 1.0).all())
        with tempfile.TemporaryDirectory() as tmp:
            png, pdf = plot_f4_paired(sub_df, stats_df, Path(tmp) / "F4_test", metric="bal_acc", dpi=300)
            self.assertTrue(png.exists() and pdf.exists())

    def test_out_csv_guard_blocks_results_metrics_dir(self):
        mod = _load_stats_script()
        cfg = load_config(REPO)
        with self.assertRaises(ValueError):
            mod._resolve_safe_out_csv(REPO, cfg, "results/metrics/rq2_bad.csv")
        safe = mod._resolve_safe_out_csv(REPO, cfg, "results/rq2_wilcoxon_holm.csv")
        self.assertEqual(safe, (REPO / "results" / "rq2_wilcoxon_holm.csv").resolve())


if __name__ == "__main__":
    unittest.main()
