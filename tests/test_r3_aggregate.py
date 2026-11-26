"""F3 numerical/validation tests: ALL fixtures are SYNTHETIC TEST ONLY.

Temporary test figures are visibly labelled and deleted with their directory.
No fixture is a scientific result or written to the repository's results/.
"""

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import yaml
from matplotlib.axes import Axes
from PIL import Image

from eegproj.config import find_root, load_config
from eegproj.evaluation.aggregate import (
    F3_MODELS,
    generate_f3,
    load_f3_inputs,
    summarise_f3,
    validate_f3_metrics,
    verify_cross_folds,
)
from eegproj.evaluation.metrics import REQUIRED_COLUMNS
from eegproj.evaluation.plots import plot_f3
from eegproj.evaluation.splits import format_bids_subject, iter_cross_folds
from eegproj.evaluation.stats import aggregate_subject_scores


def _synthetic_test_only_inputs():
    config = copy.deepcopy(load_config(find_root(Path(__file__).resolve().parent)))
    subjects = ["sub-01", "sub-03", "sub-04", "sub-05", "sub-06"]
    config["dataset"]["subjects"] = [s.removeprefix("sub-") for s in subjects]
    config["dataset"]["exclude"] = {}
    assignment = {
        format_bids_subject(s): fold for fold, _, test in iter_cross_folds(subjects, config=config) for s in test
    }
    folds = pd.DataFrame({"subject": subjects, "fold": [assignment[s] for s in subjects]})
    log = pd.DataFrame(
        {
            "subject": subjects,
            "pipeline_version": 2,
            "excluded_subject": False,
            "target_after": 10,
            "standard_after": 20,
        }
    )
    rows = []
    for model, family in F3_MODELS.items():
        for setting in ("within", "cross"):
            for window in ("full", "early"):
                for index, subject in enumerate(subjects):
                    for fold in range(5) if setting == "within" else [assignment[subject]]:
                        within = setting == "within"
                        rows.append(
                            {
                                "subject": subject,
                                "fold": fold,
                                "model": model,
                                "family": family,
                                "setting": setting,
                                "window": window,
                                "task": "tgt_std",
                                "bal_acc": (0.2 + fold * 0.2 - index * 0.02) if within else 0.9 - index * 0.1,
                                "auc": 0.7 + (fold * 0.01 if within else 0) + index * 0.01,
                                "f1_target": 0.3 + (fold * 0.05 if within else 0) + index * 0.02,
                                "n_train": 100,
                                "n_test": (fold + 1) * 2 if within else 30,
                                "seed": config["seed"],
                            }
                        )
    return config, subjects, folds, log, pd.DataFrame(rows)[REQUIRED_COLUMNS]


def _write_synthetic_test_only_repo(root, config, folds, log, raw):
    (root / "configs").mkdir()
    (root / "configs/config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    metrics = root / config["paths"]["metrics"]
    metrics.mkdir(parents=True)
    for (model, setting, window, task), group in raw.groupby(["model", "setting", "window", "task"]):
        group.to_csv(metrics / f"{model}_{setting}_{window}_{task}.csv", index=False)
    folds.to_csv(root / "results/cv_cross_subject_folds.csv", index=False)
    log.to_csv(root / "results/preproc_log.csv", index=False)


class F3AggregationTests(unittest.TestCase):
    def setUp(self):
        self.config, self.subjects, self.folds, self.log, self.raw = _synthetic_test_only_inputs()

    def test_subject_means_not_fold_or_trial_weighted_and_cross_unchanged(self):
        raw = validate_f3_metrics(self.raw, self.config, self.subjects, self.folds)
        subjects = aggregate_subject_scores(raw)
        self.assertEqual(len(subjects), 16 * 5)
        within = subjects.query("model == 'lda_tw' and setting == 'within' and window == 'full'")
        first = within.set_index("subject").loc["sub-01"]
        self.assertAlmostEqual(first.bal_acc, 0.6)
        self.assertAlmostEqual(first.auc, 0.72)
        self.assertAlmostEqual(first.f1_target, 0.4)
        # Unequal n_test deliberately makes a trial-weighted fold mean different.
        self.assertNotAlmostEqual(first.bal_acc, 22 / 30)
        cross = subjects.query("model == 'lda_tw' and setting == 'cross' and window == 'full'")
        np.testing.assert_allclose(cross.sort_values("subject").bal_acc, [0.9, 0.8, 0.7, 0.6, 0.5])

    def test_summary_quantiles_and_sample_sd_are_over_subjects(self):
        summary = summarise_f3(aggregate_subject_scores(self.raw))
        self.assertEqual(len(summary), 16 * 3)
        row = summary.query(
            "model == 'lda_tw' and setting == 'within' and window == 'full' and metric == 'bal_acc'"
        ).iloc[0]
        self.assertEqual(row.n_subjects, 5)
        for name, expected in {"mean": 0.56, "median": 0.56, "q1": 0.54, "q3": 0.58, "iqr": 0.04}.items():
            self.assertAlmostEqual(row[name], expected)
        self.assertAlmostEqual(row.sd, np.sqrt(0.001))

    def test_missing_duplicate_or_invalid_metrics_are_rejected(self):
        variants = {
            "empty input": self.raw.iloc[:0],
            "missing schema column": self.raw.drop(columns="auc"),
            "unexpected schema column": self.raw.assign(unexpected=1),
            "wrong column order": self.raw[self.raw.columns[::-1]],
            "missing within fold": self.raw.drop(index=2),
            "duplicate row": pd.concat([self.raw, self.raw.iloc[[0]]], ignore_index=True),
            "missing cohort subject": self.raw.loc[self.raw.subject.ne("sub-06")],
            "missing condition": self.raw.loc[~(self.raw.model.eq("shallow") & self.raw.window.eq("early"))],
        }
        for column, bad_values in {
            "bal_acc": [np.nan, np.inf, -0.1, 1.1],
            "auc": [np.nan, 2],
            "f1_target": [np.nan, -1],
            "fold": [0.5, 5, np.nan],
            "seed": [43, 42.5],
            "n_train": [0, 1.5],
            "n_test": [0, 1.5, 3],
            "family": ["dl"],
            "subject": ["sub-99", None],
            "task": ["dis_std"],
        }.items():
            for value in bad_values:
                table = self.raw.copy()
                table[column] = table[column].astype(object)
                table.loc[2, column] = value  # not the first fold: catches NaNs hidden by group means
                variants[f"{column}={value}"] = table
        cross = self.raw.loc[self.raw.setting.eq("cross")].iloc[[0]].copy()
        cross["fold"] = (cross["fold"] + 1) % 5
        variants["two cross rows for a subject"] = pd.concat([self.raw, cross], ignore_index=True)
        cross_count_mismatch = self.raw.copy()
        cross_count_mismatch.loc[cross_count_mismatch.setting.eq("cross"), "n_test"] = 32
        variants["different within/cross trial totals"] = cross_count_mismatch
        for name, table in variants.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_f3_metrics(table, self.config, self.subjects, self.folds)

    def test_shared_fold_mismatch_and_invalid_assignment_rejected(self):
        variants = [
            self.folds.iloc[:-1],
            pd.concat([self.folds, self.folds.iloc[[0]]]),
            self.folds.assign(fold=[1, 0, 2, 3, 4]),
            self.folds.assign(fold=[0, 1, 2, 3, 3]),
            self.folds.assign(fold=[0.5, 1, 2, 3, 4]),
        ]
        for folds in variants:
            with self.subTest(folds=folds.to_dict("list")), self.assertRaises(ValueError):
                validate_f3_metrics(self.raw, self.config, self.subjects, folds)

    def test_plot_and_summary_reject_unaggregated_folds(self):
        with self.assertRaisesRegex(ValueError, "one score per subject"):
            summarise_f3(self.raw)
        with tempfile.TemporaryDirectory(prefix="r3_synthetic_test_only_") as temp:
            with self.assertRaisesRegex(ValueError, "one score per subject"):
                plot_f3(self.raw, self.config, Path(temp) / "SYNTHETIC_TEST_ONLY", synthetic_test_only=True)
            self.assertEqual(list(Path(temp).iterdir()), [])

    def test_missing_inputs_fail_before_any_output_or_fold_creation(self):
        with tempfile.TemporaryDirectory(prefix="r3_synthetic_test_only_") as temp:
            root = Path(temp)
            with self.assertRaises(FileNotFoundError) as error:
                generate_f3(root, self.config)
            self.assertIn("lda_tw_within_full_tgt_std.csv", str(error.exception))
            self.assertIn("preproc_log.csv", str(error.exception))
            self.assertEqual(list(root.iterdir()), [])

    def test_loader_checks_filename_handoff_counts_and_provenance_inputs(self):
        with tempfile.TemporaryDirectory(prefix="r3_synthetic_test_only_") as temp:
            root = Path(temp)
            _write_synthetic_test_only_repo(root, self.config, self.folds, self.log, self.raw)
            loaded, sources, audit = load_f3_inputs(root, self.config)
            self.assertEqual(len(loaded), 240)
            self.assertEqual(len(sources), 19)
            self.assertTrue(audit["cross_fold_verification"]["supplied_fold_csv_present"])
            # Canonical filename must match row identity even when all 16 files exist.
            path = root / "results/metrics/lda_tw_within_full_tgt_std.csv"
            original = path.read_bytes()
            changed = pd.read_csv(path).assign(model="xdawn_rg")
            changed.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "filename"):
                load_f3_inputs(root, self.config)
            path.write_bytes(original)
            log_path = root / "results/preproc_log.csv"
            for bad_log in (
                self.log.iloc[:-1],
                pd.concat([self.log, self.log.iloc[[0]]]),
                self.log.assign(pipeline_version=1),
                self.log.assign(target_after=11),
                self.log.assign(excluded_subject="unknown"),
            ):
                bad_log.to_csv(log_path, index=False)
                with self.subTest(log=bad_log.to_dict("list")), self.assertRaises((ValueError, KeyError)):
                    load_f3_inputs(root, self.config)
            self.log.to_csv(log_path, index=False)
            excluded_config = copy.deepcopy(self.config)
            excluded_config["dataset"]["exclude"] = {"01": "SYNTHETIC TEST ONLY exclusion"}
            with self.assertRaisesRegex(ValueError, "a-priori excluded"):
                load_f3_inputs(root, excluded_config)

    def test_missing_original_fold_csv_uses_metrics_and_never_creates_a_manifest(self):
        with tempfile.TemporaryDirectory(prefix="r3_synthetic_test_only_") as temp:
            root = Path(temp)
            _write_synthetic_test_only_repo(root, self.config, self.folds, self.log, self.raw)
            fold_path = root / "results/cv_cross_subject_folds.csv"
            fold_path.unlink()
            loaded, sources, audit = load_f3_inputs(root, self.config)
            self.assertEqual(len(loaded), 240)
            self.assertEqual(len(sources), 18)
            evidence = audit["cross_fold_verification"]
            self.assertEqual(evidence["assignment_source"], "metrics_consensus")
            self.assertFalse(evidence["supplied_fold_csv_present"])
            self.assertFalse(evidence["is_original_r4_manifest"])
            self.assertTrue(evidence["matches_current_groupkfold"])
            self.assertEqual(evidence["n_cross_conditions"], 8)
            self.assertFalse(fold_path.exists())

    def test_cross_consensus_and_current_groupkfold_are_both_required(self):
        inconsistent = self.raw.copy()
        mask = inconsistent.setting.eq("cross") & inconsistent.model.eq("eegnet") & inconsistent.window.eq("early")
        inconsistent.loc[mask, "fold"] = (inconsistent.loc[mask, "fold"] + 1) % 5
        with self.assertRaisesRegex(ValueError, "inconsistent cross-subject"):
            verify_cross_folds(inconsistent, self.config, self.subjects)
        consistent_but_wrong = self.raw.copy()
        mask = consistent_but_wrong.setting.eq("cross")
        consistent_but_wrong.loc[mask, "fold"] = (consistent_but_wrong.loc[mask, "fold"] + 1) % 5
        with self.assertRaisesRegex(ValueError, "current GroupKFold"):
            verify_cross_folds(consistent_but_wrong, self.config, self.subjects)
        supplied_wrong = self.folds.assign(fold=(self.folds.fold + 1) % 5)
        with self.assertRaisesRegex(ValueError, "disagrees with metrics"):
            verify_cross_folds(self.raw, self.config, self.subjects, supplied_wrong)

    def test_control_task_csvs_are_audited_before_main_task_export(self):
        log = self.log.assign(distractor_after=10)
        frames = [self.raw]
        for task in ("dis_std", "tgt_dis"):
            control = self.raw.loc[self.raw.family.eq("classical")].assign(task=task)
            if task == "tgt_dis":
                control.loc[control.setting.eq("within"), "n_test"] = 4
                control.loc[control.setting.eq("cross"), "n_test"] = 20
            frames.append(control)
        raw = pd.concat(frames, ignore_index=True)
        with tempfile.TemporaryDirectory(prefix="r3_synthetic_test_only_") as temp:
            root = Path(temp)
            _write_synthetic_test_only_repo(root, self.config, self.folds, log, raw)
            loaded, _, audit = load_f3_inputs(root, self.config)
            self.assertEqual(len(loaded), 240)
            self.assertEqual(audit["n_metric_files"], 32)
            self.assertEqual(audit["n_all_metric_rows"], 480)
            self.assertEqual(audit["cross_fold_verification"]["n_cross_conditions"], 16)
            path = root / "results/metrics/lda_tw_within_full_dis_std.csv"
            corrupted = pd.read_csv(path)
            corrupted.loc[2, "bal_acc"] = np.nan
            corrupted.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "finite proportions"):
                load_f3_inputs(root, self.config)

    def test_export_matches_source_scores_plot_data_and_hashes(self):
        import json

        captured = []
        original_boxplot = Axes.boxplot

        def capture_boxplot(ax, data, *args, **kwargs):
            captured.append([values.copy() for values in data])
            return original_boxplot(ax, data, *args, **kwargs)

        with tempfile.TemporaryDirectory(prefix="r3_synthetic_test_only_") as temp:
            root = Path(temp)
            _write_synthetic_test_only_repo(root, self.config, self.folds, self.log, self.raw)
            source_bytes = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}

            def labelled_test_plot(scores, config, _stem):
                return plot_f3(scores, config, root / "SYNTHETIC_TEST_ONLY", synthetic_test_only=True)

            with (
                patch("eegproj.evaluation.plots.plot_f3", side_effect=labelled_test_plot),
                patch.object(Axes, "boxplot", capture_boxplot),
            ):
                png, pdf, subject_csv, summary_csv, manifest_path = generate_f3(root, self.config)
            self.assertEqual(png.name, "SYNTHETIC_TEST_ONLY.png")
            with Image.open(png) as picture:
                np.testing.assert_allclose(picture.info["dpi"], [300, 300], atol=0.01)
            self.assertTrue(pdf.read_bytes().startswith(b"%PDF"))
            subjects = pd.read_csv(subject_csv)
            self.assertEqual(len(subjects), 80)
            self.assertEqual(len(pd.read_csv(summary_csv)), 48)
            # Independent calculation directly from the source rows, for every plotted box.
            for panel, (setting, window) in enumerate([(s, w) for s in ("within", "cross") for w in ("full", "early")]):
                for model_index, model in enumerate(F3_MODELS):
                    source = self.raw.loc[
                        self.raw.setting.eq(setting) & self.raw.window.eq(window) & self.raw.model.eq(model)
                    ]
                    expected = [
                        source.loc[source.subject.eq(s), "bal_acc"].to_numpy().mean() for s in sorted(self.subjects)
                    ]
                    np.testing.assert_allclose(captured[panel][model_index], expected)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["n_raw_rows"], 240)
            self.assertEqual(manifest["n_subjects"], 5)
            for source in manifest["sources"]:
                self.assertEqual(source["sha256"], hashlib.sha256((root / source["path"]).read_bytes()).hexdigest())
            self.assertFalse((root / "figures/F3_balanced_accuracy.png").exists())
            for path, original in source_bytes.items():
                self.assertEqual(path.read_bytes(), original, f"Source modified: {path}")

    def test_output_guard_prevents_writing_into_source_metrics(self):
        config = copy.deepcopy(self.config)
        config["paths"]["figures"] = "results/metrics/figures"
        with (
            tempfile.TemporaryDirectory(prefix="r3_synthetic_test_only_") as temp,
            self.assertRaisesRegex(ValueError, "outside the source metrics"),
        ):
            generate_f3(Path(temp), config)


if __name__ == "__main__":
    unittest.main()
