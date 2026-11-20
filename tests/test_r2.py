"""R2 synthetic regression test; no real EEG recording is downloaded."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import pandas as pd
import r2_checks
from r2_validation import compare_release_logs


class R2IntegrationTests(unittest.TestCase):
    def test_candidate_pipeline_and_isolation(self):
        r2_checks.test_r2()


class ReleaseQCAnnotationsTests(unittest.TestCase):
    def test_qc_annotations_can_be_added_without_changing_scientific_data(self):
        source = pd.DataFrame({"subject": ["sub-01"], "target_after": [61], "visual_review": ["pending"]})
        reviewed = source.assign(visual_review="summary_plots_reviewed", visual_review_by="Codex")
        compare_release_logs(source, reviewed)

    def test_scientific_count_change_cannot_be_hidden_as_qc_update(self):
        source = pd.DataFrame({"subject": ["sub-01"], "target_after": [61], "visual_review": ["pending"]})
        with self.assertRaises(AssertionError):
            compare_release_logs(source, source.assign(target_after=62, visual_review="reviewed"))
