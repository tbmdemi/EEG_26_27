"""Test cho code giải thích mô hình của R1 (src/eegproj/explain/dl.py), dùng mô hình và dữ liệu tổng hợp nhỏ.

python -m unittest tests.test_r1_explain -v
"""

import unittest

import numpy as np
import torch

from eegproj.explain.dl import (
    MOTOR_ROI,
    P300_ROI,
    eegnet_patterns,
    filter_importance,
    integrated_gradients,
    logit_diff,
    roi_table,
    time_bins,
)
from eegproj.models.dl import TorchClassifier

SFREQ = 128.0


def tiny_eegnet(n_channels, n_samples, F1=2, D=2, seed=0):
    """EEGNet nhỏ chưa train, bọc trong TorchClassifier như một mô hình đã fit (chạy trên CPU)."""
    torch.manual_seed(seed)
    clf = TorchClassifier("eegnet", SFREQ, seed, device="cpu", F1=F1, D=D)
    clf.device_ = torch.device("cpu")
    clf.mean_ = np.zeros((1, n_channels, 1), np.float32)
    clf.std_ = np.ones((1, n_channels, 1), np.float32)
    clf.n_channels_, clf.n_samples_ = n_channels, n_samples
    clf.net_ = clf._build(n_channels, n_samples).eval()
    return clf


class TestIntegratedGradients(unittest.TestCase):
    def test_completeness(self):
        """Tổng attribution phải bằng F(x) - F(0) (sai số nhỏ do chia bước)."""
        clf = tiny_eegnet(8, 64)
        x = torch.randn(5, 8, 64) * 2
        attr, delta = integrated_gradients(clf.net_, x, steps=128)
        self.assertEqual(attr.shape, x.shape)
        with torch.no_grad():
            change = logit_diff(clf.net_, x) - logit_diff(clf.net_, torch.zeros(1, 8, 64))
        self.assertLess(float((delta.abs() / change.abs().clamp(min=1.0)).max()), 1e-2)

    def test_zero_input_has_zero_attribution(self):
        """Đầu vào bằng baseline thì attribution bằng 0."""
        clf = tiny_eegnet(8, 64)
        attr, _ = integrated_gradients(clf.net_, torch.zeros(2, 8, 64), steps=8)
        self.assertTrue(torch.all(attr == 0))


class TestPatterns(unittest.TestCase):
    def test_haufe_pattern_recovers_planted_source(self):
        """Gài một nguồn có bản đồ da đầu A đã biết; filter w = inv(C) @ a trên temporal filter 1 phải cho lại pattern A.

        Đồng thời kiểm tra: filter 3 thuộc temporal filter 1 (k // D) và pattern được đưa về đơn vị uV.
        """
        rng = np.random.default_rng(0)
        n_ch, n_t, n_trials = 6, 64, 200
        A_uv = np.array([1.0, 3.0, -2.0, 0.5, 4.0, -1.0])  # bản đồ da đầu thật của nguồn (uV)
        std = np.array([1.0, 2.0, 0.5, 3.0, 1.5, 4.0])  # độ lệch chuẩn dùng để chuẩn hoá đầu vào
        source = rng.normal(size=(n_trials, 1, n_t))
        noise = np.einsum("ij,njt->nit", rng.normal(size=(n_ch, n_ch)), rng.normal(size=(n_trials, n_ch, n_t)))
        X_uv = (A_uv[None, :, None] * source + noise).astype(np.float32)

        clf = tiny_eegnet(n_ch, n_t, F1=2, D=2)
        clf.std_ = std.reshape(1, -1, 1).astype(np.float32)
        net = clf.net_
        with torch.no_grad():
            center = net.temporal[0].weight.shape[-1] // 2
            net.temporal[0].weight.zero_()
            net.temporal[0].weight[1, 0, 0, center] = 1.0  # temporal filter 1 giữ nguyên tín hiệu
            net.temporal[0].weight[0, 0, 0, center : center + 2] = torch.tensor([1.0, -1.0])  # filter 0 = đạo hàm
            x_std = X_uv / std[None, :, None]
            cov = np.cov(x_std.transpose(1, 0, 2).reshape(n_ch, -1))
            w = np.linalg.solve(cov, A_uv / std)
            net.spatial.weight[3, 0, :, 0] = torch.tensor(w, dtype=torch.float32)

        out = eegnet_patterns(clf, [X_uv[:120], X_uv[120:]])  # hai "subject"
        self.assertEqual(out["patterns"].shape, (4, n_ch))
        np.testing.assert_array_equal(out["temporal"], [0, 0, 1, 1])
        pattern = out["patterns"][3]
        np.testing.assert_allclose(pattern / np.abs(pattern).max(), A_uv / np.abs(A_uv).max(), atol=1e-3)
        np.testing.assert_allclose(out["filters"][3], w / std, rtol=1e-5)


class TestImportance(unittest.TestCase):
    def test_shape_and_hook_removed(self):
        """Trả về một giá trị cho mỗi filter, và mạng phải trở lại như cũ sau khi chạy."""
        clf = tiny_eegnet(8, 64)
        rng = np.random.default_rng(1)
        x_list = [torch.randn(40, 8, 64) for _ in range(2)]
        y_list = [rng.permutation(np.r_[np.zeros(20, int), np.ones(20, int)]) for _ in range(2)]
        before = clf.net_(x_list[0]).detach()
        drops = filter_importance(clf, x_list, y_list)
        self.assertEqual(drops.shape, (4,))
        self.assertTrue(np.isfinite(drops).all())
        torch.testing.assert_close(clf.net_(x_list[0]).detach(), before)


class TestRoiTable(unittest.TestCase):
    def test_enrichment_and_time_bins(self):
        """Toàn bộ |IG| nằm trong vùng P300 -> enrichment P300 = số kênh / số kênh của vùng, vùng vận động = 0."""
        ch = list(P300_ROI) + list(MOTOR_ROI) + [f"X{i}" for i in range(8)]
        times = 0.003125 + np.arange(103) / SFREQ  # các mẫu của cửa sổ `full` ở 128 Hz
        ig = np.zeros((1, len(ch), len(times)), np.float32)
        ig[0, : len(P300_ROI)] = 1.0
        result = {
            "model": "eegnet",
            "task": "tgt_std",
            "setting": "cross",
            "window": "full",
            "subjects": ["sub-01"],
            "times": times,
            "ch_names": ch,
            "ig_abs": ig,
        }
        table = roi_table(result).set_index("time_bin")
        self.assertAlmostEqual(table.loc["all", "p300_enrichment"], len(ch) / len(P300_ROI))
        self.assertEqual(table.loc["all", "motor_share"], 0.0)
        # Các khoảng thời gian chia hết epoch: tổng tỉ lệ bằng 1
        parts = ["0-250 ms", "250-450 ms", "450-600 ms", "600-800 ms"]
        self.assertAlmostEqual(table.loc[parts, "bin_share"].sum(), 1.0)
        self.assertAlmostEqual(table.loc[["0-250 ms", "250-450 ms", "450-800 ms"], "bin_share"].sum(), 1.0)
        # Mẫu cuối (0.8 s) thuộc khoảng cuối; cửa sổ early không có phần sau 450 ms
        self.assertEqual(time_bins(times)[-1], 3)
        early = dict(result, times=times[times <= 0.45], ig_abs=ig[:, :, times <= 0.45])
        self.assertNotIn("450-800 ms", set(roi_table(early)["time_bin"]))


if __name__ == "__main__":
    unittest.main()
