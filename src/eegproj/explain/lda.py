"""Giải thích mô hình ML truyền thống `lda_tw` của R4 (protocol mục 9.4, câu hỏi RQ3).

Đổi trọng số LDA thành activation pattern (Haufe et al., 2014) để so sánh công bằng với spatial pattern và
Integrated Gradients của EEGNet (explain/dl.py). Chạy cross-subject trên đúng các fold mà R4 và R5 dùng
(results/cv_cross_subject_folds.csv).

Mỗi subject được lấy trung bình theo bin 50 ms ngay khi đọc (TimeWindowAverager của R4, không học gì từ dữ liệu),
nên chỉ phải giữ khoảng 10 MB mỗi subject thay vì toàn bộ epoch của khoảng 28 subject train.
"""

import mne
import numpy as np
import pandas as pd

from ..evaluation.metrics import compute_binary_metrics
from ..evaluation.splits import format_bids_subject, load_task_epochs
from ..models.classical import build_lda_tw
from ..models.runner import _align_eeg_channels
from .dl import MOTOR_ROI, P300_ROI


def load_features(root, config, subjects, window, task="tgt_std", log=print):
    """Đọc từng subject, sắp kênh theo tên (giống R4), lấy trung bình theo bin 50 ms.

    Trả về dict: Z {subject: (trial, kênh * bin) uV}, y {subject: nhãn}, ch_names, bin_edges (s), times (s).
    """
    Z, y, ch_names, times, averager = {}, {}, None, None, None
    for s in subjects:
        epochs, labels = load_task_epochs(root, config, s, task, window)
        names = [epochs.ch_names[i] for i in mne.pick_types(epochs.info, eeg=True)]
        X = epochs.get_data(picks="eeg") * 1e6
        if ch_names is None:
            ch_names, times = names, epochs.times.copy()
            averager = build_lda_tw(times, config).named_steps["time_windows"].fit(X)
        elif not np.allclose(epochs.times, times):
            raise ValueError(f"sub-{s}: thời điểm mẫu khác subject đầu tiên.")
        Z[s] = averager.transform(_align_eeg_channels(X, names, ch_names, s))
        y[s] = labels
    log(f"  {window}: {len(Z)} subject, {Z[subjects[0]].shape[1]} đặc trưng ({len(ch_names)} kênh)")
    return {"Z": Z, "y": y, "ch_names": ch_names, "bin_edges": averager.bin_edges_.copy(), "times": times}


def lda_pattern(model, Z):
    """Activation pattern a = cov(Z, s) / var(s), với s = Z @ w (Haufe et al., 2014, công thức 7), đơn vị uV.

    Cùng công thức với `extract_lda_haufe_patterns` của R4 (có test so khớp), nhưng nhận thẳng đặc trưng Z.
    model : Pipeline(scaler, clf) đã fit trên Z.
    """
    scaler, clf = model.named_steps["scaler"], model.named_steps["clf"]
    w = clf.coef_.ravel() / np.where(scaler.scale_ > 0, scaler.scale_, 1.0)  # trọng số trên Z chưa chuẩn hoá
    Zc = Z - Z.mean(0)
    s = Zc @ w
    return Zc.T @ s / (len(Z) - 1) / np.var(s, ddof=1)


def p300_reference_map(features, config):
    """Bản đồ target - standard (uV) trong cửa sổ topomap của R3, tính từ các bin nằm trong cửa sổ đó.

    Tính cho từng subject rồi lấy trung bình qua subject (giống explain.dl.p300_reference_map).
    """
    start, stop = config["erp"]["topomap_window_s"]
    edges = features["bin_edges"]
    in_window = (edges[:-1] >= start - 1e-9) & (edges[1:] <= stop + 1e-9)
    n_ch = len(features["ch_names"])
    maps = []
    for s, Z in features["Z"].items():
        y = features["y"][s]
        diff = (Z[y == 1].mean(0) - Z[y == 0].mean(0)).reshape(n_ch, -1)  # (kênh, bin)
        maps.append(diff[:, in_window].mean(1))
    return np.mean(maps, axis=0)


def lda_cross(features, folds, config):
    """Fit `lda_tw` trên từng fold cross-subject, tính pattern và điểm trên subject test.

    Trả về (patterns (fold, kênh, bin) uV, bảng metric mỗi subject test).
    """
    n_ch = len(features["ch_names"])
    patterns, rows = [], []
    for fold, train_subs, test_subs in folds:
        Z_train = np.concatenate([features["Z"][s] for s in train_subs])
        y_train = np.concatenate([features["y"][s] for s in train_subs])
        # Bỏ bước time_windows vì Z đã lấy trung bình theo bin; phần còn lại giống hệt pipeline của R4.
        model = build_lda_tw(features["times"], config)[1:].fit(Z_train, y_train)
        patterns.append(lda_pattern(model, Z_train).reshape(n_ch, -1))
        for s in test_subs:
            Z, y = features["Z"][s], features["y"][s]
            m = compute_binary_metrics(y, model.predict_proba(Z)[:, 1], model.predict(Z))
            rows.append({"subject": format_bids_subject(s), "fold": fold, **m})
    return np.array(patterns), pd.DataFrame(rows)


def pattern_table(patterns, features, ref_map, window):
    """Bảng mô tả theo từng bin 50 ms (pattern trung bình qua fold).

    r_p300       : tương quan giữa bản đồ pattern của bin và bản đồ P300 tham chiếu.
    p300_uv/motor_uv : pattern trung bình trên vùng kênh P300 / vận động trái (uV, có dấu).
    abs_share_*  : phần |pattern| của bin nằm ở vùng kênh, chia cho tỉ lệ số kênh (1 = bằng một kênh trung bình).
    """
    ch = features["ch_names"]
    edges = features["bin_edges"]
    roi = {"p300": [ch.index(c) for c in P300_ROI], "motor": [ch.index(c) for c in MOTOR_ROI]}
    mean = patterns.mean(0)  # (kênh, bin)
    rows = []
    for b in range(mean.shape[1]):
        p = mean[:, b]
        row = {"window": window, "bin_start_ms": edges[b] * 1000, "bin_end_ms": edges[b + 1] * 1000}
        row["r_p300"] = np.corrcoef(p, ref_map)[0, 1]
        for name, idx in roi.items():
            row[f"{name}_uv"] = p[idx].mean()
            row[f"{name}_enrichment"] = np.abs(p[idx]).sum() / np.abs(p).sum() / (len(idx) / len(ch))
        rows.append(row)
    return pd.DataFrame(rows)
