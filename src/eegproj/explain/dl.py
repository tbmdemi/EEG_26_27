"""Giải thích mô hình DL (R1, protocol mục 9.2-9.3 và 9.5, câu hỏi RQ3).

Ba phương pháp, chạy trên checkpoint EEGNet của R5 và trên đúng các trial mà mô hình đã được test:
  1. Spatial pattern (Haufe et al., 2014): đổi trọng số spatial filter thành bản đồ da đầu đọc được.
  2. Độ quan trọng của từng filter: AUC giảm bao nhiêu khi "tắt" filter đó.
  3. Integrated Gradients (Sundararajan et al., 2017): mỗi kênh tại mỗi thời điểm đóng góp bao nhiêu vào quyết định.

Trước khi giải thích, quyết định của từng checkpoint được tính lại và so với results/predictions/ để chắc chắn
dùng đúng dữ liệu và đúng thứ tự kênh như lúc R5 test.
Hàm vẽ hình nằm ở `explain/plots.py`.
"""

import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

from ..evaluation.splits import epochs_path, format_bids_subject, normalize_subject_id
from ..models.dl import load_checkpoint
from ..models.dl_runner import _git_commit, align_channels, load_dl_subject, window_mask

# Vùng kênh để trả lời câu hỏi ở mục 9.5: mô hình dựa vào P300 (vùng đỉnh và trung tâm) hay vào tín hiệu vận động?
# Người tham gia bấm nút bằng tay phải (docs/02_dataset.md) -> vỏ não vận động bên trái (quanh C3).
P300_ROI = ("Pz", "CPz", "POz", "CP1", "CP2", "P1", "P2")
MOTOR_ROI = ("C3", "C1", "C5", "FC3", "CP3")

# Chia epoch thành 4 khoảng thời gian. 0.45 s là cuối cửa sổ `early` (protocol mục 3); phần sau đó gọi là "muộn".
BIN_EDGES_S = (0.25, 0.45, 0.6)
BIN_LABELS = ("0-250 ms", "250-450 ms", "450-600 ms", "600-800 ms")
LATE_S = 0.45


def time_bins(times):
    """Số thứ tự khoảng thời gian (0..3) của từng mẫu. Mỗi mẫu thuộc đúng một khoảng."""
    return np.digitize(times, BIN_EDGES_S)


# ============================================================================ 1. Các phương pháp giải thích


def logit_diff(net, x):
    """Điểm quyết định của mạng: logit(target) - logit(standard). Lớn hơn 0 nghĩa là đoán 'target'."""
    out = net(x)
    return out[:, 1] - out[:, 0]


def decisions(net, x):
    """logit_diff cho cả một tensor, không tính gradient, trả về mảng numpy."""
    with torch.no_grad():
        return logit_diff(net, x).cpu().numpy()


def integrated_gradients(net, x, steps=64):
    """Integrated Gradients của logit_diff cho một lô trial `x` (trial, kênh, mẫu) đã chuẩn hoá z-score.

    Đi từ baseline 0 (= trung bình từng kênh của tập train) tới x qua `steps` bước, cộng dồn gradient ở mỗi bước
    rồi nhân với x.

    Trả về (attr, delta):
      attr  : cùng kích thước với x, là đóng góp của từng kênh tại từng thời điểm vào quyết định.
      delta : với mỗi trial, tổng attr - (F(x) - F(0)). Theo lý thuyết bằng 0, dùng để kiểm tra.
    """
    net.eval()
    x = x.detach()
    grad_sum = torch.zeros_like(x)
    for k in range(steps):
        alpha = (k + 0.5) / steps  # điểm giữa của bước thứ k
        x_step = (alpha * x).requires_grad_(True)
        grad = torch.autograd.grad(logit_diff(net, x_step).sum(), x_step)[0]
        grad_sum += grad
    attr = x * grad_sum / steps

    with torch.no_grad():
        change = logit_diff(net, x) - logit_diff(net, torch.zeros_like(x))
        delta = attr.sum(dim=(1, 2)) - change
    return attr, delta


def eegnet_patterns(clf, X_list):
    """Spatial filter của EEGNet và pattern tương ứng (bản đồ da đầu, đơn vị uV).

    EEGNet có F1 temporal filter, sau mỗi temporal filter là D spatial filter (spatial filter k dùng đầu ra của
    temporal filter k // D). Trọng số filter w không đọc trực tiếp như bản đồ da đầu được, nên đổi sang pattern
    theo Haufe et al. (2014):

        a = C @ w / (w.T @ C @ w)

    với C là ma trận hiệp phương sai giữa các kênh của đầu ra temporal filter tương ứng.

    X_list : các mảng (trial, kênh, mẫu) đơn vị uV, ví dụ mỗi subject train một mảng.
    Trả về dict: filters (K, kênh), patterns (K, kênh), temporal (K,) = temporal filter của từng spatial filter.
    """
    net = clf.net_
    W = net.spatial.weight.detach()[:, 0, :, 0].double().cpu().numpy()  # (K, kênh)
    n_filters, n_ch = W.shape
    n_temporal = net.temporal[0].out_channels
    temporal_of = np.arange(n_filters) // (n_filters // n_temporal)

    # Cộng dồn tổng và tổng tích để tính hiệp phương sai mà không phải giữ toàn bộ dữ liệu trong bộ nhớ.
    sum_x = np.zeros((n_temporal, n_ch))
    sum_xx = np.zeros((n_temporal, n_ch, n_ch))
    n_points = 0
    with torch.no_grad():
        for X in X_list:
            z = net.temporal(clf.scale(X).unsqueeze(1))  # (trial, F1, kênh, mẫu)
            for f in range(n_temporal):
                zf = z[:, f].permute(1, 0, 2).reshape(n_ch, -1)  # (kênh, trial * mẫu)
                sum_x[f] += zf.sum(1).double().cpu().numpy()
                sum_xx[f] += (zf @ zf.T).double().cpu().numpy()
            n_points += z.shape[0] * z.shape[3]
    mean = sum_x / n_points
    cov = sum_xx / n_points - mean[:, :, None] * mean[:, None, :]

    patterns = np.zeros_like(W)
    for k in range(n_filters):
        c = cov[temporal_of[k]]
        patterns[k] = c @ W[k] / (W[k] @ c @ W[k])

    # Mạng nhận dữ liệu đã chia cho độ lệch chuẩn từng kênh, nên nhân lại để đưa pattern về đơn vị uV.
    std = np.asarray(clf.std_, dtype=float).reshape(-1)
    return {"filters": W / std, "patterns": patterns * std, "temporal": temporal_of}


def filter_importance(clf, x_list, y_list):
    """Độ quan trọng của từng spatial filter = AUC giảm bao nhiêu khi "tắt" filter đó (trung bình qua subject).

    "Tắt" nghĩa là thay đầu ra của filter bằng giá trị trung bình lúc train (running mean của lớp BatchNorm ngay
    sau), nên filter không còn mang thông tin phân biệt giữa các trial.
    x_list, y_list : tensor đã chuẩn hoá và nhãn, mỗi subject test một phần tử.
    """
    net = clf.net_
    train_mean = net.block1[0].running_mean

    def mean_auc():
        return np.mean([roc_auc_score(y, decisions(net, x)) for x, y in zip(x_list, y_list)])

    auc_intact = mean_auc()
    drops = []
    for k in range(net.spatial.out_channels):
        # Hook chạy ngay sau lớp spatial và ghi đè đầu ra của filter k.
        def switch_off(module, inputs, output, k=k):
            output = output.clone()
            output[:, k] = train_mean[k]
            return output

        handle = net.spatial.register_forward_hook(switch_off)
        try:
            drops.append(auc_intact - mean_auc())
        finally:
            handle.remove()  # luôn gỡ hook để mạng trở lại như cũ
    return np.array(drops)


# ============================================================================ 2. Dữ liệu


def load_explain_data(root, config, subjects, ch_names, task="tgt_std", log=print):
    """Đọc dữ liệu đầu vào DL của từng subject (cùng hàm với R5), sắp kênh theo thứ tự trong checkpoint."""
    data = {}
    for s in subjects:
        d = load_dl_subject(root, config, s, task)
        if d["ch_names"] != list(ch_names):
            align_channels(d, ch_names)
            log(f"  sub-{s}: đã sắp lại kênh theo tên cho khớp checkpoint")
        data[s] = d
    return data


def group_info(root, config, subjects, ch_names, sfreq):
    """Info của MNE với vị trí điện cực trung bình qua các subject (giống cách R3 vẽ topomap)."""
    positions = []
    for s in subjects:
        montage = mne.read_epochs(epochs_path(root, config, s), preload=False, verbose="error").get_montage()
        pos = montage.get_positions() if montage is not None else None
        if pos is None or pos["coord_frame"] != "head":
            raise ValueError(f"sub-{s}: không có vị trí điện cực trong hệ toạ độ đầu (head)")
        positions.append([pos["ch_pos"][c] for c in ch_names])
    mean_positions = np.mean(positions, axis=0)
    montage = mne.channels.make_dig_montage(dict(zip(ch_names, mean_positions)), coord_frame="head")
    info = mne.create_info(list(ch_names), sfreq, "eeg")
    info.set_montage(montage)
    return info


def p300_reference_map(data, config):
    """Bản đồ target - standard (uV) của nhóm, trung bình trong cửa sổ topomap của R3 (config erp.topomap_window_s).

    Tính cho từng subject rồi lấy trung bình qua subject.
    """
    start, stop = config["erp"]["topomap_window_s"]
    maps = []
    for d in data.values():
        in_window = (d["times"] >= start) & (d["times"] <= stop)
        difference = d["X"][d["y"] == 1].mean(0) - d["X"][d["y"] == 0].mean(0)  # (kênh, mẫu)
        maps.append(difference[:, in_window].mean(1))
    return np.mean(maps, axis=0)


def median_rt(data):
    """RT trung vị (s) của các trial target: lấy trung vị từng subject rồi trung vị qua subject."""
    per_subject = [np.nanmedian(d["metadata"]["rt"].to_numpy()[d["y"] == 1]) for d in data.values()]
    return float(np.median(per_subject))


# ============================================================================ 3. Chạy trên các checkpoint


def checkpoint_paths(root, config, setting, window, task, model="eegnet"):
    """Danh sách file checkpoint của một tổ hợp (do scripts/04_train_dl.py lưu)."""
    folder = Path(root) / config["paths"]["checkpoints"] / model / f"{setting}_{window}_{task}"
    return sorted(folder.glob("*.pt"))


def check_checkpoint(path, extra, ch_names, times, window, task):
    """Báo lỗi nếu checkpoint không khớp với dữ liệu đang dùng."""
    if extra["window"] != window or extra["task"] != task:
        raise ValueError(f"{path}: được train cho {extra['window']}/{extra['task']}, không phải {window}/{task}.")
    if extra["ch_names"] != ch_names:
        raise ValueError(f"{path}: thứ tự kênh khác với dữ liệu đã đọc.")
    if not np.allclose(extra["times"], times, atol=1e-6):
        raise ValueError(f"{path}: thời điểm mẫu khác với dữ liệu đã đọc (đổi tần số lấy mẫu hoặc cửa sổ?).")


def test_trials(extra, data):
    """Danh sách (subject, chỉ số trial) mà checkpoint này đã được test."""
    if extra["setting"] == "within":
        # within: mỗi checkpoint test một đoạn trial của đúng một subject
        return [(normalize_subject_id(extra["subject"]), np.asarray(extra["test_index"]))]
    # cross: mỗi checkpoint test toàn bộ trial của vài subject
    subjects = [normalize_subject_id(s) for s in extra["test_subjects"]]
    return [(s, np.arange(len(data[s]["y"]))) for s in subjects]


def pattern_table(clf, extra, data, tmask, x_list, y_list, ref_map):
    """Spatial pattern + độ quan trọng của 16 filter cho một checkpoint cross. Mỗi dòng = (filter, kênh)."""
    train_subjects = [normalize_subject_id(s) for s in extra["train_subjects"]]
    # Dùng generator để không phải giữ cùng lúc dữ liệu của khoảng 28 subject train.
    pat = eegnet_patterns(clf, (data[s]["X"][:, :, tmask] for s in train_subjects))
    importance = filter_importance(clf, x_list, y_list)

    rank = np.empty(len(importance), dtype=int)
    rank[np.argsort(-importance)] = np.arange(1, len(importance) + 1)  # 1 = quan trọng nhất

    ch_names = extra["ch_names"]
    tables = []
    for k in range(len(importance)):
        p = pat["patterns"][k]
        r = np.corrcoef(p, ref_map)[0, 1]
        sign = 1.0 if r >= 0 else -1.0  # dấu của pattern là tuỳ ý -> lật cho cùng chiều với bản đồ P300
        tables.append(
            pd.DataFrame(
                {
                    "fold": int(extra["fold"]),
                    "filter": k,
                    "temporal_filter": int(pat["temporal"][k]),
                    "rank": int(rank[k]),
                    "delta_auc": importance[k],
                    "r_p300": abs(r),
                    "peak_channel": ch_names[int(np.argmax(sign * p))],  # kênh có giá trị dương lớn nhất
                    "channel": ch_names,
                    "pattern": sign * p,
                    "filter_weight": sign * pat["filters"][k],
                }
            )
        )
    return pd.concat(tables, ignore_index=True)


def explain_run(root, config, data, setting, window, task="tgt_std", steps=64, device="auto", ref_map=None):
    """Giải thích mọi checkpoint EEGNet của một tổ hợp setting, window và task.

    Với mỗi checkpoint:
      - tính lại quyết định trên các trial test (để so với results/predictions/),
      - chạy IG trên các trial target được đoán đúng,
      - riêng `cross`: tính spatial pattern và độ quan trọng của 16 filter (cần `ref_map`).

    Trả về dict:
      subjects, times, ch_names
      n_targets, n_correct : số trial target và số target đoán đúng của từng subject
      ig_mean, ig_abs      : (subject, kênh, mẫu) trung bình IG có dấu / trị tuyệt đối qua các target đoán đúng
      decisions            : bảng quyết định tính lại (subject, fold, onset_s, decision)
      ig_rel_error         : sai số completeness lớn nhất của IG, tương đối so với thay đổi của quyết định
      patterns             : bảng spatial pattern (chỉ với cross), nếu không thì None
    """
    paths = checkpoint_paths(root, config, setting, window, task)
    if not paths:
        raise FileNotFoundError(
            f"Không có checkpoint EEGNet cho {setting}/{window}/{task}; chạy scripts/04_train_dl.py."
        )
    subjects = sorted(data)
    first = data[subjects[0]]
    tmask = window_mask(first["times"], config, window)
    times = first["times"][tmask]

    attr_by_subject = {s: [] for s in subjects}  # IG của các trial target đoán đúng
    n_targets = dict.fromkeys(subjects, 0)
    decision_tables, pattern_tables = [], []
    worst_ig_error = 0.0

    for path in paths:
        clf, info = load_checkpoint(path, device=device)
        extra = info["extra"]
        check_checkpoint(path, extra, first["ch_names"], times, window, task)
        net = clf.net_
        zero_input = torch.zeros((1, len(first["ch_names"]), len(times)), device=clf.device_)
        f_zero = decisions(net, zero_input)[0]

        x_list, y_list = [], []
        for sub, idx in test_trials(extra, data):
            X = data[sub]["X"][idx][:, :, tmask]
            y = data[sub]["y"][idx]
            x = clf.scale(X)  # chuẩn hoá giống lúc train
            dec = decisions(net, x)
            decision_tables.append(
                pd.DataFrame(
                    {
                        "subject": format_bids_subject(sub),
                        "fold": int(extra["fold"]),
                        "onset_s": data[sub]["metadata"]["onset_s"].to_numpy()[idx],
                        "decision": dec,
                    }
                )
            )

            # IG chỉ trên các trial target mà mô hình đoán đúng
            n_targets[sub] += int((y == 1).sum())
            correct = np.flatnonzero((y == 1) & (dec > 0))
            if len(correct) > 0:
                attr, delta = integrated_gradients(net, x[correct], steps)
                error = np.abs(delta.cpu().numpy()) / np.maximum(np.abs(dec[correct] - f_zero), 1.0)
                worst_ig_error = max(worst_ig_error, float(error.max()))
                attr_by_subject[sub].append(attr.cpu().numpy())

            x_list.append(x)
            y_list.append(y)

        if setting == "cross":
            pattern_tables.append(pattern_table(clf, extra, data, tmask, x_list, y_list, ref_map))

    # Gộp IG theo subject
    ig_mean, ig_abs, n_correct = [], [], []
    for s in subjects:
        if not attr_by_subject[s]:
            raise ValueError(f"{setting}/{window}: sub-{s} không có trial target nào được đoán đúng.")
        a = np.concatenate(attr_by_subject[s]).astype(np.float64)  # (trial, kênh, mẫu)
        ig_mean.append(a.mean(0))
        ig_abs.append(np.abs(a).mean(0))
        n_correct.append(len(a))

    return {
        "model": "eegnet",
        "setting": setting,
        "window": window,
        "task": task,
        "subjects": [format_bids_subject(s) for s in subjects],
        "times": times,
        "ch_names": list(first["ch_names"]),
        "n_targets": np.array([n_targets[s] for s in subjects]),
        "n_correct": np.array(n_correct),
        "ig_mean": np.array(ig_mean, dtype=np.float32),
        "ig_abs": np.array(ig_abs, dtype=np.float32),
        "decisions": pd.concat(decision_tables, ignore_index=True),
        "ig_rel_error": worst_ig_error,
        "patterns": pd.concat(pattern_tables, ignore_index=True) if pattern_tables else None,
    }


def run_name(result):
    return f"{result['model']}_{result['setting']}_{result['window']}_{result['task']}"


def check_decisions(root, result, atol=1e-2):
    """So quyết định tính lại với results/predictions/ của R5. Báo lỗi nếu trial hoặc giá trị không khớp.

    Trả về độ lệch tuyệt đối lớn nhất.
    """
    name = run_name(result)
    ours = result["decisions"].copy()
    saved = pd.read_csv(Path(root) / "results" / "predictions" / f"{name}.csv.gz")
    # Ghép theo thời điểm trial (làm tròn tới ms) để không phụ thuộc thứ tự dòng.
    for table in (ours, saved):
        table["onset_ms"] = (table["onset_s"] * 1000).round().astype(int)
    keys = ["subject", "fold", "onset_ms"]
    merged = ours.merge(saved[keys + ["decision"]], on=keys, suffixes=("", "_saved"), validate="one_to_one")
    if not len(merged) == len(ours) == len(saved):
        raise ValueError(f"{name}: số trial test tính lại ({len(ours)}) khác file predictions ({len(saved)}).")
    diff = float(np.max(np.abs(merged["decision"] - merged["decision_saved"])))
    if diff > atol:
        raise ValueError(f"{name}: quyết định tính lại lệch so với predictions (lớn nhất {diff:.4f}).")
    return diff


# ============================================================================ 4. Bảng tổng hợp và lưu file


def roi_table(result):
    """Bảng |IG| theo vùng kênh và khoảng thời gian. Mỗi dòng = (subject, khoảng thời gian).

    bin_share        : phần |IG| của subject nằm trong khoảng thời gian đó.
    <vùng>_share      : trong khoảng đó, phần |IG| nằm ở vùng kênh.
    <vùng>_enrichment : <vùng>_share chia cho tỉ lệ số kênh của vùng.
                        1 = vùng nhận |IG| bằng một kênh trung bình, > 1 = mô hình dựa vào vùng này nhiều hơn.
    """
    times, ch = result["times"], result["ch_names"]
    bin_of = time_bins(times)

    periods = {"all": np.ones(len(times), dtype=bool)}
    for b, label in enumerate(BIN_LABELS):
        periods[label] = bin_of == b
    periods["450-800 ms"] = times >= LATE_S
    periods = {label: mask for label, mask in periods.items() if mask.any()}  # bỏ khoảng nằm ngoài cửa sổ

    rois = {"p300": [ch.index(c) for c in P300_ROI], "motor": [ch.index(c) for c in MOTOR_ROI]}
    rows = []
    for i, subject in enumerate(result["subjects"]):
        a = result["ig_abs"][i].astype(np.float64)  # (kênh, mẫu)
        for label, mask in periods.items():
            part = a[:, mask]
            row = {
                "model": result["model"],
                "task": result["task"],
                "setting": result["setting"],
                "window": result["window"],
                "subject": subject,
                "time_bin": label,
                "bin_share": part.sum() / a.sum(),
            }
            for name, idx in rois.items():
                share = part[idx].sum() / part.sum()
                row[f"{name}_share"] = share
                row[f"{name}_enrichment"] = share / (len(idx) / len(ch))
            rows.append(row)
    return pd.DataFrame(rows)


def summarise_roi(table):
    """Trung vị [IQR] qua subject của bảng roi_table, mỗi dòng = (setting, window, khoảng thời gian).

    p_p300_vs_motor: kiểm định Wilcoxon theo subject giữa enrichment của vùng P300 và vùng vận động
    (chỉ để mô tả, chưa hiệu chỉnh nhiều phép so sánh).
    """
    keys = ["model", "task", "setting", "window", "time_bin"]
    value_cols = ["bin_share", "p300_share", "p300_enrichment", "motor_share", "motor_enrichment"]
    rows = []
    for key, group in table.groupby(keys, sort=False):
        row = dict(zip(keys, key))
        row["n_subjects"] = len(group)
        for col in value_cols:
            row[f"{col}_median"] = group[col].median()
            row[f"{col}_q25"] = group[col].quantile(0.25)
            row[f"{col}_q75"] = group[col].quantile(0.75)
        row["p_p300_vs_motor"] = wilcoxon(group["p300_enrichment"], group["motor_enrichment"]).pvalue
        rows.append(row)
    return pd.DataFrame(rows)


def save_run(result, out_dir):
    """Lưu IG theo subject (.npz) và, với cross, bảng spatial pattern (.csv). Trả về tên lần chạy."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = run_name(result)
    np.savez_compressed(
        out_dir / f"{name}_saliency.npz",
        subjects=np.array(result["subjects"]),
        ch_names=np.array(result["ch_names"]),
        times=result["times"],
        n_targets=result["n_targets"],
        n_correct=result["n_correct"],
        ig_mean=result["ig_mean"],
        ig_abs=result["ig_abs"],
    )
    if result["patterns"] is not None:
        result["patterns"].to_csv(out_dir / f"{name}_patterns.csv", index=False)
    return name


def write_manifest(root, out_dir, checks, steps, name="manifest.json"):
    """Ghi kết quả các bước kiểm tra và commit hiện tại vào results/explain/<name>."""
    path = Path(out_dir) / name
    content = {"git_commit": _git_commit(root), "ig_steps": steps, "runs": checks}
    path.write_text(json.dumps(content, indent=2), encoding="utf-8")
    return path
