"""Hình F5 cho phần giải thích mô hình (R1). Chữ trên hình để tiếng Anh vì report viết bằng tiếng Anh.

F5a: spatial pattern của EEGNet (cross-subject).
F5b: Integrated Gradients theo kênh và thời gian.
F5c: activation pattern của LDA (lda_tw, cross-subject) để so sánh với EEGNet.
"""

import matplotlib.pyplot as plt
import mne
import numpy as np

from .dl import BIN_LABELS, LATE_S, MOTOR_ROI, P300_ROI, time_bins


def task_classes(config, task):
    """Tên lớp dương và lớp âm của bài toán (config classification.tasks), ví dụ ('target', 'standard')."""
    classes = config["classification"]["tasks"][task]
    return classes["positive"], classes["negative"]


def plot_spatial_patterns(results, info, ref_map, config, top=2):
    """F5a: bản đồ P300 tham chiếu, biểu đồ độ quan trọng của mọi filter, và `top` filter quan trọng nhất mỗi fold.

    results : {window: kết quả explain_run của setting cross}
    """
    font = config["erp"]["font_size"]
    plt.rcParams.update({"font.size": font})
    windows = list(results)
    folds = sorted(results[windows[0]]["patterns"]["fold"].unique())
    n_rows = top * len(windows)

    fig = plt.figure(figsize=(2.3 * (len(folds) + 1.4), 2.35 * n_rows + 0.8), layout="constrained")
    grid = fig.add_gridspec(n_rows, len(folds) + 1, width_ratios=[1.4] + [1] * len(folds))

    # Cột trái, nửa trên: bản đồ ERP lớp dương - lớp âm (task chính: target - standard) để so sánh
    pos, neg = task_classes(config, results[windows[0]]["task"])
    ax = fig.add_subplot(grid[: n_rows // 2, 0])
    limit = np.max(np.abs(ref_map))
    image, _ = mne.viz.plot_topomap(ref_map, info, axes=ax, show=False, cmap="RdBu_r", vlim=(-limit, limit))
    fig.colorbar(image, ax=ax, shrink=0.6, label="µV", location="bottom")
    start, stop = config["erp"]["topomap_window_s"]
    ax.set_title(f"ERP {pos} - {neg}\n{start * 1000:g}-{stop * 1000:g} ms (reference)", fontsize=font)

    # Cột trái, nửa dưới: mỗi điểm là một filter (độ giống P300 và độ quan trọng)
    ax = fig.add_subplot(grid[n_rows // 2 :, 0])
    for window, marker in zip(windows, "os"):
        one_row_per_filter = results[window]["patterns"].drop_duplicates(["fold", "filter"])
        ax.scatter(
            one_row_per_filter["r_p300"], one_row_per_filter["delta_auc"], s=18, marker=marker, alpha=0.7, label=window
        )
    ax.axhline(0, color="gray", lw=0.6)
    ax.set(xlabel="|r| with P300 map", ylabel="AUC drop when ablated", xlim=(0, 1))
    ax.set_title("All filters (16 per fold)", fontsize=font)
    ax.legend(fontsize=font - 1, title="window", title_fontsize=font - 1)

    # Các cột còn lại: pattern của filter quan trọng nhất (#1, #2, ...) ở từng fold
    topo_axes = []
    for wi, window in enumerate(windows):
        patterns = results[window]["patterns"]
        for rank in range(1, top + 1):
            row = wi * top + rank - 1
            for col, fold in enumerate(folds):
                ax = fig.add_subplot(grid[row, col + 1])
                chosen = patterns[(patterns["fold"] == fold) & (patterns["rank"] == rank)]
                chosen = chosen.set_index("channel").loc[info.ch_names]
                values = chosen["pattern"].to_numpy()
                values = values / np.max(np.abs(values))  # chuẩn hoá về [-1, 1] để so hình dạng
                image, _ = mne.viz.plot_topomap(values, info, axes=ax, show=False, cmap="RdBu_r", vlim=(-1, 1))
                ax.set_title(
                    f"{window}, fold {fold}, #{rank}\n"
                    f"AUC drop {chosen['delta_auc'].iloc[0]:.3f}, |r| {chosen['r_p300'].iloc[0]:.2f}",
                    fontsize=font - 1,
                )
                topo_axes.append(ax)
    fig.colorbar(image, ax=topo_axes, shrink=0.5, label="Pattern (normalised, a.u.)")

    fig.suptitle(
        f"F5a: EEGNet spatial patterns (cross-subject, task {pos} vs {neg})\n"
        "Top filters by importance per fold; polarity aligned with the P300 map; mean electrode positions",
        fontsize=font + 1,
    )
    return fig


def plot_saliency(results, info, config, rt_median=None):
    """F5b: với mỗi cửa sổ, vẽ bản đồ kênh theo thời gian, đường thời gian theo vùng kênh, và topomap theo khoảng.

    Dùng |IG| (độ lớn đóng góp). IG có dấu của các kênh cạnh nhau hay ngược dấu nhau nên bản đồ có dấu rất khó
    đọc; IG có dấu vẫn được lưu trong file .npz (`ig_mean`).
    results : {window: kết quả explain_run của cùng một setting}
    """
    font = config["erp"]["font_size"]
    plt.rcParams.update({"font.size": font})
    windows = list(results)
    n_bins = len(BIN_LABELS)
    fig = plt.figure(figsize=(14, 6.2 * len(windows)), layout="constrained")
    grid = fig.add_gridspec(2 * len(windows), n_bins, height_ratios=[1.5, 1] * len(windows))

    # Sắp kênh từ trước ra sau đầu theo toạ độ y; chỉ ghi tên vài kênh đường giữa cho dễ đọc.
    y_position = np.array([info["chs"][i]["loc"][1] for i in range(len(info.ch_names))])
    order = np.argsort(-y_position)
    landmarks = ("Fz", "Cz", "CPz", "Pz", "POz", "Oz")

    for wi, window in enumerate(windows):
        res = results[window]
        ch = res["ch_names"]
        times_ms = res["times"] * 1000
        ig = res["ig_abs"].astype(np.float64)  # (subject, kênh, mẫu)
        group = ig.mean(0)  # trung bình qua subject
        n_subjects = len(res["subjects"])

        # 1) Bản đồ kênh theo thời gian
        ax = fig.add_subplot(grid[2 * wi, :2])
        half_step = (times_ms[1] - times_ms[0]) / 2
        image = ax.imshow(
            group[order],
            aspect="auto",
            cmap="magma_r",
            vmin=0,
            vmax=np.percentile(group, 99.5),
            extent=(times_ms[0] - half_step, times_ms[-1] + half_step, len(ch) - 0.5, -0.5),
            interpolation="nearest",
        )
        row_of = {ch[j]: row for row, j in enumerate(order)}
        ax.set_yticks([row_of[c] for c in landmarks], landmarks)
        ax.set(xlabel="Time from stimulus onset (ms)", ylabel="Channel (anterior to posterior)")
        pos, neg = task_classes(config, res["task"])
        ax.set_title(f"{window}: mean |IG| over correct {pos} trials ({n_subjects} subjects)", fontsize=font)
        # Colorbar đặt dưới heatmap: đặt bên phải sẽ kéo giãn khe giữa cột 2 và 3 của hàng topomap
        fig.colorbar(image, ax=ax, label="|IG| (logit per sample)", location="bottom", shrink=0.8, aspect=50)

        # 2) Đường thời gian: vùng P300, vùng vận động trái và trung bình mọi kênh (trung bình và SEM qua subject)
        ax = fig.add_subplot(grid[2 * wi, 2:])
        lines = [
            (f"P300 ROI ({', '.join(P300_ROI[:3])}...)", P300_ROI, "tab:orange"),
            (f"Left motor ROI ({', '.join(MOTOR_ROI[:3])}...)", MOTOR_ROI, "tab:purple"),
            ("All channels", ch, "gray"),
        ]
        for label, channels, color in lines:
            idx = [ch.index(c) for c in channels]
            per_subject = ig[:, idx].mean(1)
            mean = per_subject.mean(0)
            sem = per_subject.std(0, ddof=1) / np.sqrt(n_subjects)
            ax.plot(times_ms, mean, color=color, label=label)
            ax.fill_between(times_ms, mean - sem, mean + sem, color=color, alpha=0.2)
        ax.axvline(LATE_S * 1000, color="gray", ls="--", lw=0.8, label="end of early window")
        if rt_median is not None:
            ax.axvline(rt_median * 1000, color="k", ls=":", lw=1, label=f"median RT {rt_median * 1000:.0f} ms")
        ax.set_xlim(0, 800)
        ax.set_ylim(bottom=0)
        ax.set(xlabel="Time from stimulus onset (ms)", ylabel="|IG| per channel (logit, mean +/- SEM)")
        ax.set_title(f"{window}: ROI time courses", fontsize=font)
        ax.legend(fontsize=font - 2, loc="upper left")

        # 3) Topomap |IG| trung bình trong từng khoảng thời gian (cùng thang màu trong một cửa sổ)
        bin_of = time_bins(res["times"])
        maps = [group[:, bin_of == b].mean(1) if (bin_of == b).any() else None for b in range(n_bins)]
        vmax = max(m.max() for m in maps if m is not None)
        topo_axes = []
        for b, (label, values) in enumerate(zip(BIN_LABELS, maps)):
            ax = fig.add_subplot(grid[2 * wi + 1, b])
            topo_axes.append(ax)  # gồm cả ô trống, để colorbar nằm ở mép phải thay vì chen vào giữa hàng
            if values is None:  # khoảng nằm ngoài cửa sổ `early`
                ax.axis("off")
                ax.text(0.5, 0.5, "outside window", ha="center", va="center", color="gray")
                continue
            image, _ = mne.viz.plot_topomap(values, info, axes=ax, show=False, cmap="magma_r", vlim=(0, vmax))
            ax.set_title(f"{window}: |IG| {label}", fontsize=font)
        fig.colorbar(image, ax=topo_axes, shrink=0.7, label="mean |IG| (logit)")

    setting = results[windows[0]]["setting"]
    fig.suptitle(
        f"F5b: EEGNet Integrated Gradients, {setting}-subject, {pos} vs {neg}\n"
        "|IG| = how much each channel/time point moved the decision; baseline = training mean; mean electrode positions",
        fontsize=font + 1,
    )
    return fig


def plot_lda_patterns(results, info, config, task="tgt_std"):
    """F5c: với mỗi cửa sổ, đường thời gian của pattern LDA ở vùng P300 và vùng vận động trái, và topomap theo khoảng.

    results : {window: {"patterns": (fold, kênh, bin) uV, "bin_edges": (bin + 1,) s}}
    Dùng cùng các khoảng thời gian với F5b để dễ đặt cạnh nhau.
    """
    font = config["erp"]["font_size"]
    plt.rcParams.update({"font.size": font})
    windows = list(results)
    n_bins = len(BIN_LABELS)
    ch = info.ch_names
    fig = plt.figure(figsize=(16, 3.6 * len(windows) + 0.8), layout="constrained")
    grid = fig.add_gridspec(len(windows), n_bins + 1, width_ratios=[1.8] + [1] * n_bins)

    for wi, window in enumerate(windows):
        patterns = results[window]["patterns"]  # (fold, kênh, bin)
        edges = results[window]["bin_edges"]
        centers_ms = (edges[:-1] + edges[1:]) / 2 * 1000

        # 1) Đường thời gian: trung bình kênh trong vùng, đường = trung bình qua fold, vùng tô = min..max qua fold
        ax = fig.add_subplot(grid[wi, 0])
        for label, channels, color in [
            (f"P300 ROI ({', '.join(P300_ROI[:3])}...)", P300_ROI, "tab:orange"),
            (f"Left motor ROI ({', '.join(MOTOR_ROI[:3])}...)", MOTOR_ROI, "tab:purple"),
        ]:
            per_fold = patterns[:, [ch.index(c) for c in channels]].mean(1)
            ax.plot(centers_ms, per_fold.mean(0), marker="o", ms=3, color=color, label=label)
            ax.fill_between(centers_ms, per_fold.min(0), per_fold.max(0), color=color, alpha=0.2)
        ax.axhline(0, color="gray", lw=0.6)
        ax.axvline(LATE_S * 1000, color="gray", ls="--", lw=0.8, label="end of early window")
        ax.set_xlim(0, 800)
        ax.set(xlabel="Time from stimulus onset (ms)", ylabel="Pattern (µV)")
        ax.set_title(f"{window}: ROI pattern per 50 ms bin (5 folds)", fontsize=font)
        ax.legend(fontsize=font - 2, loc="center left")

        # 2) Topomap pattern trung bình qua fold, gộp các bin 50 ms có tâm nằm trong từng khoảng (cùng thang màu)
        mean = patterns.mean(0)
        bin_of = time_bins(centers_ms / 1000)
        maps = [mean[:, bin_of == b].mean(1) if (bin_of == b).any() else None for b in range(n_bins)]
        limit = max(np.abs(m).max() for m in maps if m is not None)
        topo_axes = []
        for b, (label, values) in enumerate(zip(BIN_LABELS, maps)):
            ax = fig.add_subplot(grid[wi, b + 1])
            topo_axes.append(ax)  # gồm cả ô trống, để colorbar nằm ở mép phải thay vì chen vào giữa hàng
            if values is None:  # khoảng nằm ngoài cửa sổ `early`
                ax.axis("off")
                ax.text(0.5, 0.5, "outside window", ha="center", va="center", color="gray")
                continue
            image, _ = mne.viz.plot_topomap(values, info, axes=ax, show=False, cmap="RdBu_r", vlim=(-limit, limit))
            ax.set_title(f"{window}: {label}", fontsize=font)
        fig.colorbar(image, ax=topo_axes, shrink=0.7, label="LDA pattern (µV)")

    pos, neg = task_classes(config, task)
    fig.suptitle(
        f"F5c: LDA activation patterns (lda_tw, cross-subject, {pos} vs {neg})\n"
        "Haufe patterns of the 50 ms bin features, mean over 5 folds; mean electrode positions",
        fontsize=font + 1,
    )
    return fig
