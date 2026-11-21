"""R1: Hình F1, sơ đồ pipeline của cả nhóm (protocol mục 10). Chữ trên hình để tiếng Anh vì report viết bằng tiếng Anh.

Cách chạy:
    python scripts/07_plot_pipeline.py     # -> figures/F1_pipeline.{png,pdf}

Nội dung lấy từ docs/03_analysis_protocol.md và configs/config.yaml; sửa protocol thì sửa cả chữ ở đây.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # chỉ lưu hình ra file, không mở cửa sổ
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from eegproj.config import find_root, load_config

# Màu theo vai trò trong nhóm (docs/04_roles_and_tasks.md); "all" = quyết định chung trong protocol
COLORS = {"R1": "#e8e0f3", "R2": "#dbe9f6", "R3": "#e6f2e0", "R4": "#fde9d4", "R5": "#f9dcdc", "all": "#eeeeee"}

# ((x, y, rộng, cao), vai trò, tiêu đề, nội dung); khung vẽ 16 x 9.2, hàng 1 ở trên, hàng 3 (câu hỏi) ở dưới
BOXES = {
    "data": (
        (0.2, 7.0, 3.6, 1.8),
        "R2",
        "Data",
        "Nencki-Symfonia (ds004621)\n42 participants, 127 EEG channels\n3-stimulus visual oddball",
    ),
    "behaviour": (
        (4.2, 7.0, 3.6, 1.8),
        "R2",
        "Behavioural filter",
        (
            "Keep correct trials only\n(target: response in 100-1400 ms;\nstandard, distractor: none)\n"
            "Exclude sub-02, 14, 17"
        ),
    ),
    "preproc": (
        (8.2, 7.0, 3.6, 1.8),
        "R2",
        "Preprocessing (MNE)",
        (
            "0.1-40 Hz band-pass\nbad channels interpolated\naverage reference, ICA + ICLabel\n"
            "(eye, muscle), resample 250 Hz"
        ),
    ),
    "epochs": (
        (12.2, 7.0, 3.6, 1.8),
        "R2",
        "Epochs",
        "-200 to 800 ms, reject > 150 µV\n35 usable subjects\n20,600 epochs (data v2)",
    ),
    "design": (
        (0.2, 3.2, 3.6, 3.0),
        "all",
        "Tasks and windows",
        (
            "Main: target vs standard\n\nControls:\ndistractor vs standard\ntarget vs distractor\n\n"
            "full 0-800 ms\nearly 0-450 ms (before\nmost button presses)"
        ),
    ),
    "classical": (
        (4.2, 4.65, 3.6, 1.55),
        "R4",
        "Classical ML",
        "lda_tw: 50 ms bins + shrinkage LDA\nxdawn_rg: xDAWN + Riemannian\ntangent space + logistic reg.",
    ),
    "dl": ((4.2, 3.2, 3.6, 1.25), "R5", "Deep learning", "EEGNet, ShallowConvNet\n(128 Hz, early stopping)"),
    "eval": (
        (8.2, 3.2, 3.6, 3.0),
        "all",
        "Evaluation",
        (
            "within: 5 contiguous blocks\nper subject\n\ncross: 5 folds by subject\n(same folds for all models)\n\n"
            "fit on train only, class weights\nbalanced accuracy, ROC-AUC"
        ),
    ),
    "erp": (
        (12.2, 3.2, 3.6, 3.0),
        "R3",
        "ERP reference",
        "Grand-average ERP\nat Pz, CPz, Cz\n\ntarget - standard\ndifference wave\nand topomaps (F2)",
    ),
    "rq1": ((0.2, 0.3, 4.9, 1.6), "all", "RQ1: best single-trial result", "per model, setting and window (F3)"),
    "rq2": (
        (5.55, 0.3, 4.9, 1.6),
        "R4",
        "RQ2: DL vs classical",
        "classical-best vs DL-best per subject\nWilcoxon + Holm (F4)",
    ),
    "rq3": (
        (10.9, 0.3, 4.9, 1.6),
        "R1",
        "RQ3: what do the models use?",
        "LDA patterns, EEGNet spatial patterns,\nIntegrated Gradients: P300 or motor? (F5)",
    ),
}


def anchor(name, side, shift=0.0):
    """Toạ độ điểm giữa một cạnh của ô (dịch theo chiều dọc cạnh một đoạn `shift`)."""
    x, y, w, h = BOXES[name][0]
    return {
        "left": (x, y + h / 2 + shift),
        "right": (x + w, y + h / 2 + shift),
        "top": (x + w / 2 + shift, y + h),
        "bottom": (x + w / 2 + shift, y),
    }[side]


def connectors():
    """Các đường nối: mỗi đường là danh sách điểm, mũi tên ở đoạn cuối."""
    top_bus, bottom_bus = 6.6, 2.55  # đường ngang chung giữa hàng 1-2 và hàng 2-3
    epochs_x = anchor("epochs", "bottom")[0]
    eval_x = anchor("eval", "bottom")[0]
    rq_x = [anchor("rq1", "top")[0], anchor("rq2", "top")[0], anchor("rq3", "top", -0.9)[0]]
    lines = [
        [anchor(a, "right"), anchor(b, "left")]
        for a, b in [("data", "behaviour"), ("behaviour", "preproc"), ("preproc", "epochs")]
    ]
    lines += [
        [anchor("epochs", "bottom"), anchor("erp", "top")],
        [(epochs_x, top_bus), (anchor("design", "top")[0], top_bus), anchor("design", "top")],
        [anchor("design", "right"), anchor("classical", "left")],
        [anchor("design", "right"), anchor("dl", "left")],
        [anchor("classical", "right"), anchor("eval", "left", 0.4)],
        [anchor("dl", "right"), anchor("eval", "left", -0.4)],
        [anchor("erp", "bottom", 0.9), anchor("rq3", "top", 0.6)],
    ]
    for x in rq_x:
        lines.append([anchor("eval", "bottom"), (eval_x, bottom_bus), (x, bottom_bus), (x, 1.9)])
    return lines


def plot_pipeline(font):
    fig, ax = plt.subplots(figsize=(16, 9.2))
    ax.set(xlim=(0, 16), ylim=(0, 9.2))
    ax.axis("off")
    for (x, y, w, h), role, title, body in BOXES.values():
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15", fc=COLORS[role], ec="#555555", lw=1
            )
        )
        ax.text(x + w / 2, y + h - 0.15, title, ha="center", va="top", fontsize=font + 1, weight="bold")
        # Nội dung căn giữa phần còn lại của ô (dưới tiêu đề)
        ax.text(x + w / 2, y + (h - 0.45) / 2, body, ha="center", va="center", fontsize=font, linespacing=1.3)
        if role != "all":
            ax.text(x + w - 0.08, y + 0.06, role, ha="right", va="bottom", fontsize=font - 1, color="#555555")
    style = {"color": "#444444", "lw": 1.2}
    for points in connectors():
        xs, ys = zip(*points)
        ax.plot(xs[:-1], ys[:-1], **style)
        ax.annotate(
            "", xy=points[-1], xytext=points[-2], arrowprops={"arrowstyle": "-|>", "shrinkA": 0, "shrinkB": 1, **style}
        )
    ax.set_title("F1: Analysis pipeline for single-trial target vs standard classification", fontsize=font + 3)
    ax.text(
        0.2,
        9.0,
        "Colour and label = team role (R1-R5); grey = shared protocol decisions",
        fontsize=font - 1,
        color="#555555",
        va="center",
    )
    return fig


def main():
    root = find_root(Path(__file__).resolve().parent)
    config = load_config(root)
    path = root / config["paths"]["figures"] / "F1_pipeline"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_pipeline(max(10, config["erp"]["font_size"] - 1))  # protocol mục 10: chữ >= 10 pt
    for ext in (".png", ".pdf"):
        fig.savefig(path.with_suffix(ext), dpi=config["erp"]["figure_dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"đã lưu {path.relative_to(root)}.png/.pdf")


if __name__ == "__main__":
    main()
