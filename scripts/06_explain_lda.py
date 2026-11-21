"""R1: Activation pattern của LDA (`lda_tw` của R4) theo Haufe et al. 2014 (protocol mục 9.4, RQ3) -> hình F5c.

Fit lại `lda_tw` trên đúng 5 fold cross-subject của R4/R5 rồi đổi trọng số thành pattern (uV), để so với spatial
pattern và Integrated Gradients của EEGNet (scripts/05_explain_dl.py).

Cách chạy:
    python scripts/06_explain_lda.py            # cross, full + early, task tgt_std (khoảng 5 phút, chỉ CPU)

Cần: dữ liệu v2 (scripts/00_get_data.py) và results/cv_cross_subject_folds.csv.

Đầu ra (gitignored):
    results/explain/lda_cross_<window>_<task>_patterns.npz   pattern từng fold (fold, kênh, bin), mốc bin, tên kênh
    results/explain/lda_cross_<task>_metrics.csv             điểm từng subject test (để so với CSV của R4)
    results/explain/lda_pattern_summary_<task>.csv           theo bin 50 ms: tương quan với bản đồ P300, pattern ở 2 vùng kênh
    figures/F5c_lda_patterns.{png,pdf}   (task đối chứng: F5c_lda_patterns_<task>)
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # chỉ lưu hình ra file, không mở cửa sổ
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eegproj.config import find_root, load_config
from eegproj.evaluation.folds import cross_subject_folds
from eegproj.evaluation.splits import load_valid_subjects
from eegproj.explain.dl import group_info
from eegproj.explain.lda import lda_cross, load_features, p300_reference_map, pattern_table
from eegproj.explain.plots import plot_lda_patterns


def save_figure(fig, path, dpi):
    """Lưu hình thành .png và .pdf (quy ước hình của protocol mục 10)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        fig.savefig(path.with_suffix(ext), dpi=dpi)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--windows", nargs="+", default=None, help="mặc định: mọi cửa sổ trong config")
    parser.add_argument("--task", default="tgt_std")
    args = parser.parse_args()

    root = find_root(Path(__file__).resolve().parent)
    config = load_config(root)
    windows = args.windows or list(config["classification"]["windows"])
    out_dir = root / "results" / "explain"
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = load_valid_subjects(root, config)
    folds = cross_subject_folds(root, config, subjects)  # đúng file fold dùng chung với R4/R5
    print(f"{len(subjects)} subject, {len(folds)} fold cross-subject, task {args.task}", flush=True)

    results, metrics, tables, ref_map = {}, [], [], None
    for window in windows:
        features = load_features(root, config, subjects, window, args.task)
        if ref_map is None:
            ch_names = features["ch_names"]
            # Bản đồ P300 cần đủ 250-600 ms nên lấy từ cửa sổ dài nhất (full); `early` chỉ tới 450 ms.
            ref_features = features if window == "full" else load_features(root, config, subjects, "full", args.task)
            ref_map = p300_reference_map(ref_features, config)
        elif features["ch_names"] != ch_names:
            raise ValueError("Thứ tự kênh khác giữa các cửa sổ.")

        patterns, scores = lda_cross(features, folds, config)
        scores.insert(0, "window", window)
        metrics.append(scores)
        tables.append(pattern_table(patterns, features, ref_map, window))
        results[window] = {"patterns": patterns, "bin_edges": features["bin_edges"]}
        np.savez_compressed(
            out_dir / f"lda_cross_{window}_{args.task}_patterns.npz",
            patterns=patterns,
            bin_edges=features["bin_edges"],
            ch_names=np.array(ch_names),
            ref_map=ref_map,
        )
        print(
            f"[lda_tw cross {window}] bal_acc trung bình {scores['bal_acc'].mean():.4f} | "
            f"AUC trung bình {scores['auc'].mean():.4f} (so với CSV của R4)",
            flush=True,
        )

    metrics = pd.concat(metrics, ignore_index=True)
    metrics.to_csv(out_dir / f"lda_cross_{args.task}_metrics.csv", index=False)
    summary = pd.concat(tables, ignore_index=True)
    summary.to_csv(out_dir / f"lda_pattern_summary_{args.task}.csv", index=False)
    print(summary.round(3).to_string(index=False))

    info = group_info(root, config, subjects, ch_names, 250.0)
    save_figure(
        plot_lda_patterns(results, info, config, args.task),
        root / config["paths"]["figures"] / ("F5c_lda_patterns" + ("" if args.task == "tgt_std" else f"_{args.task}")),
        config["erp"]["figure_dpi"],
    )


if __name__ == "__main__":
    main()
