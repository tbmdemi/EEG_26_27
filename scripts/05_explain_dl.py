"""R1: Giải thích EEGNet (protocol mục 9.2-9.3, RQ3): spatial pattern + Integrated Gradients -> hình F5.

Cách chạy:
    python scripts/05_explain_dl.py                    # cross + within, full + early, task tgt_std
    python scripts/05_explain_dl.py --settings cross   # chỉ mô hình cross-subject
    python scripts/05_explain_dl.py --task dis_std     # task đối chứng (cần checkpoint của task đó)

Cần: dữ liệu v2 (scripts/00_get_data.py) và checkpoint/predictions của R5 trong results/ (scripts/04_get_dl_results.py hoặc scripts/04_train_dl.py).

Đầu ra (gitignored):
    results/explain/eegnet_<setting>_<window>_<task>_saliency.npz   IG trung bình theo subject (có dấu và |IG|)
    results/explain/eegnet_cross_<window>_<task>_patterns.csv       spatial pattern và mức giảm AUC của từng filter, từng fold
    results/explain/saliency_roi_<task>.csv, saliency_summary_<task>.csv   |IG| ở vùng P300 / vùng vận động trái
    results/explain/manifest.json                                    kết quả các bước kiểm tra
    figures/F5a_eegnet_spatial_patterns.{png,pdf}, figures/F5b_eegnet_saliency_<setting>.{png,pdf}
Với task đối chứng, manifest và tên hình có thêm hậu tố _<task>, ví dụ F5b_eegnet_saliency_cross_dis_std.
"""

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # chỉ lưu hình ra file, không mở cửa sổ
import matplotlib.pyplot as plt
import pandas as pd

from eegproj.config import find_root, load_config
from eegproj.evaluation.splits import load_valid_subjects
from eegproj.explain.dl import (
    check_decisions,
    checkpoint_paths,
    explain_run,
    group_info,
    load_explain_data,
    median_rt,
    p300_reference_map,
    roi_table,
    save_run,
    summarise_roi,
    write_manifest,
)
from eegproj.explain.plots import plot_saliency, plot_spatial_patterns, task_classes
from eegproj.models.dl import load_checkpoint, pick_device


def save_figure(fig, path, dpi):
    """Lưu hình thành .png và .pdf (quy ước hình của protocol mục 10)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        fig.savefig(path.with_suffix(ext), dpi=dpi, bbox_inches="tight")  # không cắt nhãn colorbar ở mép hình
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--settings", nargs="+", default=["cross", "within"], choices=["within", "cross"])
    parser.add_argument("--windows", nargs="+", default=None, help="mặc định: mọi cửa sổ trong config")
    parser.add_argument("--task", default="tgt_std")
    parser.add_argument("--steps", type=int, default=64, help="số bước của Integrated Gradients")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    args = parser.parse_args()

    root = find_root(Path(__file__).resolve().parent)
    config = load_config(root)
    windows = args.windows or list(config["classification"]["windows"])
    out_dir = root / "results" / "explain"
    fig_dir = root / config["paths"]["figures"]
    dpi = config["erp"]["figure_dpi"]
    suffix = "" if args.task == "tgt_std" else f"_{args.task}"  # task chính giữ tên file cũ
    device = str(pick_device(args.device))
    start = time.monotonic()

    # 1. Đọc dữ liệu mọi subject, sắp kênh theo đúng thứ tự R5 đã dùng khi train (lấy từ một checkpoint).
    paths = checkpoint_paths(root, config, args.settings[0], windows[0], args.task)
    if not paths:
        raise SystemExit("Chưa có checkpoint EEGNet trong results/checkpoints/; chạy scripts/04_train_dl.py trước.")
    ch_names = load_checkpoint(paths[0])[1]["extra"]["ch_names"]
    subjects = load_valid_subjects(root, config)
    print(f"device {device} | đọc {len(subjects)} subject ({args.task}) ...", flush=True)
    data = load_explain_data(root, config, subjects, ch_names, task=args.task)

    # 2. Thông tin dùng chung: vị trí điện cực, bản đồ P300 tham chiếu, RT trung vị.
    info = group_info(root, config, subjects, ch_names, config["models"]["dl"]["sfreq"])
    ref_map = p300_reference_map(data, config)
    # Chỉ trial target có bấm nút, nên chỉ có RT khi lớp dương là target
    positive = task_classes(config, args.task)[0]
    rt_median = median_rt(data) if positive == "target" else None

    # 3. Giải thích từng tổ hợp setting và window, kiểm tra, lưu kết quả và vẽ hình.
    checks, roi_tables = [], []
    for setting in args.settings:
        results = {}
        for window in windows:
            t0 = time.monotonic()
            res = explain_run(root, config, data, setting, window, args.task, args.steps, device, ref_map)
            diff = check_decisions(root, res)  # dừng nếu lệch so với predictions của R5
            name = save_run(res, out_dir)
            roi_tables.append(roi_table(res))
            checks.append(
                {
                    "run": name,
                    "n_subjects": len(res["subjects"]),
                    "n_correct_targets": int(res["n_correct"].sum()),
                    "n_targets": int(res["n_targets"].sum()),
                    "max_decision_diff_vs_predictions": diff,
                    "max_ig_completeness_rel_error": res["ig_rel_error"],
                }
            )
            print(
                f"[{name}] {res['n_correct'].sum()}/{res['n_targets'].sum()} trial {positive} đoán đúng | "
                f"lệch quyết định {diff:.1e} | sai số IG {res['ig_rel_error']:.1e} | {time.monotonic() - t0:.0f}s",
                flush=True,
            )
            results[window] = res

        save_figure(
            plot_saliency(results, info, config, rt_median), fig_dir / f"F5b_eegnet_saliency_{setting}{suffix}", dpi
        )
        if setting == "cross":
            fig = plot_spatial_patterns(results, info, ref_map, config)
            save_figure(fig, fig_dir / f"F5a_eegnet_spatial_patterns{suffix}", dpi)

    # 4. Bảng |IG| theo vùng kênh và khoảng thời gian, cùng file manifest.
    table = pd.concat(roi_tables, ignore_index=True)
    table.to_csv(out_dir / f"saliency_roi_{args.task}.csv", index=False)
    summary = summarise_roi(table)
    summary.to_csv(out_dir / f"saliency_summary_{args.task}.csv", index=False)
    write_manifest(root, out_dir, checks, args.steps, f"manifest{suffix}.json")

    columns = ["setting", "window", "time_bin", "bin_share_median", "p300_enrichment_median"]
    columns += ["motor_enrichment_median", "p_p300_vs_motor"]
    print(summary[columns].round(3).to_string(index=False))
    if rt_median is not None:
        print(f"RT trung vị của target đúng: {rt_median * 1000:.0f} ms")
    print(f"xong sau {(time.monotonic() - start) / 60:.1f} phút")


if __name__ == "__main__":
    main()
