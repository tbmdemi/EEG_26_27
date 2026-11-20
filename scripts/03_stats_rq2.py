"""Run RQ2 Wilcoxon signed-rank + Holm statistical comparison and generate Figure F4 (protocol §8 & §10).

Usage:
    python scripts/03_stats_rq2.py
    python scripts/03_stats_rq2.py --metric auc --out-csv results/rq2_wilcoxon_holm_auc.csv
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

from eegproj.config import find_root, load_config
from eegproj.evaluation import (
    VALID_TASKS,
    aggregate_subject_scores,
    compute_rq2_wilcoxon_holm,
    load_all_metrics,
    plot_f4_paired,
)


def _resolve_safe_out_csv(root: Path, config: dict, out_csv_arg: str) -> Path:
    """Resolve `out_csv_arg` relative to `root` and ensure it is never inside `results/metrics/`."""
    out_csv = (root / out_csv_arg).resolve()
    metrics_dir = (root / config["paths"]["metrics"]).resolve()
    if out_csv == metrics_dir or metrics_dir in out_csv.parents:
        raise ValueError(f"RQ2 summary CSV must be saved outside {metrics_dir}, got {out_csv}.")
    return out_csv


def main() -> None:
    """CLI entry point for RQ2 Wilcoxon-Holm testing and Figure F4 generation."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="tgt_std", choices=[*sorted(VALID_TASKS), "all"])
    parser.add_argument("--metric", default="bal_acc", choices=["bal_acc", "auc"])
    parser.add_argument(
        "--out-csv",
        default="results/rq2_wilcoxon_holm.csv",
        help="Output summary CSV path (must remain outside results/metrics/).",
    )
    parser.add_argument(
        "--fig-stem",
        default=None,
        help="Output stem for Figure F4 (.png and .pdf); defaults to <paths.figures>/F4_paired_classical_vs_dl.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution in dots per inch (default: 300).")
    args = parser.parse_args()

    root = find_root(Path(__file__).resolve().parent)
    config = load_config(root)

    task_filter = None if args.task == "all" else args.task
    all_df = load_all_metrics(root, config, task_name=task_filter)
    sub_df = aggregate_subject_scores(all_df)
    stats_df = compute_rq2_wilcoxon_holm(sub_df, metric=args.metric)

    out_csv = _resolve_safe_out_csv(root, config, args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(out_csv, index=False)

    fig_stem = (
        root / args.fig_stem
        if args.fig_stem is not None
        else root / config["paths"]["figures"] / "F4_paired_classical_vs_dl"
    )
    png_path, pdf_path = plot_f4_paired(sub_df, stats_df, fig_stem, metric=args.metric, dpi=args.dpi)

    print(stats_df.to_string(index=False), flush=True)
    print(f"SAVED {out_csv}, {png_path}, {pdf_path}", flush=True)


if __name__ == "__main__":
    main()
