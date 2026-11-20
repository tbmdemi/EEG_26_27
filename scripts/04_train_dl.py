"""R5 — Deep learning (EEGNet, ShallowConvNet) on the shared CV splits; one protocol §7 CSV per combination.

    python scripts/04_train_dl.py                              # eegnet + shallow, within + cross, full + early, tgt_std
    python scripts/04_train_dl.py --models eegnet --settings cross --windows full
    python scripts/04_train_dl.py --tasks dis_std tgt_dis      # control tasks (only if time allows, protocol §6)
    python scripts/04_train_dl.py --subjects 01 03 04 --settings within   # quick check -> results/dev/
    python scripts/04_train_dl.py --summary                    # table of finished results (for the report)

Needs data v2 (python scripts/00_get_data.py). A GPU is used automatically when available (--device cuda/cpu).
Runs resume after an interruption: finished CSVs are skipped and unfinished ones continue fold by fold
(--overwrite recomputes). Outputs go to results/ (git-ignored): metrics/, predictions/, dl_runs/, checkpoints/.
"""

import argparse
import sys
import time
from pathlib import Path

import torch

from eegproj.config import find_root, load_config
from eegproj.models.dl import NETWORKS, pick_device
from eegproj.models.dl_runner import run_dl_grid, summarise_dl


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=list(NETWORKS), choices=list(NETWORKS))
    parser.add_argument("--settings", nargs="+", default=["within", "cross"], choices=["within", "cross"])
    parser.add_argument("--windows", nargs="+", default=None, help="default: all windows in config")
    parser.add_argument("--tasks", nargs="+", default=["tgt_std"], help="default: main task tgt_std")
    parser.add_argument("--subjects", nargs="+", default=None, help="subset for quick checks (e.g. 01 03)")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    parser.add_argument("--threads", type=int, default=None, help="CPU threads for torch")
    parser.add_argument("--overwrite", action="store_true", help="recompute existing results")
    parser.add_argument("--no-checkpoints", action="store_true", help="do not save fitted models")
    parser.add_argument("--summary", action="store_true", help="only print a table of finished DL results")
    parser.add_argument(
        "--allow-fewer-splits", action="store_true", help="cross-subject with < n_splits subjects (dev only)"
    )
    args = parser.parse_args()

    root = find_root(Path(__file__).resolve().parent)
    config = load_config(root)
    windows = args.windows or list(config["classification"]["windows"])
    unknown = set(args.tasks) - set(config["classification"]["tasks"]) or set(windows) - set(
        config["classification"]["windows"]
    )
    if unknown:
        sys.exit(f"Unknown task/window: {sorted(unknown)}")
    if args.summary:
        table = summarise_dl(root, config, models=args.models, tasks=args.tasks)
        print(table.round(3).to_string(index=False) if not table.empty else "no finished DL results yet")
        return
    if args.threads:
        torch.set_num_threads(args.threads)
    print(f"device: {pick_device(args.device)} | torch {torch.__version__} | threads {torch.get_num_threads()}")

    start = time.monotonic()
    for model in args.models:
        run_dl_grid(
            root,
            config,
            model,
            settings=args.settings,
            windows=windows,
            tasks=args.tasks,
            subjects=args.subjects,
            device=args.device,
            overwrite=args.overwrite,
            save_checkpoints=not args.no_checkpoints,
            allow_fewer_splits=args.allow_fewer_splits,
            log=lambda msg: print(msg, flush=True),
        )
    print(f"all done in {(time.monotonic() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
