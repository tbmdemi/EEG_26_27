"""Run classical ML baselines end-to-end across settings, windows, and tasks (protocol §4–§7).

Examples
--------
    python scripts/02_run_classical_ml.py
    python scripts/02_run_classical_ml.py -m lda_tw xdawn_rg --settings within --subjects 01 03
    python scripts/02_run_classical_ml.py --allow-dev-splits
"""

import argparse
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "4")
os.environ.setdefault("MPLBACKEND", "Agg")

from eegproj.config import find_root, load_config
from eegproj.models import CLASSICAL_MODELS, run_classical_grid


def main() -> None:
    """Parse CLI options and run the configured classical ML benchmark grid."""
    root = find_root(os.path.dirname(os.path.abspath(__file__)))
    config = load_config(root)
    valid_windows = list(config["classification"]["windows"])
    valid_tasks = list(config["classification"]["tasks"])

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-m", "--models", nargs="+", default=list(CLASSICAL_MODELS), choices=list(CLASSICAL_MODELS))
    parser.add_argument("--settings", nargs="+", default=["within", "cross"], choices=["within", "cross"])
    parser.add_argument(
        "-w",
        "--windows",
        nargs="+",
        default=valid_windows,
        choices=valid_windows,
        help=f"time windows in s (default: {valid_windows})",
    )
    parser.add_argument(
        "-t",
        "--tasks",
        nargs="+",
        default=valid_tasks,
        choices=valid_tasks,
        help=f"classification tasks (default: {valid_tasks})",
    )
    parser.add_argument("-s", "--subjects", nargs="+", help="subject IDs, e.g. 01 03 (default: usable v2 subjects)")
    parser.add_argument(
        "--allow-dev-splits",
        action="store_true",
        help="allow fewer than config n_splits (5) subjects in cross-subject CV for exploratory subset",
    )
    parser.add_argument("--no-overwrite", action="store_true", help="skip combinations whose CSV already exists")
    args = parser.parse_args()

    run_classical_grid(
        root=root,
        config=config,
        models=args.models,
        settings=args.settings,
        windows=args.windows,
        tasks=args.tasks,
        subjects=args.subjects,
        allow_fewer_splits=args.allow_dev_splits,
        overwrite=not args.no_overwrite,
    )


if __name__ == "__main__":
    main()
