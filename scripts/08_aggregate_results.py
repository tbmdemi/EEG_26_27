"""Validate R4/R5 CSVs and export F3; never fit models or regenerate folds.

Usage: python scripts/08_aggregate_results.py
Requires 16 tgt_std CSVs and the v2 handoff log. Verifies all supplied metric
folds against current GroupKFold; checks the shared fold CSV when present.
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

from eegproj.config import find_root, load_config
from eegproj.evaluation.aggregate import generate_f3


def main():
    """Export validated subject-level F3 artifacts, or fail without outputs."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    root = find_root(Path(__file__).resolve().parent)
    try:
        outputs = generate_f3(root, load_config(root))
    except (FileNotFoundError, ValueError, KeyError) as error:
        parser.exit(1, f"F3 blocked: {error}\n")
    for path in outputs:
        print(f"SAVED {path}", flush=True)


if __name__ == "__main__":
    main()
