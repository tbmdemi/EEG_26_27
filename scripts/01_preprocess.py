"""Run the preprocessing pipeline end-to-end from the command line.

    python scripts/01_preprocess.py                    # all subjects in config, resume finished ones
    python scripts/01_preprocess.py -s 01 02 --overwrite
"""
import argparse
import os

for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '4')
os.environ.setdefault('MPLBACKEND', 'Agg')

from eegproj.config import find_root, load_config
from eegproj.preprocessing import prepare_raw, run, save_epoch_qc, verify_epochs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-s', '--subjects', nargs='+', help="subject IDs, e.g. 01 02 (default: config)")
    parser.add_argument('--overwrite', action='store_true', help='re-run subjects that already have a summary')
    parser.add_argument('--skip-verify', action='store_true', help='skip FIF checks and epoch QC figures')
    args = parser.parse_args()

    root = find_root(os.path.dirname(os.path.abspath(__file__)))
    config = load_config(root)
    subjects = args.subjects or config['dataset']['subjects']
    prepare_raw(root, config)
    run(root, subjects=subjects, overwrite=args.overwrite)
    if not args.skip_verify:
        verify_epochs(root, config, subjects)
        save_epoch_qc(root, config, subjects)


if __name__ == '__main__':
    main()
