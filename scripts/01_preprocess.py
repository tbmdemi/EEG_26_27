"""Run the preprocessing pipeline end-to-end from the command line.

    python scripts/01_preprocess.py                              # all subjects in config, resume finished ones
    python scripts/01_preprocess.py -s 01 03 --overwrite         # re-run some subjects
    python scripts/01_preprocess.py --download --delete-raw      # full dataset with little disk space:
                                                                 # download one subject (~0.7 GB), process it,
                                                                 # delete its raw recording, go to the next

Subjects listed under `dataset.exclude` in configs/config.yaml are always skipped.
Output: data/derivatives/preproc/*_epo.fif (+ ICA, summary), results/preproc_log.csv, figures/preprocessing/.
"""
import argparse
import os

for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '4')
os.environ.setdefault('MPLBACKEND', 'Agg')

from pathlib import Path  # noqa: E402

from eegproj.config import find_root, included_subjects, load_config  # noqa: E402
from eegproj.preprocessing import (  # noqa: E402
    download_subject,
    ensure_utf8_tsv,
    has_recording,
    prepare_raw,
    remove_raw_recording,
    run,
    save_epoch_qc,
    usable_subjects,
    verify_epochs,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-s', '--subjects', nargs='+', help="subject IDs, e.g. 01 03 (default: config)")
    parser.add_argument('--overwrite', action='store_true', help='re-run subjects that already have a current summary')
    parser.add_argument('--download', action='store_true', help='download missing oddball recordings from OpenNeuro')
    parser.add_argument('--delete-raw', action='store_true',
                        help='after processing a subject, delete its .eeg/.vhdr/.vmrk (sidecars are kept)')
    parser.add_argument('--skip-verify', action='store_true', help='skip FIF checks and epoch QC figures')
    args = parser.parse_args()

    root = find_root(os.path.dirname(os.path.abspath(__file__)))
    config = load_config(root)
    raw_root = Path(root) / config['paths']['raw']
    task = config['dataset']['task']
    requested = args.subjects or config['dataset']['subjects']
    subjects = included_subjects(config, requested)
    # A-priori exclusions: nothing to process; older derivatives are moved to <derivatives>/_excluded/.
    run(root, subjects=[s for s in requested if s not in subjects])
    prepare_raw(root, config, subjects)        # small sidecars + CapTrak coordinates
    for subject in subjects:
        if args.download:
            download_subject(raw_root, subject, task)
            ensure_utf8_tsv(raw_root / f'sub-{subject}')
        elif not has_recording(raw_root, subject, task):
            print(f'sub-{subject}: raw recording not found, skipped (use --download)', flush=True)
            continue
        run(root, subjects=[subject], overwrite=args.overwrite)
        if args.delete_raw:
            removed = remove_raw_recording(raw_root, subject, task)
            if removed:
                print(f'sub-{subject}: deleted raw recording {removed}', flush=True)
    if not args.skip_verify:
        done = [s for s in usable_subjects(root, config) if s in subjects]
        verify_epochs(root, config, done)
        save_epoch_qc(root, config, done)


if __name__ == '__main__':
    main()
