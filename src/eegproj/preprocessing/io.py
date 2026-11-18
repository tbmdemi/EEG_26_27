"""Raw-data preparation: download oddball recordings, fetch small sidecars, fix TSV encodings."""
import urllib.request
from pathlib import Path

import defusedxml.ElementTree  # noqa: F401  required by MNE CapTrak reader; fail early if missing

from ..config import included_subjects

OPENNEURO_S3 = 'https://s3.amazonaws.com/openneuro.org/ds004621/'


def fetch_metadata(raw_root, subjects, task='oddball'):
    """Download the task EEG sidecar and per-subject CapTrak coordinates if missing.

    Only small files are fetched; no EEG recordings or extra subjects.
    """
    raw_root = Path(raw_root)
    relative_files = [f'task-{task}_eeg.json'] + [
        f'sourcedata/sub-{s}/coords/sub-{s}_captrak.bvct' for s in subjects
    ]
    fetched = []
    for relative in relative_files:
        destination = raw_root / relative
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(urllib.request.urlopen(OPENNEURO_S3 + relative, timeout=60).read())
            fetched.append(relative)
    return fetched


def ensure_utf8_tsv(root):
    """Re-encode any non-UTF-8 BIDS .tsv sidecar (Latin-1 in ds004621) to UTF-8, in place.

    Some ds004621 files (e.g. *_channels.tsv) store 'µV' as Latin-1 byte 0xB5, which makes
    mne_bids raise UnicodeDecodeError. Idempotent: files already in UTF-8 are skipped.
    """
    root = Path(root)
    converted = []
    for path in sorted(root.rglob('*.tsv')):
        data = path.read_bytes()
        try:
            data.decode('utf-8')
        except UnicodeDecodeError:
            path.write_bytes(data.decode('latin-1').encode('utf-8'))
            converted.append(path.relative_to(root))
    return converted


RECORDING_SUFFIXES = ('_eeg.eeg', '_eeg.vhdr', '_eeg.vmrk')
TOP_LEVEL_FILES = ['dataset_description.json', 'participants.tsv', 'participants.json', 'README', 'CHANGES',
                   'task-{task}_events.json']


def has_recording(raw_root, subject, task='oddball'):
    """True if the BrainVision triplet and events.tsv of this subject/task are on disk."""
    eeg = Path(raw_root) / f'sub-{subject}' / 'eeg'
    names = [f'sub-{subject}_task-{task}{s}' for s in RECORDING_SUFFIXES + ('_events.tsv',)]
    return all((eeg / n).exists() for n in names)


def download_subject(raw_root, subject, task='oddball', dataset='ds004621'):
    """Download only this subject's `task` files (+ small top-level metadata) from OpenNeuro.

    About 0.7 GB per subject for the oddball task. Files already on disk are skipped.
    """
    import openneuro  # openneuro-py; imported here so the rest of the package works without it

    raw_root = Path(raw_root)
    if has_recording(raw_root, subject, task):
        return False
    include = [f'sub-{subject}/eeg/*task-{task}*'] + [f.format(task=task) for f in TOP_LEVEL_FILES]
    openneuro.download(dataset=dataset, target_dir=raw_root, include=include)
    if not has_recording(raw_root, subject, task):
        raise FileNotFoundError(f'sub-{subject}: download finished but recording files are missing')
    return True


def remove_raw_recording(raw_root, subject, task='oddball'):
    """Delete the large BrainVision files of one subject; keep the small .tsv/.json sidecars."""
    eeg = Path(raw_root) / f'sub-{subject}' / 'eeg'
    removed = []
    for suffix in RECORDING_SUFFIXES:
        path = eeg / f'sub-{subject}_task-{task}{suffix}'
        if path.exists():
            path.unlink()
            removed.append(path.name)
    return removed


def prepare_raw(root, config, subjects=None, verbose=True):
    """Make the raw BIDS folder readable: fetch missing metadata, then fix TSV encodings."""
    raw_root = Path(root) / config['paths']['raw']
    fetched = fetch_metadata(raw_root, included_subjects(config, subjects), config['dataset']['task'])
    converted = ensure_utf8_tsv(raw_root)
    if verbose:
        print(f'Fetched {len(fetched)} metadata file(s); converted {len(converted)} TSV file(s) to UTF-8.')
        for path in converted:
            print('  ', path)
    return fetched, converted
