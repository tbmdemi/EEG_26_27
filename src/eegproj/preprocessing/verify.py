"""Post-run checks on the saved epoch files."""
import json
from pathlib import Path

import mne
import numpy as np

from .qc import epoch_qc_plot


def _epochs_path(root, config, subject):
    name = f'sub-{subject}_task-{config["dataset"]["task"]}'
    return name, Path(root) / config['paths']['derivatives'] / f'{name}_epo.fif'


def verify_epochs(root, config, subjects=None):
    """Re-read each FIF and assert sfreq, window, baseline, classes, PTP, ordering and blocks.

    Writes results/preproc_verification.json and returns the per-subject rows.
    """
    root = Path(root)
    pre = config['preprocessing']
    results = []
    for subject in subjects or config['dataset']['subjects']:
        _, path = _epochs_path(root, config, subject)
        epochs = mne.read_epochs(path, preload=True, verbose=False)
        data = epochs.get_data()
        assert epochs.info['sfreq'] == pre['resample_sfreq']
        assert np.allclose([epochs.tmin, epochs.tmax], [pre['epoch']['tmin'], pre['epoch']['tmax']])
        assert set(epochs.event_id) == {'standard', 'target', 'distractor'}
        assert np.isfinite(data).all()
        assert not epochs.info['bads']
        assert data.shape[-1] == round((epochs.tmax - epochs.tmin) * epochs.info['sfreq']) + 1
        assert np.max(np.ptp(data, axis=-1)) <= pre['epoch']['reject_ptp_uv'] * 1e-6 * 1.00001
        baseline = (epochs.times >= pre['epoch']['baseline'][0]) & (epochs.times <= pre['epoch']['baseline'][1])
        assert np.max(np.abs(data[:, :, baseline].mean(axis=-1))) < 1e-10
        assert list(epochs.metadata.columns) == ['subject', 'onset_s', 'block']
        assert epochs.metadata['onset_s'].is_monotonic_increasing
        assert set(epochs.metadata['block']) <= set(range(config['evaluation']['within']['n_splits']))
        results.append({'subject': subject, 'epochs': len(epochs), 'channels': len(epochs.ch_names),
                        'samples': len(epochs.times), 'blocks': sorted(epochs.metadata.block.unique().tolist()),
                        'verified': True})
        print(results[-1], flush=True)
    (root / 'results/preproc_verification.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    return results


def save_epoch_qc(root, config, subjects=None):
    """Save the whole-session epoch QC figure for each subject; return the figure paths."""
    root = Path(root)
    pre = config['preprocessing']
    qc_dir = root / 'figures/preprocessing'
    paths = []
    for subject in subjects or config['dataset']['subjects']:
        prefix, path = _epochs_path(root, config, subject)
        epochs = mne.read_epochs(path, preload=True, verbose=False)
        output = qc_dir / f'{prefix}_epoch_qc.png'
        epoch_qc_plot(epochs, prefix, output, pre['epoch']['reject_ptp_uv'], pre['qc']['dpi'])
        paths.append(output)
        del epochs
    return paths
