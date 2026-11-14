"""Single-subject preprocessing pipeline and batch runner (protocol section 1).

Reproducible oddball preprocessing, with original event timing and QC logs.
"""
import gc
import hashlib
import json
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import yaml
from mne_bids import BIDSPath, read_raw_bids
from mne_icalabel.iclabel import iclabel_label_components
from threadpoolctl import threadpool_limits

from .channels import bad_channel_scores
from .ica import ICAProgress
from .qc import qc_plot


def preprocess(root, subject, config):
    """Process one full recording and save all three stimulus classes in FIF."""
    start = time.monotonic()
    root = Path(root)
    pre = config['preprocessing']
    task = config['dataset']['task']
    rawroot = root / config['paths']['raw']
    out = root / config['paths']['derivatives']
    qc = root / 'figures' / 'preprocessing'
    out.mkdir(parents=True, exist_ok=True)
    qc.mkdir(parents=True, exist_ok=True)
    prefix = f'sub-{subject}_task-{task}'
    print(f'BEGIN {prefix}', flush=True)
    # BIDS trial_type only contains stimulus/response; recover event_type below.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Unable to map the following column')
        raw = read_raw_bids(BIDSPath(root=rawroot, subject=subject, task=task,
            datatype='eeg', suffix='eeg', extension='.vhdr'), verbose=False)
    raw.pick('eeg')
    raw.rename_channels({k: v for k, v in pre['channel_aliases'].items() if k in raw.ch_names})
    raw.set_montage(mne.channels.read_dig_captrak(rawroot / 'sourcedata' / f'sub-{subject}' / 'coords' / f'sub-{subject}_captrak.bvct'), on_missing='raise')
    table = pd.read_csv(rawroot / f'sub-{subject}' / 'eeg' / f'{prefix}_events.tsv', sep='\t')
    descriptions = ['Stimulus/' + v for v in table['event_type']]
    raw.set_annotations(mne.Annotations(table['onset'].to_numpy(), np.zeros(len(table)), descriptions))
    raw.load_data()
    if not np.isfinite(raw.get_data()).all():
        raise ValueError('Non-finite raw samples')
    before = (raw.info['sfreq'], raw.get_data(start=0, stop=int(pre['qc']['duration_s'] * raw.info['sfreq'])))
    scores = bad_channel_scores(raw, pre['bad_channels'])
    auto = scores.loc[scores.auto_bad, 'channel'].tolist()
    bads = sorted(set(raw.info['bads'] + auto + pre['bad_channels']['manual'].get(subject, [])))
    raw.info['bads'] = bads
    scores.to_csv(qc / f'{prefix}_channels.csv', index=False)
    raw.filter(pre['l_freq'], pre['h_freq'], method='fir', phase='zero', verbose=False)
    # 50 Hz is already outside the 40 Hz low-pass; no unnecessary notch.
    if bads:
        raw.interpolate_bads(reset_bads=True, mode='accurate', method={'eeg': 'spline'}, verbose=False)
    raw.set_eeg_reference(pre['reference'], projection=False, verbose=False)
    fit_raw = raw.copy().filter(pre['ica']['fit_l_freq'], None, verbose=False)
    ica = mne.preprocessing.ICA(n_components=pre['ica']['n_components'], method=pre['ica']['method'],
        fit_params={'extended': pre['ica']['extended'], 'w_change': pre['ica']['w_change'], 'verbose': True}, max_iter=pre['ica']['max_iter'], random_state=config['seed'])
    print(f'{prefix}: ICA fit, bad channels={bads}', flush=True)
    progress = ICAProgress()
    mne.utils.logger.addHandler(progress)
    try:
        ica.fit(fit_raw, decim=pre['ica']['decim'], verbose=True)
    finally:
        mne.utils.logger.removeHandler(progress)
    if not progress.step or (progress.step >= pre['ica']['max_iter'] and progress.change >= pre['ica']['w_change']):
        raise RuntimeError('ICA convergence not verified from iteration log')
    probabilities = iclabel_label_components(fit_raw, ica, backend='onnx')
    names = ['brain', 'muscle artifact', 'eye blink', 'heart beat', 'line noise', 'channel noise', 'other']
    labels = pd.DataFrame(probabilities, columns=names)
    labels.index.name = 'component'
    reject_cols = pre['ica']['iclabel_reject']
    labels['excluded'] = labels[reject_cols].max(axis=1) >= pre['ica']['iclabel_threshold']
    ica.exclude = labels.index[labels.excluded].tolist()
    labels.to_csv(qc / f'{prefix}_iclabel.csv')
    ica.save(out / f'{prefix}_ica.fif', overwrite=True, verbose=False)
    if ica.exclude:
        figs = ica.plot_components(picks=ica.exclude, show=False)
        if not isinstance(figs, list): figs = [figs]
        for i, fig in enumerate(figs):
            fig.savefig(qc / f'{prefix}_excluded_ics_{i}.png', dpi=pre['qc']['dpi'])
            plt.close(fig)
    del fit_raw
    gc.collect()
    ica.apply(raw, verbose=False)
    event_id = {name: i + 1 for i, name in enumerate(['standard', 'target', 'distractor'])}
    mapped = {config['dataset']['events'][k]: v for k, v in event_id.items()}
    events, _ = mne.events_from_annotations(raw, event_id=mapped, verbose=False)
    original_onsets = events[:, 0] / raw.info['sfreq']
    breaks = table.loc[table.event_type == 'S 10', 'onset'].to_numpy()
    blocks = np.searchsorted(breaks, original_onsets, side='right')
    raw, events = raw.resample(pre['resample_sfreq'], events=events, verbose=False)
    qc_plot(before, raw, scores, qc / f'{prefix}_qc.png', pre['qc'])
    ep = pre['epoch']
    epochs = mne.Epochs(raw, events, event_id, tmin=ep['tmin'], tmax=ep['tmax'], baseline=tuple(ep['baseline']),
        reject={'eeg': ep['reject_ptp_uv'] * 1e-6}, preload=True,
        metadata=pd.DataFrame({'subject': subject, 'onset_s': original_onsets, 'block': blocks}), verbose=False)
    epochs.drop_bad(verbose=False)
    counts_before = {k: int(np.sum(events[:, 2] == v)) for k, v in event_id.items()}
    counts_after = {k: len(epochs[k]) for k in event_id}
    excluded = counts_after['target'] / counts_before['target'] < pre['min_target_retained']
    epochs.save(out / f'{prefix}_epo.fif', overwrite=True, verbose=False)
    if not np.isfinite(epochs.get_data()).all(): raise ValueError('Non-finite epochs')
    row = {'subject': f'sub-{subject}', 'n_channels': len(raw.ch_names), 'sfreq': epochs.info['sfreq'],
        'bad_channels': ';'.join(bads), 'n_bad_channels': len(bads), 'n_ica': int(ica.n_components_),
        'ica_iterations': progress.step, 'ica_final_weight_change': progress.change,
        'n_ic_removed': len(ica.exclude), 'excluded_subject': excluded,
        'exclusion_reason': 'target_retention_below_threshold' if excluded else '',
        'montage': 'subject CapTrak', 'notch_applied': False, 'visual_review': 'pending',
        'seconds': round(time.monotonic() - start, 1)}
    for k in event_id:
        row.update({f'{k}_before': counts_before[k], f'{k}_after': counts_after[k], f'{k}_retained': counts_after[k] / counts_before[k]})
    (out / f'{prefix}_summary.json').write_text(json.dumps(row, indent=2), encoding='utf-8')
    print('DONE ' + json.dumps(row), flush=True)
    return row


def run(root, subjects=None, overwrite=False):
    """Run selected configured subjects; resume completed subjects by default."""
    root = Path(root)
    config_path = root / 'configs/config.yaml'
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    subjects = subjects or config['dataset']['subjects']
    out = root / config['paths']['derivatives']
    rows = []
    for subject in subjects:
        summary = out / f'sub-{subject}_task-{config["dataset"]["task"]}_summary.json'
        if summary.exists() and not overwrite:
            rows.append(json.loads(summary.read_text()))
        else:
            with threadpool_limits(limits=4):
                rows.append(preprocess(root, subject, config))
        pd.DataFrame(rows).to_csv(root / 'results/preproc_log.csv', index=False)
        gc.collect()
    provenance = {'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
        'config': config, 'mne_version': mne.__version__, 'subjects': subjects,
        'notes': ['ICA fitted on full unlabelled recording', 'ICA uses 1-40 Hz per protocol; ICLabel was trained on 1-100 Hz',
                  'ICA fitting decimates filtered data by 4; final recording resampled after ICA',
                  'Five-subject exploratory subset; visual QC required before final analysis']}
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
