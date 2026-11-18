"""Single-subject preprocessing pipeline and batch runner (protocol section 1).

Reproducible oddball preprocessing, with original event timing and QC logs.
Only behaviourally correct trials are epoched (protocol §1, config `behaviour`).
"""
import gc
import hashlib
import json
import shutil
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

from ..config import excluded_subjects
from .behaviour import match_onsets, score_trials, summarise
from .channels import bad_channel_scores
from .ica import ICAProgress
from .qc import qc_plot

# Bump when the content of the saved epochs changes; older summaries are then re-processed.
# v1: all trials (2026-11-13). v2: correct trials only, rt in metadata, discontinuities marked BAD.
PIPELINE_VERSION = 2
EXPECTED_STIMULI = 660


def _prefix(subject, config):
    return f'sub-{subject}_task-{config["dataset"]["task"]}'


def _discontinuities(vmrk_path):
    """Onsets (s) of 'New Segment' markers after t=0, i.e. joins between concatenated files."""
    if not vmrk_path.exists():
        return np.array([])
    ann = mne.read_annotations(vmrk_path)
    return np.array([o for d, o in zip(ann.description, ann.onset) if d.startswith('New Segment') and o > 0])


def preprocess(root, subject, config):
    """Process one full recording and save the correct trials of all three stimulus classes in FIF."""
    start = time.monotonic()
    root = Path(root)
    pre = config['preprocessing']
    beh_cfg = config['behaviour']
    task = config['dataset']['task']
    rawroot = root / config['paths']['raw']
    out = root / config['paths']['derivatives']
    qc = root / 'figures' / 'preprocessing'
    out.mkdir(parents=True, exist_ok=True)
    qc.mkdir(parents=True, exist_ok=True)
    prefix = _prefix(subject, config)
    print(f'BEGIN {prefix}', flush=True)

    # Behaviour first: cheap, and decides whether the subject is usable at all.
    table = pd.read_csv(rawroot / f'sub-{subject}' / 'eeg' / f'{prefix}_events.tsv', sep='\t')
    trials = score_trials(table, beh_cfg['rt_min'], beh_cfg['rt_max'])
    behaviour = summarise(trials)
    flags = []
    if subject in (config['dataset'].get('flag') or {}):
        flags.append(config['dataset']['flag'][subject])
    if behaviour['n_stimuli'] != EXPECTED_STIMULI:
        flags.append(f"n_stimuli={behaviour['n_stimuli']}")
    if behaviour['hit_rate'] < beh_cfg['flag_hit_rate_below']:
        flags.append(f"hit_rate={behaviour['hit_rate']:.2f}")
    row = {'subject': f'sub-{subject}', 'pipeline_version': PIPELINE_VERSION, **behaviour}
    if behaviour['target_correct'] < beh_cfg['min_correct_targets']:
        _set_aside(out, prefix)   # never leave an older epoch file of an excluded subject in the shared folder
        row.update(excluded_subject=True, exclusion_reason=f"correct_targets={behaviour['target_correct']}",
                   flags='; '.join(flags), seconds=round(time.monotonic() - start, 1))
        (out / f'{prefix}_summary.json').write_text(json.dumps(row, indent=2), encoding='utf-8')
        print('EXCLUDED ' + json.dumps(row), flush=True)
        return row

    # BIDS trial_type only contains stimulus/response; recover event_type below.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Unable to map the following column')
        raw = read_raw_bids(BIDSPath(root=rawroot, subject=subject, task=task,
            datatype='eeg', suffix='eeg', extension='.vhdr'), verbose=False)
    raw.pick('eeg')
    raw.rename_channels({k: v for k, v in pre['channel_aliases'].items() if k in raw.ch_names})
    raw.set_montage(mne.channels.read_dig_captrak(rawroot / 'sourcedata' / f'sub-{subject}' / 'coords' / f'sub-{subject}_captrak.bvct'), on_missing='raise')
    descriptions = ['Stimulus/' + v for v in table['event_type']]
    annotations = mne.Annotations(table['onset'].to_numpy(), np.zeros(len(table)), descriptions)
    # Joins between concatenated files (sub-37, sub-38): skip in filtering, ICA fit and epoching.
    joins = _discontinuities(rawroot / f'sub-{subject}' / 'eeg' / f'{prefix}_eeg.vmrk')
    pad = pre['discontinuity_pad_s']
    if joins.size:
        onsets = np.clip(joins - pad, 0, None)
        annotations += mne.Annotations(onsets, joins + pad - onsets, ['BAD_ACQ_SKIP'] * joins.size)
        flags.append(f'discontinuities={joins.size}')
    raw.set_annotations(annotations)
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
    # Attach behaviour to every stimulus event and keep only correct trials.
    index = match_onsets(original_onsets, trials['onset_s'].to_numpy())
    names_by_code = {v: k for k, v in event_id.items()}
    if not (trials['condition'].to_numpy()[index] == [names_by_code[c] for c in events[:, 2]]).all():
        raise ValueError('Stimulus classes in annotations and events.tsv disagree')
    correct = trials['correct'].to_numpy(bool)[index]
    rts = trials['rt'].to_numpy(float)[index]
    breaks = table.loc[table.event_type == 'S 10', 'onset'].to_numpy()
    blocks = np.searchsorted(breaks, original_onsets, side='right')
    counts_before = {k: int(np.sum(events[:, 2] == v)) for k, v in event_id.items()}
    events, original_onsets, blocks, rts = events[correct], original_onsets[correct], blocks[correct], rts[correct]
    counts_correct = {k: int(np.sum(events[:, 2] == v)) for k, v in event_id.items()}
    raw, events = raw.resample(pre['resample_sfreq'], events=events, verbose=False)
    qc_plot(before, raw, scores, qc / f'{prefix}_qc.png', pre['qc'])
    ep = pre['epoch']
    epochs = mne.Epochs(raw, events, event_id, tmin=ep['tmin'], tmax=ep['tmax'], baseline=tuple(ep['baseline']),
        reject={'eeg': ep['reject_ptp_uv'] * 1e-6}, reject_by_annotation=True, preload=True,
        metadata=pd.DataFrame({'subject': subject, 'onset_s': original_onsets, 'block': blocks, 'rt': rts}),
        verbose=False)
    epochs.drop_bad(verbose=False)
    counts_after = {k: len(epochs[k]) for k in event_id}
    excluded = counts_after['target'] / counts_correct['target'] < pre['min_target_retained']
    if excluded:   # keep the file for inspection, but outside the folder the team loads from
        _set_aside(out, prefix)
        (out / '_excluded').mkdir(exist_ok=True)
    epochs.save((out / '_excluded' if excluded else out) / f'{prefix}_epo.fif', overwrite=True, verbose=False)
    if not np.isfinite(epochs.get_data()).all(): raise ValueError('Non-finite epochs')
    row.update({'n_channels': len(raw.ch_names), 'sfreq': epochs.info['sfreq'],
        'bad_channels': ';'.join(bads), 'n_bad_channels': len(bads), 'n_ica': int(ica.n_components_),
        'ica_iterations': progress.step, 'ica_final_weight_change': progress.change,
        'n_ic_removed': len(ica.exclude), 'excluded_subject': excluded,
        'exclusion_reason': 'target_retention_below_threshold' if excluded else '',
        'flags': '; '.join(flags),
        'montage': 'subject CapTrak', 'notch_applied': False, 'visual_review': 'pending',
        'seconds': round(time.monotonic() - start, 1)})
    for k in event_id:
        row.update({f'{k}_before': counts_before[k], f'{k}_correct_trials': counts_correct[k],
                    f'{k}_after': counts_after[k],
                    f'{k}_retained': counts_after[k] / counts_correct[k] if counts_correct[k] else np.nan})
    (out / f'{prefix}_summary.json').write_text(json.dumps(row, indent=2), encoding='utf-8')
    print('DONE ' + json.dumps(row), flush=True)
    return row


def _set_aside(out, prefix):
    """Move derivatives of a now-excluded subject to <derivatives>/_excluded/ (never deleted)."""
    moved = []
    for path in out.glob(f'{prefix}_*'):
        target = out / '_excluded' / path.name
        target.parent.mkdir(exist_ok=True)
        shutil.move(str(path), str(target))
        moved.append(path.name)
    return moved


def write_log(root, config, save=True):
    """Rebuild results/preproc_log.csv from every summary JSON and the a-priori exclusions.

    With save=False the table is only returned (used by readers, so they never overwrite the shared log).
    """
    root = Path(root)
    out = root / config['paths']['derivatives']
    excluded = excluded_subjects(config)
    rows = []
    for subject in config['dataset']['subjects']:
        if subject in excluded:
            rows.append({'subject': f'sub-{subject}', 'pipeline_version': PIPELINE_VERSION,
                         'excluded_subject': True, 'exclusion_reason': excluded[subject]})
            continue
        summary = out / f'{_prefix(subject, config)}_summary.json'
        if summary.exists():
            rows.append(json.loads(summary.read_text()))
    log = pd.DataFrame(rows)
    if save:
        log.to_csv(root / 'results/preproc_log.csv', index=False)
    return log


def usable_subjects(root, config):
    """Subjects with a current (PIPELINE_VERSION) epoch file that are not excluded."""
    log = write_log(root, config, save=False)
    if log.empty:
        return []
    ok = log[(log.pipeline_version == PIPELINE_VERSION) & (~log.excluded_subject.fillna(True).astype(bool))]
    return [s.replace('sub-', '') for s in ok.subject]


def run(root, subjects=None, overwrite=False):
    """Run selected configured subjects; resume subjects already done with the current pipeline version."""
    root = Path(root)
    config_path = root / 'configs/config.yaml'
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    subjects = config['dataset']['subjects'] if subjects is None else subjects
    excluded = excluded_subjects(config)
    out = root / config['paths']['derivatives']
    for subject in subjects:
        prefix = _prefix(subject, config)
        if subject in excluded:
            moved = _set_aside(out, prefix)
            print(f'SKIP {prefix}: excluded ({excluded[subject]})' + (f'; moved old files {moved}' if moved else ''))
            continue
        summary = out / f'{prefix}_summary.json'
        current = summary.exists() and json.loads(summary.read_text()).get('pipeline_version') == PIPELINE_VERSION
        if current and not overwrite:
            print(f'RESUME {prefix}: already processed with pipeline v{PIPELINE_VERSION}')
            continue
        with threadpool_limits(limits=4):
            preprocess(root, subject, config)
        gc.collect()
    log = write_log(root, config)
    is_excluded = log.excluded_subject.fillna(True).astype(bool) if not log.empty else pd.Series(dtype=bool)
    provenance = {'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
        'config': config, 'mne_version': mne.__version__, 'pipeline_version': PIPELINE_VERSION,
        'subjects_usable': log.loc[~is_excluded, 'subject'].tolist() if not log.empty else [],
        'subjects_excluded': log.loc[is_excluded, ['subject', 'exclusion_reason']].to_dict('records') if not log.empty else [],
        'notes': ['Only behaviourally correct trials are epoched (config `behaviour`)',
                  'ICA fitted on full unlabelled recording', 'ICA uses 1-40 Hz per protocol; ICLabel was trained on 1-100 Hz',
                  'ICA fitting decimates filtered data by 4; final recording resampled after ICA',
                  'Visual QC required before final analysis (visual_review=pending)']}
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2, default=str), encoding='utf-8')
    return log


__all__ = ['PIPELINE_VERSION', 'preprocess', 'run', 'usable_subjects', 'write_log']
