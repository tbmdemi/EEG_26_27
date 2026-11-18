"""Behavioural scoring of the oddball task from the BIDS *_events.tsv (protocol §1).

A response (event_type 'S  1') counts for a stimulus when it falls in [rt_min, rt_max] seconds
after the stimulus onset (Dzianok et al. 2022: 100 ms to 200 ms + 1200 ms shortest ISI).
    target      correct = at least one response in the window; RT = first response
    standard    correct = no response in the window
    distractor  correct = no response in the window
"""
import numpy as np
import pandas as pd

STIMULUS_CODES = {'S  5': 'standard', 'S  6': 'target', 'S  7': 'distractor'}
RESPONSE_CODE = 'S  1'


def score_trials(table, rt_min, rt_max):
    """Return one row per stimulus: onset (s), condition, correct (bool), rt (s, NaN if none).

    `table` is the events.tsv DataFrame with columns `onset` and `event_type`.
    """
    codes = table['event_type'].astype(str)
    responses = np.sort(table.loc[codes == RESPONSE_CODE, 'onset'].to_numpy(float))
    stim = table.loc[codes.isin(STIMULUS_CODES)].sort_values('onset')
    rows = []
    for onset, code in zip(stim['onset'].to_numpy(float), codes[stim.index]):
        latency = responses - onset
        hits = latency[(latency >= rt_min) & (latency <= rt_max)]
        condition = STIMULUS_CODES[code]
        responded = hits.size > 0
        rows.append({'onset_s': onset, 'condition': condition,
                     'correct': responded if condition == 'target' else not responded,
                     'rt': float(hits.min()) if responded else np.nan})
    return pd.DataFrame(rows)


def summarise(trials):
    """Behavioural summary for the log: counts, hit rate, false-alarm rates, RT of correct targets."""
    out = {'n_stimuli': len(trials)}
    for condition in STIMULUS_CODES.values():
        sub = trials[trials.condition == condition]
        out[f'{condition}_n'] = len(sub)
        out[f'{condition}_correct'] = int(sub.correct.sum())
    target = trials[trials.condition == 'target']
    out['hit_rate'] = float(target.correct.mean()) if len(target) else np.nan
    for condition in ('standard', 'distractor'):
        sub = trials[trials.condition == condition]
        out[f'false_alarm_{condition}'] = float(1 - sub.correct.mean()) if len(sub) else np.nan
    rts = target.loc[target.correct, 'rt']
    out['rt_mean_ms'] = float(rts.mean() * 1000) if len(rts) else np.nan
    out['rt_sd_ms'] = float(rts.std() * 1000) if len(rts) > 1 else np.nan
    return out


def match_onsets(event_onsets, trial_onsets, tol=0.002):
    """Index into `trial_onsets` for each event onset (nearest, must be within `tol` seconds)."""
    trial_onsets = np.asarray(trial_onsets, float)
    order = np.argsort(trial_onsets)
    sorted_onsets = trial_onsets[order]
    pos = np.clip(np.searchsorted(sorted_onsets, event_onsets), 1, len(sorted_onsets) - 1)
    left, right = sorted_onsets[pos - 1], sorted_onsets[pos]
    nearest = np.where(np.abs(event_onsets - left) <= np.abs(event_onsets - right), pos - 1, pos)
    if np.max(np.abs(sorted_onsets[nearest] - event_onsets)) > tol:
        raise ValueError('Epoch events do not match events.tsv onsets')
    return order[nearest]
