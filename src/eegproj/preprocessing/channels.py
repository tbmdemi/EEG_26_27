"""Automatic bad-channel detection."""
import numpy as np
import pandas as pd


def bad_channel_scores(raw, cfg):
    """Flag flat channels or extreme log standard deviation (robust MAD z)."""
    data = raw.get_data()[:, ::cfg['sample_step']]
    std = np.std(data, axis=1)
    logs = np.log(np.maximum(std, np.finfo(float).tiny))
    mad = np.median(np.abs(logs - np.median(logs)))
    z = (logs - np.median(logs)) / max(1.4826 * mad, np.finfo(float).eps)
    frame = pd.DataFrame({'channel': raw.ch_names, 'std_uv': std * 1e6, 'robust_z': z})
    frame['auto_bad'] = (np.abs(z) > cfg['robust_z']) | (std * 1e6 < cfg['flat_std_uv'])
    return frame
