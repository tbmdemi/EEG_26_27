"""Quality-control figures (saved to disk, never shown interactively)."""
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch


def qc_plot(before, raw, scores, output, cfg):
    """Save signal, spectrum, and channel amplitude checks for visual review."""
    sfreq, original = before
    cleaned = raw.get_data(start=0, stop=int(cfg['duration_s'] * raw.info['sfreq']))
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for data, rate, label in [(original, sfreq, 'Raw'), (cleaned, raw.info['sfreq'], 'Cleaned')]:
        f, power = welch(data, fs=rate, nperseg=min(int(rate * 2), data.shape[1]), axis=-1)
        axes[0, 0].plot(f, 10 * np.log10(np.maximum(np.median(power, axis=0) * 1e12, 1e-30)), label=label)
    axes[0, 0].set(xlim=(0, 100), xlabel='Frequency (Hz)', ylabel='Median PSD (dB uV²/Hz)')
    axes[0, 0].legend()
    axes[0, 1].bar(np.arange(len(scores)), scores['std_uv'], color=np.where(scores['auto_bad'], 'red', 'steelblue'))
    axes[0, 1].set(yscale='log', xlabel='Channel index', ylabel='Raw standard deviation (uV)')
    picks = [raw.ch_names.index(n) for n in ['Fp1', 'C3', 'Cz', 'Pz', 'O1']]
    for ax, data, rate, title in [(axes[1, 0], original, sfreq, 'Before'), (axes[1, 1], cleaned, raw.info['sfreq'], 'After')]:
        count = min(int(rate * cfg['trace_duration_s']), data.shape[1])
        for i in picks:
            ax.plot(np.arange(count) / rate, data[i, :count] * 1e6, label=raw.ch_names[i], linewidth=.6)
        ax.set(title=title, xlabel='Time (s)', ylabel='Amplitude (uV)')
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=cfg['dpi'])
    plt.close(fig)


def epoch_qc_plot(epochs, prefix, output, reject_ptp_uv, dpi):
    """Save per-trial peak-to-peak map over the whole session and the Pz ERP per condition."""
    amplitudes = np.ptp(epochs.get_data(), axis=-1) * 1e6
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    image = axes[0].imshow(amplitudes.T, origin='lower', aspect='auto', vmin=0, vmax=reject_ptp_uv)
    axes[0].set(xlabel='Retained trial index', ylabel='EEG channel index', title=f'{prefix}: peak-to-peak')
    fig.colorbar(image, ax=axes[0], label='Amplitude (uV)')
    for condition in ['standard', 'target', 'distractor']:
        if len(epochs[condition]):
            evoked = epochs[condition].average()
            axes[1].plot(evoked.times * 1000, evoked.get_data(picks=['Pz'])[0] * 1e6, label=condition)
    axes[1].axvline(0, color='gray', linewidth=.6)
    axes[1].set(xlabel='Time (ms)', ylabel='Amplitude at Pz (uV)', title='ERP quality check (exploratory)')
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
