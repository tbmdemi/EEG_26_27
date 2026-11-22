"""Time-window amplitude feature extraction for classical ERP decoding (protocol §6)."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


def compute_bin_masks(times: np.ndarray, bin_ms: float) -> tuple[np.ndarray, list[np.ndarray]]:
    """Compute time-bin edges (in s) and boolean sample masks for contiguous bins.

    Parameters
    ----------
    times : np.ndarray of shape (n_times,)
        Epoch timestamps in seconds (s), e.g., `[0.0, 0.8]` s at 250 Hz.
    bin_ms : float
        Target bin width in milliseconds (ms), e.g., `50.0` ms from `configs/config.yaml`.

    Returns
    -------
    edges : np.ndarray of shape (n_bins + 1,)
        Bin boundary timestamps in seconds (s).
    masks : list[np.ndarray]
        Boolean sample masks of shape `(n_times,)` for each bin; the final bin includes `tmax`.
    """
    times_arr = np.asarray(times, dtype=float)
    if times_arr.ndim != 1 or times_arr.size < 2:
        raise ValueError("times must be a 1D array with at least 2 time points.")
    if bin_ms <= 0:
        raise ValueError(f"bin_ms must be positive, got {bin_ms}.")

    tmin, tmax = float(times_arr[0]), float(times_arr[-1])
    duration_s = tmax - tmin
    if duration_s <= 0:
        raise ValueError(f"times must span a positive duration, got [{tmin}, {tmax}] s.")

    bin_s = float(bin_ms) / 1000.0
    n_bins = max(1, round(duration_s / bin_s))
    edges = np.linspace(tmin, tmax, n_bins + 1)

    masks = []
    for idx in range(n_bins):
        lo, hi = edges[idx], edges[idx + 1]
        if idx < n_bins - 1:
            mask = (times_arr >= lo - 1e-9) & (times_arr < hi - 1e-9)
        else:
            mask = (times_arr >= lo - 1e-9) & (times_arr <= hi + 1e-9)
        if not np.any(mask):
            raise ValueError(f"Empty time bin [{lo:.4f}, {hi:.4f}] s for given sampling rate.")
        masks.append(mask)
    return edges, masks


class TimeWindowAverager(BaseEstimator, TransformerMixin):
    """Average multi-channel EEG epochs within contiguous time bins (stateless transformer).

    Transforms an epoch tensor `X` of shape `(n_trials, n_channels, n_times)` in microvolts (µV)
    into a 2D feature matrix of shape `(n_trials, n_channels * n_bins)` in microvolts (µV),
    without hardcoding channel count or sampling rate (Hz).

    Parameters
    ----------
    times : np.ndarray of shape (n_times,)
        Epoch time axis in seconds (s).
    bin_ms : float, default=50.0
        Time-bin width in milliseconds (ms).
    """

    def __init__(self, times: np.ndarray, bin_ms: float = 50.0):
        self.times = times
        self.bin_ms = bin_ms

    def fit(self, X: np.ndarray, y=None):
        """Validate 3D epoch shape `(n_trials, n_channels, n_times)` (µV) and precompute bin masks.

        Learns no data statistics from `X` or `y`.
        """
        X_arr = np.asarray(X)
        if X_arr.ndim != 3:
            raise ValueError(f"Expected 3D input (n_trials, n_channels, n_times), got shape {X_arr.shape}.")
        times_arr = np.asarray(self.times, dtype=float)
        if X_arr.shape[-1] != times_arr.size:
            raise ValueError(f"Last axis of X ({X_arr.shape[-1]}) does not match len(times) ({times_arr.size}).")
        self.bin_edges_, self.bin_masks_ = compute_bin_masks(times_arr, self.bin_ms)
        self.n_channels_ = int(X_arr.shape[1])
        self.n_times_ = int(X_arr.shape[2])
        self.n_bins_ = len(self.bin_masks_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Compute per-channel mean amplitude (µV) in each time bin and flatten to 2D.

        Parameters
        ----------
        X : np.ndarray of shape (n_trials, n_channels, n_times)
            Epoch amplitudes in microvolts (µV).

        Returns
        -------
        np.ndarray of shape (n_trials, n_channels * n_bins)
            Mean bin amplitudes in microvolts (µV), ordered row-major as `(channel, bin)`.
        """
        check_is_fitted(self, ["bin_masks_", "n_channels_", "n_times_", "n_bins_"])
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 3 or X_arr.shape[1] != self.n_channels_ or X_arr.shape[2] != self.n_times_:
            raise ValueError(
                f"Input shape {X_arr.shape} is incompatible with fitted "
                f"(n_channels={self.n_channels_}, n_times={self.n_times_})."
            )
        binned = np.stack([X_arr[:, :, mask].mean(axis=-1) for mask in self.bin_masks_], axis=-1)
        return binned.reshape(X_arr.shape[0], self.n_channels_ * self.n_bins_)
