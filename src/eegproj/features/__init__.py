"""Feature extraction modules for classical ML models."""

from .time_windows import TimeWindowAverager, compute_bin_masks

__all__ = ["TimeWindowAverager", "compute_bin_masks"]
