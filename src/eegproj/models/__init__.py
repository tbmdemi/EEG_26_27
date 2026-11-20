"""Model definitions and runners for classical ML (and deep learning modules in dl/dl_runner)."""

from .classical import (
    CLASSICAL_MODELS,
    build_classical_model,
    build_lda_tw,
    build_xdawn_rg,
    extract_lda_haufe_patterns,
)
from .runner import run_classical_experiment, run_classical_grid

__all__ = [
    "CLASSICAL_MODELS",
    "build_classical_model",
    "build_lda_tw",
    "build_xdawn_rg",
    "extract_lda_haufe_patterns",
    "run_classical_experiment",
    "run_classical_grid",
]
