"""Classical machine learning pipelines and Haufe pattern extraction (protocol §6 & §9.4)."""

import numpy as np
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

try:
    from pyriemann.geometry.covariance import covariances
except ImportError:  # pragma: no cover
    from pyriemann.utils.covariance import covariances

from ..features.time_windows import TimeWindowAverager

CLASSICAL_MODELS = ("lda_tw", "xdawn_rg")


class _RankSafeXdawnEstimator:
    """Covariance callable for xDAWN generalized eigenproblem on rank-deficient ICA-cleaned EEG.

    In `Xdawn.fit`, the evoked template `P` has shape `(n_channels, n_times)` whereas the
    concatenated signal `X` has shape `(n_channels, n_times * n_trials)`. On average-referenced,
    ICA-cleaned EEG (e.g., 127 channels with removed ICs), applying isotropic shrinkage to `P`
    injects positive variance into the null space where `X` has near-zero variance, corrupting
    `eigh(C_P, C_X)`. Using `'scm'` when `x.shape[1] <= max_evoked_samples` (`P`) and `estimator`
    (e.g., `'lwf'`) for concatenated trials (`X`) preserves well-conditioned SPD estimation.

    Parameters
    ----------
    estimator : str, default='lwf'
        Covariance estimator name for concatenated signal matrix `X`.
    max_evoked_samples : int, default=1000
        Sample-count threshold distinguishing evoked template `P` (`<= 1000`) from `X`.
    """

    def __init__(self, estimator: str = "lwf", max_evoked_samples: int = 1000):
        self.estimator = estimator
        self.max_evoked_samples = max_evoked_samples

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Estimate `(n_channels, n_channels)` covariance in µV² from 2D EEG matrix `x` (µV)."""
        est = "scm" if x.shape[1] <= self.max_evoked_samples else self.estimator
        return covariances(x[None, ...], estimator=est)[0]


def build_lda_tw(times: np.ndarray, config: dict) -> Pipeline:
    """Build leakage-free `lda_tw` pipeline: bin mean (µV) -> StandardScaler -> Shrinkage LDA.

    Parameters
    ----------
    times : np.ndarray of shape (n_times,)
        Epoch time points in seconds (s).
    config : dict
        Parsed `configs/config.yaml` providing `bin_ms` (ms), `shrinkage`, and `class_balance`.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted pipeline expecting 3D epochs `(n_trials, n_channels, n_times)` in microvolts (µV).
    """
    cfg_lda = config["models"]["classical"]["lda_tw"]
    bin_ms = float(cfg_lda["bin_ms"])
    shrinkage = cfg_lda["shrinkage"]
    balance_mode = config["classification"].get("class_balance", "class_weight")
    priors = [0.5, 0.5] if balance_mode == "class_weight" else None

    return Pipeline(
        [
            ("time_windows", TimeWindowAverager(times=np.asarray(times, dtype=float), bin_ms=bin_ms)),
            ("scaler", StandardScaler()),
            ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage=shrinkage, priors=priors)),
        ]
    )


def build_xdawn_rg(config: dict) -> Pipeline:
    """Build leakage-free `xdawn_rg` pipeline: xDAWN covariances -> TangentSpace -> LogisticRegression.

    Parameters
    ----------
    config : dict
        Parsed `configs/config.yaml` providing `nfilter`, `estimator`, `C`, `seed`, and `class_balance`.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted pipeline expecting 3D epochs `(n_trials, n_channels, n_times)` in microvolts (µV).
    """
    cfg_xdawn = config["models"]["classical"]["xdawn_rg"]
    nfilter = int(cfg_xdawn["nfilter"])
    estimator = str(cfg_xdawn["estimator"])
    c_val = float(cfg_xdawn["C"])
    seed = int(config["seed"])
    balance_mode = config["classification"].get("class_balance", "class_weight")
    class_weight = "balanced" if balance_mode == "class_weight" else None

    return Pipeline(
        [
            (
                "xdawn_cov",
                XdawnCovariances(
                    nfilter=nfilter,
                    estimator=estimator,
                    xdawn_estimator=_RankSafeXdawnEstimator(estimator=estimator),
                ),
            ),
            ("tangent", TangentSpace(metric="riemann")),
            (
                "clf",
                LogisticRegression(
                    C=c_val,
                    class_weight=class_weight,
                    max_iter=1000,
                    random_state=seed,
                ),
            ),
        ]
    )


def build_classical_model(model_name: str, config: dict, times: np.ndarray) -> Pipeline:
    """Instantiate an unfitted classical ML pipeline (`'lda_tw'` or `'xdawn_rg'`) expecting µV input.

    Parameters
    ----------
    model_name : str
        Model identifier (`'lda_tw'` or `'xdawn_rg'`).
    config : dict
        Parsed `configs/config.yaml`.
    times : np.ndarray of shape (n_times,)
        Epoch time points in seconds (s).

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted classification pipeline expecting 3D epochs in microvolts (µV).
    """
    if model_name == "lda_tw":
        return build_lda_tw(times=times, config=config)
    if model_name == "xdawn_rg":
        return build_xdawn_rg(config=config)
    raise ValueError(f"Unknown classical model '{model_name}'. Expected one of {CLASSICAL_MODELS}.")


def extract_lda_haufe_patterns(pipeline: Pipeline, X_train_uv: np.ndarray) -> dict[str, np.ndarray]:
    """Compute forward activation patterns (Haufe et al., 2014) for a fitted `lda_tw` pipeline.

    Implements Eq. (7) of Haufe et al. (2014), :math:`A = \\Sigma_Z w / \\operatorname{Var}(Z w)`,
    mapping classifier weights back into interpretable spatial-temporal patterns in microvolts (µV).

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Fitted `lda_tw` pipeline containing `'time_windows'`, `'scaler'`, and `'clf'` steps.
    X_train_uv : np.ndarray of shape (n_trials, n_channels, n_times)
        Training epochs in microvolts (µV) used to fit the pipeline.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary containing:
        - `'pattern_uv'`: forward activation pattern of shape `(n_channels, n_bins)` in microvolts (µV),
        - `'weights'`: effective unscaled linear weights of shape `(n_channels, n_bins)` in `1/µV`,
        - `'bin_edges_s'`: time-bin boundary timestamps of shape `(n_bins + 1,)` in seconds (s).
    """
    tw: TimeWindowAverager = pipeline.named_steps["time_windows"]
    scaler: StandardScaler = pipeline.named_steps["scaler"]
    clf: LinearDiscriminantAnalysis = pipeline.named_steps["clf"]
    check_is_fitted(clf)

    Z = tw.transform(X_train_uv)
    if Z.shape[0] < 2:
        raise ValueError(f"Need at least 2 training trials to estimate Haufe covariance, got {Z.shape[0]}.")

    scale = np.where(scaler.scale_ > 0, scaler.scale_, 1.0)
    w = clf.coef_.ravel() / scale

    Z_centered = Z - Z.mean(axis=0, keepdims=True)
    s_hat = Z_centered @ w
    var_s = float(np.var(s_hat, ddof=1))
    if var_s <= np.finfo(float).eps:
        pattern_flat = np.zeros_like(w)
    else:
        cov_z_s = (Z_centered.T @ s_hat) / (len(Z) - 1)
        pattern_flat = cov_z_s / var_s

    shape_2d = (tw.n_channels_, tw.n_bins_)
    return {
        "pattern_uv": pattern_flat.reshape(shape_2d),
        "weights": w.reshape(shape_2d),
        "bin_edges_s": tw.bin_edges_.copy(),
    }
