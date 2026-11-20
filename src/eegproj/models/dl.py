"""Deep-learning classifiers for single-trial ERP classification (protocol §6): EEGNet and ShallowConvNet.

Both networks are written out in plain PyTorch, following the published architectures
(EEGNet-8,2: Lawhern et al., 2018; ShallowFBCSPNet: Schirrmeister et al., 2017; same layer layout as
braindecode's `EEGNetv4` / `ShallowFBCSPNet`). Keeping the layers explicit and named (`temporal`, `spatial`, ...)
lets the explainability step (R1) read the EEGNet spatial filters and run Integrated Gradients directly.

`TorchClassifier` wraps a network with a scikit-learn-like interface. Everything learned from data is fitted
on the training part of the current fold only:
  * channel-wise mean / std used to standardise the input,
  * balanced class weights of the loss,
  * early stopping on a validation subset carved out of the training data (never the test fold).
"""

import copy
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn

# ----------------------------------------------------------------------------------------------- networks


def _odd(n):
    """Round to the nearest odd integer >= 1 (odd kernels give symmetric 'same' padding)."""
    n = max(round(n), 1)
    return n if n % 2 else n + 1


class EEGNet(nn.Module):
    """EEGNet (Lawhern et al., 2018): temporal conv -> depthwise spatial conv -> separable conv -> linear.

    Input `x`: (batch, channels, samples), standardised EEG. Output: logits (batch, n_classes).
    `spatial.weight` has shape (F1 * D, 1, n_channels, 1): one scalp topography per spatial filter.
    """

    def __init__(
        self,
        n_channels,
        n_samples,
        sfreq,
        F1=8,
        D=2,
        F2=None,
        kernel_length_s=0.5,
        separable_kernel=16,
        pool1=4,
        pool2=8,
        drop_prob=0.25,
        n_classes=2,
    ):
        super().__init__()
        F2 = F2 or F1 * D
        kern = _odd(kernel_length_s * sfreq)  # 0.5 s -> 65 samples at 128 Hz (64 in the paper)
        self.temporal = nn.Sequential(nn.Conv2d(1, F1, (1, kern), padding="same", bias=False), nn.BatchNorm2d(F1))
        self.spatial = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.block1 = nn.Sequential(nn.BatchNorm2d(F1 * D), nn.ELU(), nn.AvgPool2d((1, pool1)), nn.Dropout(drop_prob))
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, _odd(separable_kernel)), padding="same", groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, pool2)),
            nn.Dropout(drop_prob),
        )
        n_out = n_samples // pool1 // pool2
        if n_out < 1:
            raise ValueError(f"EEGNet: {n_samples} samples is too short for pooling {pool1}x{pool2}.")
        self.classifier = nn.Linear(F2 * n_out, n_classes)

    def forward(self, x):
        x = self.spatial(self.temporal(x.unsqueeze(1)))
        return self.classifier(self.block2(self.block1(x)).flatten(1))

    def constrain(self):
        """Max-norm constraints of the original EEGNet: 1 on spatial filters, 0.25 on the dense layer."""
        with torch.no_grad():
            self.spatial.weight.copy_(torch.renorm(self.spatial.weight, p=2, dim=0, maxnorm=1.0))
            self.classifier.weight.copy_(torch.renorm(self.classifier.weight, p=2, dim=0, maxnorm=0.25))


class ShallowConvNet(nn.Module):
    """ShallowFBCSPNet (Schirrmeister et al., 2017): temporal conv -> spatial conv -> square -> mean pool -> log.

    Kernel / pooling lengths are given in seconds so the published values (25 / 75 / 15 samples at 250 Hz
    = 0.1 / 0.3 / 0.06 s) carry over to the DL sampling rate.
    """

    def __init__(
        self,
        n_channels,
        n_samples,
        sfreq,
        n_filters=40,
        filter_time_s=0.1,
        pool_time_s=0.3,
        pool_stride_s=0.06,
        drop_prob=0.5,
        n_classes=2,
    ):
        super().__init__()
        kern = max(round(filter_time_s * sfreq), 1)
        pool = max(round(pool_time_s * sfreq), 1)
        stride = max(round(pool_stride_s * sfreq), 1)
        pool = min(pool, n_samples - kern + 1)
        if pool < 1:
            raise ValueError(f"ShallowConvNet: {n_samples} samples is shorter than the temporal kernel ({kern}).")
        self.temporal = nn.Conv2d(1, n_filters, (1, kern))
        self.spatial = nn.Conv2d(n_filters, n_filters, (n_channels, 1), bias=False)
        self.norm = nn.BatchNorm2d(n_filters)
        self.pool = nn.AvgPool2d((1, pool), stride=(1, stride))
        self.drop = nn.Dropout(drop_prob)
        n_out = (n_samples - kern + 1 - pool) // stride + 1
        self.classifier = nn.Linear(n_filters * n_out, n_classes)

    def forward(self, x):
        x = self.norm(self.spatial(self.temporal(x.unsqueeze(1))))
        x = torch.log(torch.clamp(self.pool(x * x), min=1e-6))
        return self.classifier(self.drop(x).flatten(1))

    def constrain(self):
        """No weight constraint in the original ShallowFBCSPNet."""


NETWORKS = {"eegnet": EEGNet, "shallow": ShallowConvNet}

# --------------------------------------------------------------------------------------------- classifier


def channel_stats(X, idx, chunk=1024):
    """Channel-wise mean and std (µV) over trials `idx` and all time points; each of shape (1, channels, 1).

    Accumulated in float64 over chunks of trials, so a large cross-subject training set is not copied.
    """
    total = np.zeros(X.shape[1])
    total_sq = np.zeros(X.shape[1])
    for i in range(0, len(idx), chunk):
        block = np.asarray(X[idx[i : i + chunk]], dtype=np.float64)
        total += block.sum(axis=(0, 2))
        total_sq += (block**2).sum(axis=(0, 2))
    n = len(idx) * X.shape[2]
    mean = total / n
    std = np.sqrt(np.maximum(total_sq / n - mean**2, 0.0)) + 1e-6
    return mean[None, :, None].astype(np.float32), std[None, :, None].astype(np.float32)


def pick_device(device="auto"):
    """'auto' -> CUDA if available, else Apple MPS, else CPU."""
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TorchClassifier:
    """fit / predict_proba / decision_function around one of `NETWORKS`.

    Parameters
    ----------
    network : str
        Key of `NETWORKS` ('eegnet' or 'shallow').
    sfreq : float
        Sampling rate of the input (Hz); sets kernel lengths given in seconds.
    seed : int
        Seed for weight init, batch order and the fallback validation split.
    batch_size, max_epochs, lr, weight_decay, patience, val_fraction :
        Training hyper-parameters (`config['models']['dl']`).
    device : str
        'auto', 'cpu', 'cuda', ...
    **net_kwargs :
        Architecture hyper-parameters passed to the network.
    """

    def __init__(
        self,
        network,
        sfreq,
        seed,
        batch_size=64,
        max_epochs=200,
        lr=1e-3,
        weight_decay=0.0,
        patience=20,
        val_fraction=0.2,
        min_delta=1e-4,
        device="auto",
        **net_kwargs,
    ):
        if network not in NETWORKS:
            raise ValueError(f"Unknown network '{network}'. Expected one of {sorted(NETWORKS)}.")
        self.network, self.sfreq, self.seed = network, float(sfreq), int(seed)
        self.batch_size, self.max_epochs, self.lr = int(batch_size), int(max_epochs), float(lr)
        self.weight_decay, self.patience, self.val_fraction = float(weight_decay), int(patience), float(val_fraction)
        self.min_delta, self.device, self.net_kwargs = float(min_delta), device, dict(net_kwargs)

    # -- helpers
    def _batch(self, X, idx):
        """Standardise trials `idx` of X (µV) with the training statistics and move them to the device."""
        xb = (np.asarray(X[idx], dtype=np.float32) - self.mean_) / self.std_
        return torch.from_numpy(xb).to(self.device_)

    def _build(self, n_channels, n_samples):
        net = NETWORKS[self.network](n_channels, n_samples, self.sfreq, **self.net_kwargs)
        return net.to(self.device_)

    # -- training
    def fit(self, X, y, val_mask=None):
        """Train on X (trials, channels, samples; µV) and y (0/1).

        `val_mask` (bool, n_trials) marks the trials used only for early stopping. Callers pass a split that
        respects the CV structure (blocked in time within a subject, held-out subjects across subjects);
        if None, a stratified random split of `val_fraction` is used. Statistics, class weights and the
        weights of the network are learned from the remaining (fit) trials only.
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        if X.ndim != 3 or len(X) != len(y):
            raise ValueError(f"X must be (trials, channels, samples) matching y; got {X.shape} and {y.shape}.")
        if val_mask is None:
            fit_idx, val_idx = train_test_split(
                np.arange(len(y)), test_size=self.val_fraction, stratify=y, random_state=self.seed
            )
            self.val_scheme_ = "stratified_random"
        else:
            val_mask = np.asarray(val_mask, dtype=bool)
            fit_idx, val_idx = np.flatnonzero(~val_mask), np.flatnonzero(val_mask)
            self.val_scheme_ = "given"
        for name, idx in (("fit", fit_idx), ("validation", val_idx)):
            if len(np.unique(y[idx])) < 2:
                raise ValueError(f"The {name} part of the training data lacks one of the classes.")

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.device_ = pick_device(self.device)
        self.mean_, self.std_ = channel_stats(X, fit_idx)
        self.n_channels_, self.n_samples_ = X.shape[1], X.shape[2]
        self.net_ = self._build(self.n_channels_, self.n_samples_)

        counts = np.bincount(y[fit_idx], minlength=2)
        self.class_weight_ = (len(fit_idx) / (2.0 * counts)).astype(np.float32)  # 'balanced'
        loss_fn = nn.CrossEntropyLoss(weight=torch.as_tensor(self.class_weight_, device=self.device_))
        optim = torch.optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        y_t = torch.as_tensor(y, device=self.device_)

        best, best_state, best_epoch, wait = np.inf, None, -1, 0
        self.history_ = []
        start = time.monotonic()
        for epoch in range(self.max_epochs):
            self.net_.train()
            order = rng.permutation(fit_idx)
            train_loss, n_seen = 0.0, 0
            for i in range(0, len(order), self.batch_size):
                batch = np.sort(order[i : i + self.batch_size])
                if len(batch) < 2:  # BatchNorm needs > 1 sample
                    continue
                optim.zero_grad()
                loss = loss_fn(self.net_(self._batch(X, batch)), y_t[batch])
                loss.backward()
                optim.step()
                self.net_.constrain()
                train_loss += loss.item() * len(batch)
                n_seen += len(batch)
            val_loss = self._loss(X, y_t, val_idx, loss_fn)
            self.history_.append({"epoch": epoch, "train_loss": train_loss / max(n_seen, 1), "val_loss": val_loss})
            if val_loss < best - self.min_delta:
                best, best_epoch, wait = val_loss, epoch, 0
                best_state = copy.deepcopy(self.net_.state_dict())
            else:
                wait += 1
                if wait >= self.patience:
                    break
        self.net_.load_state_dict(best_state)
        self.net_.eval()
        self.best_epoch_, self.best_val_loss_ = best_epoch, float(best)
        self.n_epochs_, self.fit_seconds_ = len(self.history_), time.monotonic() - start
        self.n_fit_, self.n_val_ = len(fit_idx), len(val_idx)
        return self

    def _loss(self, X, y_t, idx, loss_fn):
        self.net_.eval()
        total = 0.0
        with torch.no_grad():
            for i in range(0, len(idx), 512):
                b = idx[i : i + 512]
                total += loss_fn(self.net_(self._batch(X, b)), y_t[b]).item() * len(b)
        return total / len(idx)

    # -- inference
    def logits(self, X):
        """Raw network outputs (trials, 2) for X in µV."""
        X = np.asarray(X, dtype=np.float32)
        self.net_.eval()
        with torch.no_grad():
            out = [self.net_(self._batch(X, np.arange(i, min(i + 512, len(X))))).cpu() for i in range(0, len(X), 512)]
        return torch.cat(out).numpy()

    def decision_function(self, X):
        """Logit difference (positive minus negative class); > 0 means 'positive'."""
        out = self.logits(X)
        return out[:, 1] - out[:, 0]

    def predict_proba(self, X):
        """Softmax probabilities (trials, 2); column 1 = positive class."""
        out = torch.from_numpy(self.logits(X))
        return torch.softmax(out, dim=1).numpy()

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(np.int64)

    def scale(self, X):
        """Standardised input tensor (as the network sees it) — for attribution methods."""
        return self._batch(np.asarray(X, dtype=np.float32), np.arange(len(X)))

    # -- persistence
    def save(self, path, **extra):
        """Save weights + everything needed to rebuild the model (no pickled code; loads with weights_only)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {k: v.detach().cpu() for k, v in self.net_.state_dict().items()}
        payload = {
            "format": "eegproj-dl-v1",
            "network": self.network,
            "sfreq": self.sfreq,
            "seed": self.seed,
            "net_kwargs": self.net_kwargs,
            "train_kwargs": {
                "batch_size": self.batch_size,
                "max_epochs": self.max_epochs,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "patience": self.patience,
                "val_fraction": self.val_fraction,
                "min_delta": self.min_delta,
            },
            "n_channels": int(self.n_channels_),
            "n_samples": int(self.n_samples_),
            "mean": torch.from_numpy(np.ascontiguousarray(self.mean_)),
            "std": torch.from_numpy(np.ascontiguousarray(self.std_)),
            "class_weight": torch.from_numpy(self.class_weight_),
            "best_epoch": int(self.best_epoch_),
            "n_epochs": int(self.n_epochs_),
            "history": self.history_,
            "state_dict": state,
            "extra": extra,
        }
        torch.save(payload, path)
        return path


def load_checkpoint(path, device="cpu"):
    """Rebuild a fitted `TorchClassifier` from a file written by `TorchClassifier.save`.

    Returns `(clf, info)` where `info` holds the saved metadata (`extra`: channel names, times in s,
    train/test subjects, fold, window, task ...).
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("format") != "eegproj-dl-v1":
        raise ValueError(f"{path} is not an eegproj DL checkpoint.")
    clf = TorchClassifier(
        payload["network"], payload["sfreq"], payload["seed"], device=device, **payload["train_kwargs"]
    )
    clf.net_kwargs = payload["net_kwargs"]
    clf.device_ = pick_device(device)
    clf.mean_, clf.std_ = payload["mean"].numpy(), payload["std"].numpy()
    clf.class_weight_ = payload["class_weight"].numpy()
    clf.n_channels_, clf.n_samples_ = payload["n_channels"], payload["n_samples"]
    clf.net_ = clf._build(clf.n_channels_, clf.n_samples_)
    clf.net_.load_state_dict(payload["state_dict"])
    clf.net_.eval()
    clf.best_epoch_, clf.n_epochs_, clf.history_ = payload["best_epoch"], payload["n_epochs"], payload["history"]
    info = {k: v for k, v in payload.items() if k not in ("state_dict", "mean", "std")}
    return clf, info


def make_dl(config, network, device="auto"):
    """Return a factory `seed -> TorchClassifier` configured from `config['models']['dl']`."""
    cfg = config["models"]["dl"]
    if network not in cfg:
        warnings.warn(f"No architecture settings for '{network}' in config; using defaults.", stacklevel=2)
    net_kwargs = dict(cfg.get(network) or {})

    def factory(seed):
        return TorchClassifier(
            network,
            cfg["sfreq"],
            seed,
            batch_size=cfg["batch_size"],
            max_epochs=cfg["max_epochs"],
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            patience=cfg["early_stopping_patience"],
            val_fraction=cfg["val_fraction"],
            min_delta=cfg.get("min_delta", 1e-4),
            device=device,
            **net_kwargs,
        )

    return factory
