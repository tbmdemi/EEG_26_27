"""Run the deep-learning models over the protocol grid (setting x window x task) — protocol §4–§7.

Uses the shared evaluation code of R4 (`evaluation.splits`, `evaluation.metrics`) and the fixed cross-subject
folds in `results/cv_cross_subject_folds.csv` (`evaluation.folds`), so DL and classical models are scored on
identical splits and written in the identical CSV format.

Outputs (all git-ignored):
    results/metrics/<model>_<setting>_<window>_<task>.csv        metrics, protocol §7
    results/predictions/<model>_<setting>_<window>_<task>.csv.gz per-trial test scores (stats / permutation tests)
    results/dl_runs/<model>_<setting>_<window>_<task>.json       hyper-parameters, epochs, timing per fold
    results/checkpoints/<model>/<setting>_<window>_<task>/*.pt   fitted models for explainability (R1)
An interrupted run resumes fold by fold from results/metrics/<...>.partial.json.
"""

import json
import platform
import subprocess
import time
import warnings
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import torch

from ..evaluation.folds import cross_subject_folds, within_cv_metadata
from ..evaluation.metrics import compute_binary_metrics, make_metric_row, metrics_csv_path, save_metrics_csv
from ..evaluation.splits import epochs_path, format_bids_subject, iter_within_folds, load_valid_subjects
from .dl import NETWORKS, make_dl, pick_device

# ------------------------------------------------------------------------------------------------- data


class _TrialTable:
    """Minimal Epochs-like view (len + metadata) so arrays can go through the shared `iter_within_folds`."""

    def __init__(self, metadata):
        self.metadata = metadata.reset_index(drop=True)

    def __len__(self):
        return len(self.metadata)


def load_dl_subject(root, config, subject, task_name, sfreq=None):
    """Load one subject's correct trials for a binary task, resampled for DL and cropped to the union of windows.

    The whole epoch (-0.2 to 0.8 s) is resampled first (to `config['models']['dl']['sfreq']`, 128 Hz) and only
    then cropped, so the resampling has no edge effect inside the analysis windows.

    Returns
    -------
    dict with
        X : float32 array (trials, channels, samples), µV
        y : int64 array (trials,), 1 = positive class of the task
        times : float array (samples,), s relative to stimulus onset
        metadata : DataFrame (subject, onset_s, block, rt), chronological
        ch_names : list of str
    """
    spec = config["classification"]["tasks"][task_name]
    pos_name, neg_name = spec["positive"], spec["negative"]
    sfreq = float(sfreq or config["models"]["dl"]["sfreq"])
    path = epochs_path(root, config, subject)
    if not path.exists():
        raise FileNotFoundError(f"Preprocessed epochs not found: {path}. Run scripts/00_get_data.py first.")
    epochs = mne.read_epochs(path, preload=True, verbose="error")
    if epochs.metadata is None or "rt" not in epochs.metadata.columns:
        raise ValueError(f"{path.name} has no 'rt' metadata: this is a v1 file; re-download data v2.")
    epochs = epochs[[neg_name, pos_name]]
    if not epochs.metadata["onset_s"].is_monotonic_increasing:
        raise ValueError(f"Epochs of subject {subject} are not in chronological order.")
    if epochs.info["sfreq"] != sfreq:
        epochs.resample(sfreq, verbose="error")
    windows = config["classification"]["windows"].values()
    tmin, tmax = min(w[0] for w in windows), max(w[1] for w in windows)
    epochs.crop(tmin=tmin, tmax=tmax, include_tmax=True)
    y = (epochs.events[:, 2] == epochs.event_id[pos_name]).astype(np.int64)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError(f"Subject {subject} task {task_name} is missing one of the classes.")
    X = (epochs.get_data(picks="eeg") * 1e6).astype(np.float32)
    return {
        "X": X,
        "y": y,
        "times": epochs.times.copy(),
        "metadata": epochs.metadata.reset_index(drop=True),
        "ch_names": [epochs.ch_names[i] for i in mne.pick_types(epochs.info, eeg=True)],
    }


def align_channels(subject_data, ch_ref):
    """Reorder `subject_data['X']` rows (in place) so its channels follow the names in `ch_ref`.

    Some recordings store the same 127 electrodes in a different order (sub-13, sub-42 in data v2). Stacking
    subjects by array position would then mix electrodes, so channels are matched by name. Raises if the
    channel sets differ.
    """
    names = subject_data["ch_names"]
    if set(names) != set(ch_ref) or len(names) != len(ch_ref):
        missing, extra = sorted(set(ch_ref) - set(names)), sorted(set(names) - set(ch_ref))
        raise ValueError(f"Channel set differs from the reference (missing {missing}, extra {extra}).")
    order = [names.index(c) for c in ch_ref]
    subject_data["X"] = np.ascontiguousarray(subject_data["X"][:, order, :])
    subject_data["ch_names"] = list(ch_ref)
    return subject_data


def window_mask(times, config, window_name):
    """Boolean mask of samples inside `config['classification']['windows'][window_name]` (inclusive, s)."""
    tmin, tmax = config["classification"]["windows"][window_name]
    return (times >= tmin - 1e-9) & (times <= tmax + 1e-9)


# ----------------------------------------------------------------------------------- validation splits


def within_validation_mask(metadata, train_idx, y, val_fraction):
    """Early-stopping set for within-subject CV: the last `val_fraction` of the trials of every training block.

    Keeps validation trials in contiguous stretches (like the test fold) instead of interleaving them with
    fit trials, so temporally correlated neighbours rarely sit on both sides. Returns a bool mask over
    `train_idx`, or None if a part would miss a class (caller then falls back to a stratified random split).
    """
    blocks = (
        metadata["block"].to_numpy()[train_idx] if "block" in metadata.columns else np.zeros(len(train_idx), dtype=int)
    )
    mask = np.zeros(len(train_idx), dtype=bool)
    for b in np.unique(blocks):
        pos = np.flatnonzero(blocks == b)
        n_val = int(np.ceil(val_fraction * len(pos)))
        mask[pos[len(pos) - n_val :]] = True
    y_train = y[train_idx]
    if len(np.unique(y_train[mask])) < 2 or len(np.unique(y_train[~mask])) < 2:
        return None
    return mask


def cross_validation_subjects(train_subjects, val_fraction, seed):
    """Early-stopping subjects for cross-subject CV: a seeded random `val_fraction` of the training subjects."""
    rng = np.random.default_rng(seed)
    n_val = min(max(1, round(val_fraction * len(train_subjects))), len(train_subjects) - 1)
    return sorted(rng.choice(sorted(train_subjects), size=n_val, replace=False).tolist())


# ------------------------------------------------------------------------------------------------ runner


def _git_commit(root):
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _paths(root, config, model, setting, window, task, dev=False):
    """Output paths of one combination. `dev=True` (subject subset) keeps everything under results/dev/."""
    name = f"{model}_{setting}_{window}_{task}"
    root = Path(root)
    if dev:
        base = root / "results" / "dev"
        metrics, checkpoints = base / "metrics" / f"{name}.csv", base / "checkpoints"
    else:
        base = root / "results"
        metrics, checkpoints = (
            metrics_csv_path(root, config, model, setting, window, task),
            root / config["paths"]["checkpoints"],
        )
    return {
        "name": name,
        "metrics": metrics,
        "partial": metrics.with_suffix(".partial.json"),
        "predictions": base / "predictions" / f"{name}.csv.gz",
        "runinfo": base / "dl_runs" / f"{name}.json",
        "checkpoints": checkpoints / model / f"{setting}_{window}_{task}",
    }


def _save_partial(path, state):
    """Write progress after every fold so an interrupted run can resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state), encoding="utf-8")
    tmp.replace(path)


def _fold_record(clf, **extra):
    return {
        "n_fit": int(clf.n_fit_),
        "n_val": int(clf.n_val_),
        "val_scheme": clf.val_scheme_,
        "best_epoch": int(clf.best_epoch_),
        "n_epochs": int(clf.n_epochs_),
        "best_val_loss": float(clf.best_val_loss_),
        "seconds": round(float(clf.fit_seconds_), 1),
        **extra,
    }


def _predictions(subject, fold, meta, y, proba, decision):
    return pd.DataFrame(
        {
            "subject": format_bids_subject(subject),
            "fold": int(fold),
            "onset_s": meta["onset_s"].to_numpy(),
            "block": meta["block"].to_numpy() if "block" in meta.columns else -1,
            "rt": meta["rt"].to_numpy() if "rt" in meta.columns else np.nan,
            "y_true": y,
            "prob_pos": proba,
            "decision": decision,
        }
    )


def run_dl_grid(
    root,
    config,
    model,
    settings=("within", "cross"),
    windows=("full", "early"),
    tasks=("tgt_std",),
    subjects=None,
    device="auto",
    overwrite=False,
    save_checkpoints=True,
    allow_fewer_splits=False,
    log=print,
):
    """Train and evaluate `model` ('eegnet' or 'shallow') for every (task, setting, window).

    One CSV per combination (protocol §7). Data of a task are loaded once for all windows / settings.
    `subjects`: optional subset for quick checks; its outputs go to results/dev/ and the fold file is not used.
    Returns a list of the metrics files written (or found complete when `overwrite` is False).
    """
    if model not in NETWORKS:
        raise ValueError(f"Unknown DL model '{model}'. Expected one of {sorted(NETWORKS)}.")
    root = Path(root)
    seed = int(config["seed"])
    dl_cfg = config["models"]["dl"]
    make_model = make_dl(config, model, device=device)
    all_subjects = load_valid_subjects(root, config)
    subjects = load_valid_subjects(root, config, subjects=subjects)
    dev = subjects != all_subjects  # a subset: never touch the real results or the shared fold file
    if dev:
        log(f"subject subset {subjects}: outputs go to results/dev/ (not used for the report)")
    written = []

    for task in tasks:
        todo = [
            (s, w)
            for s in settings
            for w in windows
            if overwrite or not _paths(root, config, model, s, w, task, dev)["metrics"].exists()
        ]
        for s, w in sorted({(s, w) for s in settings for w in windows} - set(todo)):
            path = _paths(root, config, model, s, w, task, dev)["metrics"]
            log(f"skip {path.name} (exists; use --overwrite to recompute)")
            written.append(path)
        if not todo:
            continue
        log(f"[{model}] loading {len(subjects)} subjects for task {task} ...")
        data = {s: load_dl_subject(root, config, s, task) for s in subjects}
        ch_ref = data[subjects[0]]["ch_names"]
        for s in subjects:
            if data[s]["ch_names"] != ch_ref:
                align_channels(data[s], ch_ref)
                log(f"  sub-{s}: channels reordered by name to match sub-{subjects[0]}")
        times = data[subjects[0]]["times"]

        for setting, window in todo:
            paths = _paths(root, config, model, setting, window, task, dev)
            tmask = window_mask(times, config, window)
            within_split = config["evaluation"]["within"].get("split", "s10_blocks")
            state = {"rows": [], "predictions": [], "folds": [], "within_split": within_split}
            if paths["partial"].exists() and not overwrite:
                saved = json.loads(paths["partial"].read_text(encoding="utf-8"))
                if setting == "cross" or saved.get("within_split") == within_split:
                    state = saved
                else:
                    log(f"  {paths['partial'].name}: made with another within split; starting this run again")
            done = {(r["subject"], r["fold"]) for r in state["rows"]}
            start = time.monotonic()
            log(f"[{paths['name']}] {tmask.sum()} samples ({times[tmask][0]:.3f}-{times[tmask][-1]:.3f} s)")

            if setting == "within":
                for sub in subjects:
                    d = data[sub]
                    X, y, meta = d["X"][:, :, tmask], d["y"], d["metadata"]
                    cv_meta = within_cv_metadata(meta, config)  # 'block' = CV segment (config evaluation.within)
                    for fold, train_idx, test_idx in iter_within_folds(_TrialTable(cv_meta), y, config=config):
                        if (format_bids_subject(sub), fold) in done:
                            continue
                        val = within_validation_mask(cv_meta, train_idx, y, dl_cfg["val_fraction"])
                        if val is None:
                            warnings.warn(
                                f"sub-{sub} fold {fold}: blocked validation lacks a class; "
                                "using a stratified random split.",
                                RuntimeWarning,
                                stacklevel=2,
                            )
                        clf = make_model(seed + fold).fit(X[train_idx], y[train_idx], val_mask=val)
                        if val is not None:
                            clf.val_scheme_ = "blocked_tail"
                        proba = clf.predict_proba(X[test_idx])[:, 1]
                        y_pred = (proba > 0.5).astype(np.int64)
                        metrics = compute_binary_metrics(y[test_idx], proba, y_pred)
                        state["rows"].append(
                            make_metric_row(
                                subject=sub,
                                fold=fold,
                                model=model,
                                family="dl",
                                setting=setting,
                                window=window,
                                task=task,
                                metrics=metrics,
                                n_train=len(train_idx),
                                n_test=len(test_idx),
                                seed=seed,
                            )
                        )
                        pred = _predictions(
                            sub, fold, meta.iloc[test_idx], y[test_idx], proba, clf.decision_function(X[test_idx])
                        )
                        state["predictions"].extend(pred.to_dict("records"))
                        state["folds"].append(_fold_record(clf, subject=format_bids_subject(sub), fold=fold))
                        if save_checkpoints:
                            clf.save(
                                paths["checkpoints"] / f"sub-{sub}_fold-{fold}.pt",
                                setting=setting,
                                window=window,
                                task=task,
                                fold=fold,
                                subject=format_bids_subject(sub),
                                ch_names=ch_ref,
                                times=times[tmask].tolist(),
                                test_index=test_idx.tolist(),
                            )
                        _save_partial(paths["partial"], state)
                        log(
                            f"  sub-{sub} fold {fold}: bal_acc={metrics['bal_acc']:.3f} auc={metrics['auc']:.3f} "
                            f"epochs={clf.n_epochs_} ({clf.fit_seconds_:.0f}s)"
                        )
            else:
                folds = cross_subject_folds(
                    root, config, subjects, allow_fewer_splits=allow_fewer_splits, use_file=not dev
                )
                for fold, train_subs, test_subs in folds:
                    if all((format_bids_subject(s), fold) in done for s in test_subs):
                        continue
                    X_train = np.concatenate([data[s]["X"][:, :, tmask] for s in train_subs])
                    y_train = np.concatenate([data[s]["y"] for s in train_subs])
                    if len(train_subs) >= 2:
                        val_subs = cross_validation_subjects(train_subs, dl_cfg["val_fraction"], seed + fold)
                        val_mask = np.concatenate([np.full(len(data[s]["y"]), s in val_subs) for s in train_subs])
                        scheme = "held_out_subjects"
                    else:  # dev runs on 2 subjects only: blocked tail of the single training subject
                        val_subs, d0 = [], data[train_subs[0]]
                        val_mask = within_validation_mask(
                            d0["metadata"], np.arange(len(d0["y"])), d0["y"], dl_cfg["val_fraction"]
                        )
                        scheme = "blocked_tail"
                    clf = make_model(seed + fold).fit(X_train, y_train, val_mask=val_mask)
                    clf.val_scheme_ = scheme if val_mask is not None else clf.val_scheme_
                    del X_train
                    for sub in test_subs:
                        d = data[sub]
                        X, y = d["X"][:, :, tmask], d["y"]
                        proba = clf.predict_proba(X)[:, 1]
                        metrics = compute_binary_metrics(y, proba, (proba > 0.5).astype(np.int64))
                        state["rows"].append(
                            make_metric_row(
                                subject=sub,
                                fold=fold,
                                model=model,
                                family="dl",
                                setting=setting,
                                window=window,
                                task=task,
                                metrics=metrics,
                                n_train=len(y_train),
                                n_test=len(y),
                                seed=seed,
                            )
                        )
                        pred = _predictions(sub, fold, d["metadata"], y, proba, clf.decision_function(X))
                        state["predictions"].extend(pred.to_dict("records"))
                        log(f"  fold {fold} test sub-{sub}: bal_acc={metrics['bal_acc']:.3f} auc={metrics['auc']:.3f}")
                    state["folds"].append(
                        _fold_record(
                            clf,
                            fold=fold,
                            train_subjects=[format_bids_subject(s) for s in train_subs],
                            val_subjects=[format_bids_subject(s) for s in val_subs],
                            test_subjects=[format_bids_subject(s) for s in test_subs],
                        )
                    )
                    if save_checkpoints:
                        clf.save(
                            paths["checkpoints"] / f"fold-{fold}.pt",
                            setting=setting,
                            window=window,
                            task=task,
                            fold=fold,
                            train_subjects=[format_bids_subject(s) for s in train_subs],
                            val_subjects=[format_bids_subject(s) for s in val_subs],
                            test_subjects=[format_bids_subject(s) for s in test_subs],
                            ch_names=ch_ref,
                            times=times[tmask].tolist(),
                        )
                    _save_partial(paths["partial"], state)
                    log(f"  fold {fold}: epochs={clf.n_epochs_} best={clf.best_epoch_} ({clf.fit_seconds_:.0f}s)")

            rows = pd.DataFrame(state["rows"]).sort_values(["subject", "fold"]).reset_index(drop=True)
            save_metrics_csv(rows, paths["metrics"])
            paths["predictions"].parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(state["predictions"]).to_csv(paths["predictions"], index=False)
            runinfo = {
                "model": model,
                "family": "dl",
                "setting": setting,
                "window": window,
                "task": task,
                "seed": seed,
                "subjects": [format_bids_subject(s) for s in subjects],
                "n_samples": int(tmask.sum()),
                "sfreq": float(dl_cfg["sfreq"]),
                "window_s": [float(times[tmask][0]), float(times[tmask][-1])],
                "n_channels": len(ch_ref),
                "within_split": within_split if setting == "within" else None,
                "hyperparameters": {k: v for k, v in dl_cfg.items() if not isinstance(v, dict)},
                "architecture": dl_cfg.get(model, {}),
                "folds": state["folds"],
                "git_commit": _git_commit(root),
                "torch": torch.__version__,
                "mne": mne.__version__,
                "python": platform.python_version(),
                "device": str(pick_device(device)),
                "seconds_this_session": round(time.monotonic() - start, 1),
            }
            paths["runinfo"].parent.mkdir(parents=True, exist_ok=True)
            paths["runinfo"].write_text(json.dumps(runinfo, indent=2), encoding="utf-8")
            paths["partial"].unlink(missing_ok=True)
            summary = rows.groupby("subject")[["bal_acc", "auc"]].mean()
            log(
                f"[{paths['name']}] done: bal_acc median={summary.bal_acc.median():.3f} "
                f"mean={summary.bal_acc.mean():.3f}, auc mean={summary.auc.mean():.3f} "
                f"over {len(summary)} subjects ({time.monotonic() - start:.0f}s)"
            )
            written.append(paths["metrics"])
        del data
    return written


def summarise_dl(root, config, models=tuple(NETWORKS), tasks=("tgt_std",)):
    """Per-subject summary of finished DL runs (protocol §8 unit: within = mean over folds; cross = test score).

    Returns a DataFrame with one row per (model, setting, window, task): n_subjects, median / mean / SD of balanced
    accuracy, mean / SD of AUC, and the median number of training epochs from results/dl_runs/.
    """
    out = []
    for model in models:
        for task in tasks:
            for setting in ("within", "cross"):
                for window in config["classification"]["windows"]:
                    paths = _paths(root, config, model, setting, window, task)
                    if not paths["metrics"].exists():
                        continue
                    per_subject = pd.read_csv(paths["metrics"]).groupby("subject")[["bal_acc", "auc"]].mean()
                    epochs = np.nan
                    if paths["runinfo"].exists():
                        info = json.loads(paths["runinfo"].read_text(encoding="utf-8"))
                        epochs = float(np.median([f["n_epochs"] for f in info["folds"]]))
                    out.append(
                        {
                            "model": model,
                            "setting": setting,
                            "window": window,
                            "task": task,
                            "n_subjects": len(per_subject),
                            "bal_acc_median": per_subject.bal_acc.median(),
                            "bal_acc_mean": per_subject.bal_acc.mean(),
                            "bal_acc_sd": per_subject.bal_acc.std(ddof=1),
                            "auc_mean": per_subject.auc.mean(),
                            "auc_sd": per_subject.auc.std(ddof=1),
                            "epochs_median": epochs,
                        }
                    )
    return pd.DataFrame(out)
