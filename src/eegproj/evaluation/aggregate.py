"""Validate R4/R5 inputs and prepare subject-level F3 summaries (protocol §§7–10).

No models are fitted and no CV assignment is created here. Cross folds are
verified across metrics and against current GroupKFold; a supplied fold CSV is
checked when present. A metrics-derived assignment is not an original manifest.
"""

import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from .folds import folds_csv_path
from .metrics import metrics_csv_path, validate_metrics_df
from .splits import format_bids_subject, iter_cross_folds, load_valid_subjects
from .stats import aggregate_subject_scores, load_all_metrics

F3_MODELS = {"lda_tw": "classical", "xdawn_rg": "classical", "eegnet": "dl", "shallow": "dl"}
F3_SETTINGS = ("within", "cross")
F3_WINDOWS = ("full", "early")
F3_TASK = "tgt_std"
F3_METRICS = ("bal_acc", "auc", "f1_target")
_CONDITION = ["model", "setting", "window", "task"]
_SUBJECT_KEY = [*_CONDITION, "subject"]


def _conditions(models=None):
    return product(F3_MODELS if models is None else models, F3_SETTINGS, F3_WINDOWS)


def _integers(series, label, minimum=0):
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < minimum).any() or (values != np.floor(values)).any():
        raise ValueError(f"{label}: expected finite integers >= {minimum}")
    return values.astype(np.int64)


def _validate_subject_grid(df, subjects, *, task_name=F3_TASK, models=None):
    models = F3_MODELS if models is None else models
    if df.empty or df[_SUBJECT_KEY].isna().any().any():
        raise ValueError("F3 requires non-empty, non-missing subject/condition keys")
    if df.duplicated(_SUBJECT_KEY).any():
        raise ValueError("F3 requires exactly one score per subject and condition; average within folds first")
    expected = {(m, s, w, task_name) for m, s, w in _conditions(models)}
    if set(df[_CONDITION].itertuples(index=False, name=None)) != expected:
        raise ValueError(f"{task_name}: requires all {len(expected)} model/setting/window combinations")
    for key, group in df.groupby(_CONDITION):
        if set(group["subject"]) != subjects:
            raise ValueError(f"{key}: cohort mismatch (missing/extra subjects)")
        if not group["family"].eq(models[key[0]]).all():
            raise ValueError(f"{key}: model/family mismatch")
    scores = df[list(F3_METRICS)].to_numpy(dtype=float)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("F3 metrics must be finite proportions in [0, 1]; no rows are dropped or imputed")


def validate_f3_metrics(df, config, subjects, cross_folds, *, task_name=F3_TASK, models=None):
    """Return a validated copy of raw metrics; reject missing subjects/folds/NaNs.

    ``subjects`` contains BIDS IDs from the v2 handoff; ``cross_folds`` is the
    verified subject/fold table. Metrics are dimensionless proportions [0, 1].
    Optional task/model arguments also audit the supplied control-task grids.
    """
    table = df.copy()
    models = F3_MODELS if models is None else models
    for column in (*F3_METRICS, "fold", "seed", "n_train", "n_test"):
        if column in table:
            table[column] = pd.to_numeric(table[column], errors="coerce")
    validate_metrics_df(table)
    for column in ("fold", "seed", "n_train", "n_test"):
        table[column] = _integers(table[column], column, minimum=1 if column.startswith("n_") else 0)
    if not table["seed"].eq(config["seed"]).all():
        raise ValueError("F3 seed differs from config")
    expected_subjects = set(subjects)
    if not expected_subjects or len(expected_subjects) != len(subjects):
        raise ValueError("F3 expected cohort must be non-empty and unique")
    if table.duplicated([*_SUBJECT_KEY, "fold"]).any():
        raise ValueError("Duplicate subject/fold/condition metric rows")
    if not table["family"].eq(table["model"].map(models)).all():
        raise ValueError("F3 model/family mismatch")
    # Check the complete grid without using means (which could hide missing data).
    _validate_subject_grid(table.drop_duplicates(_SUBJECT_KEY), expected_subjects, task_name=task_name, models=models)
    scores = table[list(F3_METRICS)].to_numpy(dtype=float)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("F3 metrics must be finite proportions in [0, 1]; no rows are dropped or imputed")

    if list(cross_folds.columns) != ["subject", "fold"]:
        raise ValueError("Cross-fold CSV must have columns subject,fold")
    folds = cross_folds.copy()
    if folds["subject"].duplicated().any() or set(folds["subject"]) != expected_subjects:
        raise ValueError("Cross-fold CSV cohort mismatch or duplicate subject")
    folds["fold"] = _integers(folds["fold"], "cross fold")
    if set(folds["fold"]) != set(range(config["evaluation"]["cross"]["n_splits"])):
        raise ValueError("Cross-fold CSV does not cover all configured folds")
    assignment = folds.set_index("subject")["fold"]
    for key, group in table.groupby(_SUBJECT_KEY):
        if key[1] == "within":
            if set(group["fold"]) != set(range(config["evaluation"]["within"]["n_splits"])):
                raise ValueError(f"{key}: missing or unexpected within folds")
        elif len(group) != 1 or group["fold"].iloc[0] != assignment[key[-1]]:
            raise ValueError(f"{key}: cross requires one test-subject row matching the shared fold CSV")
    # All four models/windows must have evaluated the same test trials.
    if table.groupby(["setting", "subject", "fold"])["n_test"].nunique().gt(1).any():
        raise ValueError("n_test differs between models/windows on the same subject/fold")
    totals = table.groupby(_SUBJECT_KEY)["n_test"].sum().reset_index()
    if totals.groupby("subject")["n_test"].nunique().gt(1).any():
        raise ValueError("Within/cross test-trial totals differ for a subject")
    return table


def verify_cross_folds(df, config, subjects, supplied_folds=None):
    """Verify every cross-condition against each other and current GroupKFold.

    Returns a metrics-derived subject/fold table and explicit verification
    evidence. Never writes ``cv_cross_subject_folds.csv`` or claims historical
    training provenance. A supplied fold CSV must agree when available.
    """
    cross = df.loc[df["setting"].eq("cross")].copy()
    if cross.empty:
        raise ValueError("No cross-subject metrics available to verify folds")
    cross["fold"] = _integers(cross["fold"], "cross metric fold")
    expected_subjects = set(subjects)
    reference = None
    n_conditions = 0
    for key, group in cross.groupby(["model", "window", "task"], dropna=False):
        if group["subject"].duplicated().any() or set(group["subject"]) != expected_subjects:
            raise ValueError(f"{key}: cross metrics must contain each cohort subject exactly once")
        assignment = group.set_index("subject")["fold"].sort_index()
        if set(assignment) != set(range(config["evaluation"]["cross"]["n_splits"])):
            raise ValueError(f"{key}: cross metrics do not cover all configured folds")
        if reference is None:
            reference = assignment
        elif not assignment.equals(reference):
            raise ValueError(f"{key}: inconsistent cross-subject folds between metric conditions")
        n_conditions += 1
    if supplied_folds is not None:
        if list(supplied_folds.columns) != ["subject", "fold"]:
            raise ValueError("Supplied cross-fold CSV must have columns subject,fold")
        supplied = supplied_folds.copy()
        supplied["fold"] = _integers(supplied["fold"], "supplied cross fold")
        if supplied["subject"].duplicated().any() or set(supplied["subject"]) != expected_subjects:
            raise ValueError("Supplied cross-fold CSV cohort mismatch or duplicate subject")
        if not supplied.set_index("subject")["fold"].sort_index().equals(reference):
            raise ValueError("Supplied cross-fold CSV disagrees with metrics")
    current = {}
    split_details = []
    for fold, train, test in iter_cross_folds(subjects, config=config):
        current.update({format_bids_subject(subject): fold for subject in test})
        split_details.append(
            {
                "fold": fold,
                "train_subjects": [format_bids_subject(s) for s in train],
                "test_subjects": [format_bids_subject(s) for s in test],
            }
        )
    expected = pd.Series(current, name="fold", dtype="int64").sort_index()
    if not reference.equals(expected):
        mismatch = reference.index[reference.ne(expected)].tolist()
        raise ValueError(f"Metrics cross-subject folds differ from current GroupKFold for {mismatch}")
    evidence = {
        "assignment_source": "metrics_consensus",
        "supplied_fold_csv_present": supplied_folds is not None,
        "is_original_r4_manifest": False,
        "all_cross_conditions_agree": True,
        "matches_current_groupkfold": True,
        "n_cross_conditions": n_conditions,
        "n_cross_rows": len(cross),
        "splitter": "eegproj.evaluation.splits.iter_cross_folds: GroupKFold(shuffle=False)",
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
        "splitter_source_sha256": hashlib.sha256(Path(__file__).with_name("splits.py").read_bytes()).hexdigest(),
        "historical_training_provenance": "not established by metric CSVs",
        "verified_current_splits": split_details,
    }
    return reference.rename_axis("subject").reset_index(), evidence


def _load_handoff(root, config):
    log = pd.read_csv(root / "results" / "preproc_log.csv")
    required = {"subject", "pipeline_version", "excluded_subject"}
    if not required <= set(log.columns):
        raise ValueError("Handoff log is missing subject/version/inclusion columns")
    expected = {format_bids_subject(s) for s in config["dataset"]["subjects"]}
    excluded = {format_bids_subject(s) for s in config["dataset"].get("exclude", {})}
    if log["subject"].duplicated().any() or set(log["subject"]) != expected:
        raise ValueError("Handoff log must cover all configured subjects exactly once")
    if not log["pipeline_version"].eq(2).all():
        raise ValueError("Handoff log must use pipeline v2 for every subject")
    if not log["excluded_subject"].isin([True, False]).all():
        raise ValueError("Handoff log has an invalid subject inclusion flag")
    if log.loc[log["subject"].isin(excluded), "excluded_subject"].eq(False).any():
        raise ValueError("Handoff log includes an a-priori excluded subject")
    subjects = [format_bids_subject(s) for s in load_valid_subjects(root, config)]
    return log.set_index("subject").loc[subjects].copy()


def load_f3_inputs(root, config):
    """Audit all supplied CSVs and return F3 rows, source paths and audit evidence.

    The 16 main-task CSVs and v2 log/config are required. A missing original fold
    CSV is acceptable only when all cross metrics agree with current GroupKFold.
    Control-task CSVs are validated but are not plotted in F3.
    """
    root = Path(root)
    paths = [metrics_csv_path(root, config, m, s, w, F3_TASK) for m, s, w in _conditions()]
    sources = [*paths, root / "results" / "preproc_log.csv", root / "configs/config.yaml"]
    missing = [str(path.relative_to(root)) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError("F3 inputs missing; no outputs written:\n" + "\n".join(missing))
    cohort = _load_handoff(root, config)
    paths = sorted((root / config["paths"]["metrics"]).glob("*.csv"))
    inventory = []
    for path in paths:
        identity = pd.read_csv(path, usecols=["model", "family", "setting", "window", "task"])
        unique = identity.drop_duplicates()
        if len(unique) != 1:
            raise ValueError(f"{path.name}: filename requires exactly one row identity")
        row = unique.iloc[0]
        expected_path = metrics_csv_path(root, config, row.model, row.setting, row.window, row.task)
        if path != expected_path or row.family != F3_MODELS.get(row.model):
            raise ValueError(f"{path.name}: filename and row identity disagree")
        inventory.append({"path": str(path.relative_to(root)), "n_rows": len(identity), **row.to_dict()})
    all_raw = load_all_metrics(root, config, task_name=None)
    fold_path = folds_csv_path(root)
    supplied = pd.read_csv(fold_path, dtype={"subject": str}) if fold_path.is_file() else None
    folds, fold_evidence = verify_cross_folds(all_raw, config, cohort.index.tolist(), supplied)
    validated = []
    for task_name, task_raw in all_raw.groupby("task"):
        models = {model: F3_MODELS[model] for model in task_raw["model"].unique()}
        required_models = (
            set(F3_MODELS) if task_name == F3_TASK else {m for m, f in F3_MODELS.items() if f == "classical"}
        )
        if not required_models <= set(models):
            raise ValueError(f"{task_name}: missing required models {sorted(required_models - set(models))}")
        task_raw = validate_f3_metrics(
            task_raw, config, cohort.index.tolist(), folds, task_name=task_name, models=models
        )
        task = config["classification"]["tasks"][task_name]
        count_columns = [f"{task[label]}_after" for label in ("positive", "negative")]
        counts = cohort[count_columns].apply(pd.to_numeric, errors="coerce")
        for column in count_columns:
            _integers(counts[column], f"handoff {column}", minimum=1)
        expected_counts = counts.sum(axis=1)
        for key, group in task_raw.groupby(_SUBJECT_KEY):
            if group["n_test"].sum() != expected_counts[key[-1]]:
                raise ValueError(f"{key}: test-trial count differs from the v2 handoff log")
        validated.append(task_raw)
    all_raw = pd.concat(validated, ignore_index=True)
    raw = all_raw.loc[all_raw["task"].eq(F3_TASK)].copy()
    sources = [*paths, root / "results" / "preproc_log.csv", root / "configs/config.yaml"]
    if supplied is not None:
        sources.append(fold_path)
    audit = {
        "n_metric_files": len(paths),
        "n_all_metric_rows": len(all_raw),
        "files": inventory,
        "cross_fold_verification": fold_evidence,
    }
    return raw, sources, audit


def summarise_f3(subject_scores):
    """Summarise one score per subject: mean/sample SD, median, Q1/Q3 and IQR.

    Quartiles use linear interpolation. All scores remain proportions [0, 1];
    IQR is Q3 minus Q1, whereas q1/q3 are the interval endpoints used in LaTeX.
    """
    subjects = set(subject_scores["subject"])
    _validate_subject_grid(subject_scores, subjects)
    if len(subjects) < 2:
        raise ValueError("At least two subjects are required for sample SD")
    rows = []
    for model, setting, window in _conditions():
        group = subject_scores.loc[
            subject_scores["model"].eq(model)
            & subject_scores["setting"].eq(setting)
            & subject_scores["window"].eq(window)
        ]
        for metric in F3_METRICS:
            values = group[metric].to_numpy(dtype=float)
            q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75], method="linear")
            rows.append(
                {
                    "model": model,
                    "family": F3_MODELS[model],
                    "setting": setting,
                    "window": window,
                    "task": F3_TASK,
                    "metric": metric,
                    "n_subjects": len(values),
                    "mean": values.mean(),
                    "sd": values.std(ddof=1),
                    "median": median,
                    "q1": q1,
                    "q3": q3,
                    "iqr": q3 - q1,
                    "minimum": values.min(),
                    "maximum": values.max(),
                }
            )
    return pd.DataFrame(rows)


def generate_f3(root, config):
    """Validate all real inputs before exporting F3, subject/summary CSVs and hashes.

    Uses the existing R4 ``aggregate_subject_scores``: an unweighted fold mean
    within each subject, and the single test score for cross-subject evaluation.
    """
    from .plots import plot_f3

    root = Path(root)
    output = (root / "results" / "aggregation").resolve()
    figures = (root / config["paths"]["figures"]).resolve()
    metrics_dir = (root / config["paths"]["metrics"]).resolve()
    for directory in (output, figures):
        if directory == metrics_dir or metrics_dir in directory.parents:
            raise ValueError("F3 outputs must be outside the source metrics directory")
    raw, sources, audit = load_f3_inputs(root, config)
    subject_scores = aggregate_subject_scores(raw)
    summary = summarise_f3(subject_scores)
    manifest = {
        "task": F3_TASK,
        "seed": config["seed"],
        "statistical_unit": "subject",
        "within_reduction": "unweighted arithmetic mean of all configured folds per subject",
        "cross_reduction": "one test-subject score; all metric conditions and current GroupKFold verified",
        "input_audit": audit,
        "quartiles": "linear interpolation",
        "sd_ddof": 1,
        "n_raw_rows": len(raw),
        "n_subjects": subject_scores["subject"].nunique(),
        "sources": [
            {"path": str(path.relative_to(root)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in sources
        ],
    }
    png, pdf = plot_f3(subject_scores, config, figures / "F3_balanced_accuracy")
    output.mkdir(parents=True, exist_ok=True)
    subject_csv = output / "F3_subject_scores.csv"
    summary_csv = output / "F3_summary.csv"
    manifest_path = output / "F3_manifest.json"
    subject_scores.to_csv(subject_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return png, pdf, subject_csv, summary_csv, manifest_path
