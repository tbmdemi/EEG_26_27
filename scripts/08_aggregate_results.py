"""Gom metric target-standard và tạo hình F3 cho report."""

import hashlib
import json
from pathlib import Path

from eegproj.config import find_root, load_config
from eegproj.evaluation.aggregate import (
    F3_TASK,
    aggregate_f3_subject_scores,
    plot_f3_balanced_accuracy,
    summarise_f3_scores,
)
from eegproj.evaluation.splits import format_bids_subject, load_valid_subjects
from eegproj.evaluation.stats import load_all_metrics


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    root = find_root(Path(__file__).resolve().parent)
    config = load_config(root)
    metrics_dir = root / config["paths"]["metrics"]
    metric_paths = sorted(metrics_dir.glob(f"*_{F3_TASK}.csv"))
    if not metric_paths:
        raise FileNotFoundError(f"No {F3_TASK} metric CSVs found in {metrics_dir}.")

    metrics = load_all_metrics(root, config, task_name=F3_TASK)
    expected_subjects = [format_bids_subject(subject) for subject in load_valid_subjects(root, config)]
    configured_models = {
        "classical": config["models"]["classical"],
        "dl": config["models"]["dl"],
    }
    # Chỉ các entry dạng mapping là model; các giá trị scalar là tham số chung.
    expected_models = {
        model: family
        for family, settings_by_name in configured_models.items()
        for model, settings in settings_by_name.items()
        if isinstance(settings, dict)
    }
    subject_scores = aggregate_f3_subject_scores(
        metrics,
        expected_subjects=expected_subjects,
        expected_models=expected_models,
        within_splits=int(config["evaluation"]["within"]["n_splits"]),
    )
    summary = summarise_f3_scores(subject_scores)

    output_dir = root / "results" / "aggregation"
    output_dir.mkdir(parents=True, exist_ok=True)
    subject_scores_path = output_dir / "F3_subject_scores.csv"
    summary_path = output_dir / "F3_summary.csv"
    manifest_path = output_dir / "F3_manifest.json"
    subject_scores.to_csv(subject_scores_path, index=False)
    summary.to_csv(summary_path, index=False)

    figure_stem = root / config["paths"]["figures"] / "F3_balanced_accuracy"
    png_path, pdf_path = plot_f3_balanced_accuracy(
        subject_scores,
        expected_models=expected_models,
        window_seconds=config["classification"]["windows"],
        output_stem=figure_stem,
        dpi=int(config["erp"]["figure_dpi"]),
    )
    manifest = {
        "task": F3_TASK,
        "statistical_unit": "subject",
        "within_reduction": "arithmetic mean across configured folds per subject",
        "cross_reduction": "one held-out score per subject",
        "n_subjects": len(expected_subjects),
        "n_metric_rows": len(metrics),
        "metric_files": [{"path": path.relative_to(root).as_posix(), "sha256": _sha256(path)} for path in metric_paths],
        "outputs": [
            path.relative_to(root).as_posix() for path in (subject_scores_path, summary_path, png_path, pdf_path)
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Aggregated {len(metrics)} metric rows for {len(expected_subjects)} subjects.", flush=True)
    print(
        f"Saved {subject_scores_path}, {summary_path}, {manifest_path}, {png_path}, and {pdf_path}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
