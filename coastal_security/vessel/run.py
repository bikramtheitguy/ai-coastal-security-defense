"""Write a new, inspectable experiment bundle without touching legacy artifacts."""

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import joblib
import pandas as pd

from coastal_security.vessel.data import GENERATOR_VERSION, generate_snapshots, split_snapshots
from coastal_security.vessel.model import FEATURES, TrainingConfig, evaluate, train_select

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def source_metadata() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    # Normalize source newlines so Windows and Linux identify the same program.
    sources = {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(
            path.read_text(encoding="utf-8").encode("utf-8")
        ).hexdigest()
        for path in sorted((ROOT / "coastal_security").rglob("*.py"))
    }
    lock = ROOT / "uv.lock"
    return {
        "git_commit_context": commit,
        "source_sha256": sources,
        "uv_lock_sha256": hashlib.sha256(lock.read_text(encoding="utf-8").encode()).hexdigest(),
        "note": "Source hashes identify executed files, including any uncommitted changes.",
    }


def model_card(report: dict) -> str:
    metrics = report["test"]["selected_threshold"]
    return f"""# Synthetic vessel-speed benchmark

**Research only. An anomaly is a synthetic label, not evidence of a security threat.**

Generator: {GENERATOR_VERSION}; seed: {report["config"]["seed"]}.
One fictional observation per fictional vessel. No AIS or operational records are used.
The only model feature is speed in knots. UTC timestamps define the 60/20/20 split.

Training fits the feature selector and Isolation Forest on training features only.
Validation chooses the forest and threshold by F1 subject to an empirical false-positive
budget of {report["config"]["max_false_positive_rate"]:.1%}. Test data is evaluated after
selection, without refitting. This budget is an experiment setting, not an operational SLA.

## Held-out synthetic results

- Observations: {metrics["rows"]}; positive cases: {metrics["positives"]}.
- Detected: {metrics["tp"]}; false alerts: {metrics["fp"]}; missed: {metrics["fn"]}.
- Precision: {metrics["precision"]}; recall: {metrics["recall"]}; F1: {metrics["f1"]}.
- False-positive rate: {metrics["false_positive_rate"]}.
- False alerts per 1,000 observations: {metrics["false_alerts_per_1000_observations"]}.
- Average precision: {metrics["average_precision"]}; ROC-AUC: {metrics["roc_auc"]}.

Undefined rates are null in report.json. Scores are anomaly rankings, not probabilities.
The report includes the always-normal and default Isolation Forest baselines, plus
scenario slices; denominators differ between false-positive rate and false-alert frequency.

## Limitations

Slow ordinary and slow anomalous scenarios deliberately overlap. Speed alone cannot
separate them. The generator's distributions and 8% anomaly probability are invented,
not estimates of maritime prevalence. All splits come from the same stationary generator;
chronological separation does not establish robustness to real temporal or geographic drift.
No traffic exposure, event deduplication or trajectories exist, so alerts per hour/day and
event-level detection performance cannot be estimated. No real-world confidence claim is made.

This publicly reproducible test set is a software benchmark. After inspecting it, do not
reuse it as an independent final holdout when changing the model or generator. Future
evaluation needs separately approved, representative, vessel/scene/time-isolated data.

## Reproduction and artifact safety

manifest.json records source, lock, data and output hashes. splits.csv preserves membership;
predictions.csv preserves IDs, labels and scores. model.joblib contains the newly fitted
feature pipeline and threshold. It is pickle-based: load only your own trusted local output
in the recorded environment. A checksum is not authentication. No repository command loads
downloaded or historical models. The Streamlit demo continues to show its original assets.
"""


def run_experiment(output: Path, rows: int = 3000, config: TrainingConfig | None = None) -> dict:
    config = config or TrainingConfig()
    config.validate()
    frame = generate_snapshots(rows=rows, seed=config.seed)
    splits = split_snapshots(frame)
    output.mkdir(parents=True, exist_ok=False)  # even an empty existing directory is protected
    selected = train_select(splits["train"], splits["validation"], config)
    evaluation, predictions = evaluate(selected, splits["test"])
    report = {
        "schema_version": 1,
        "generator_version": GENERATOR_VERSION,
        "config": asdict(config),
        "feature_contract": {
            "columns": FEATURES,
            "units": {"speed_knots": "knots"},
            "missing_values": "reject",
            "preprocessing": "numeric passthrough",
        },
        "splits": {
            name: {
                "rows": len(split),
                "positives": int(split["is_anomalous"].sum()),
                "start_utc": split["observed_at"].min(),
                "end_utc": split["observed_at"].max(),
            }
            for name, split in splits.items()
        },
        "selected_parameters": selected.parameters,
        "selected_threshold": selected.threshold,
        "score_definition": "negative IsolationForest.score_samples; larger is more anomalous",
        "selection_rule": (
            "validation F1 with FPR budget; ties prefer fewer false alerts, then smaller forest"
        ),
        "validation": selected.validation_metrics,
        "test": evaluation,
        "test_fpr_exceeds_validation_budget": (
            evaluation["selected_threshold"]["false_positive_rate"] > config.max_false_positive_rate
        ),
        "limitations": [
            "Invented synthetic distributions, not maritime prevalence or detection evidence",
            "One snapshot per vessel; no trajectories, geographic context or event-level alerts",
            "Stationary generator; time separation is not a temporal-drift evaluation",
            "No confidence interval or real-world acceptance threshold is established",
            "Public benchmark cannot remain an independent holdout after iterative development",
        ],
    }
    frame.to_csv(output / "dataset.csv", index=False, lineterminator="\n")
    pd.concat(
        [
            split[["record_id", "vessel_id", "observed_at"]].assign(split=name)
            for name, split in splits.items()
        ],
        ignore_index=True,
    ).to_csv(output / "splits.csv", index=False, lineterminator="\n")
    predictions.to_csv(output / "predictions.csv", index=False, lineterminator="\n")
    pd.DataFrame(selected.search).to_csv(
        output / "validation_search.csv", index=False, lineterminator="\n"
    )
    joblib.dump(
        {
            "schema_version": 1,
            "pipeline": selected.pipeline,
            "threshold": selected.threshold,
            "features": FEATURES,
        },
        output / "model.joblib",
    )
    write_json(output / "report.json", report)
    (output / "MODEL_CARD.md").write_text(model_card(report), encoding="utf-8", newline="\n")
    manifest = {
        "schema_version": 1,
        "complete": True,
        "source": source_metadata(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            **{name: version(name) for name in ("numpy", "pandas", "scikit-learn", "joblib")},
        },
        "provenance": {
            "source": "generated locally; no external inputs",
            "license": "MIT",
            "generator_version": GENERATOR_VERSION,
            "seed": config.seed,
            "anomaly_probability": 0.08,
            "operational_data": False,
        },
        "files": {path.name: {"sha256": sha256(path)} for path in sorted(output.iterdir())},
    }
    write_json(output / "manifest.json", manifest)  # written last: absent on interrupted runs
    return report
