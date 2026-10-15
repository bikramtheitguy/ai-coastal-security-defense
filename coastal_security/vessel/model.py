"""Train-only fitting, validation-only selection, and held-out evaluation."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

from coastal_security.vessel.data import assert_separated, validate_snapshots

FEATURES = ["speed_knots"]


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    max_false_positive_rate: float = 0.05
    n_estimators: tuple[int, ...] = (100, 200)
    max_samples: tuple[int, ...] = (128, 256)

    def validate(self) -> None:
        if not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("seed must be an unsigned 32-bit integer")
        if not np.isfinite(self.max_false_positive_rate) or not (
            0 <= self.max_false_positive_rate <= 1
        ):
            raise ValueError("max_false_positive_rate must be between 0 and 1")
        if not self.n_estimators or not self.max_samples:
            raise ValueError("The model search grid must be nonempty")
        if any(n <= 0 for n in (*self.n_estimators, *self.max_samples)):
            raise ValueError("Model search values must be positive integers")


def raw_to_binary(raw: np.ndarray) -> np.ndarray:
    """Map scikit-learn -1 outliers to class 1, and +1 inliers to class 0."""
    values = np.asarray(raw)
    if values.ndim != 1 or not np.isin(values, [-1, 1]).all():
        raise ValueError("Isolation Forest predictions must be a vector of -1 and +1")
    return (values == -1).astype(int)


def summarize(actual: np.ndarray, predicted: np.ndarray, scores=None) -> dict:
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if actual.ndim != 1 or not len(actual) or actual.shape != predicted.shape:
        raise ValueError("Labels must be nonempty vectors of the same shape")
    if not np.isin(actual, [0, 1]).all() or not np.isin(predicted, [0, 1]).all():
        raise ValueError("Metrics require binary labels 0 and 1")
    tp = int(((actual == 1) & (predicted == 1)).sum())
    fp = int(((actual == 0) & (predicted == 1)).sum())
    fn = int(((actual == 1) & (predicted == 0)).sum())
    tn = int(((actual == 0) & (predicted == 0)).sum())
    result = {
        "rows": len(actual),
        "positives": tp + fn,
        "negatives": tn + fp,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "accuracy": (tp + tn) / len(actual),
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "miss_rate": fn / (tp + fn) if tp + fn else None,
        "alerts_per_1000_observations": 1000 * (tp + fp) / len(actual),
        "false_alerts_per_1000_observations": 1000 * fp / len(actual),
    }
    if scores is not None:
        scores = np.asarray(scores)
        if scores.shape != actual.shape or not np.isfinite(scores).all():
            raise ValueError("Scores must be finite and aligned with labels")
        both_classes = len(np.unique(actual)) == 2
        result["average_precision"] = (
            float(average_precision_score(actual, scores)) if both_classes else None
        )
        result["roc_auc"] = float(roc_auc_score(actual, scores)) if both_classes else None
        result["average_precision_baseline"] = float(actual.mean())
    return result


def choose_threshold(actual: np.ndarray, scores: np.ndarray, max_fpr: float) -> tuple[float, dict]:
    """Maximize validation F1 under a declared empirical false-positive budget."""
    summarize(actual, np.zeros(len(actual), dtype=int), scores)  # validate inputs
    if set(actual) != {0, 1}:
        raise ValueError("Threshold selection requires both validation classes")
    if not np.isfinite(max_fpr) or not 0 <= max_fpr <= 1:
        raise ValueError("The false-positive budget must be between 0 and 1")
    thresholds = np.append(np.unique(scores), np.nextafter(max(scores), np.inf))
    best = None
    for threshold in thresholds:
        metrics = summarize(actual, (scores >= threshold).astype(int))
        if metrics["false_positive_rate"] > max_fpr:
            continue
        rank = (metrics["f1"], -metrics["fp"], float(threshold))
        if best is None or rank > best[0]:
            best = (rank, float(threshold), metrics)
    return best[1], best[2]  # the no-alert threshold always satisfies the budget


@dataclass
class SelectedModel:
    pipeline: Pipeline
    threshold: float
    parameters: dict
    validation_metrics: dict
    search: list[dict]
    used_record_ids: frozenset[str]
    used_vessel_ids: frozenset[str]
    validation_end: str


def train_select(
    train: pd.DataFrame, validation: pd.DataFrame, config: TrainingConfig
) -> SelectedModel:
    """No test argument: fit only training features and select only on validation."""
    config.validate()
    validate_snapshots(train)
    validate_snapshots(validation)
    assert_separated(train, validation)
    if max(config.max_samples) > len(train):
        raise ValueError("Training rows must cover the requested max_samples grid")
    best, search = None, []
    for trees in sorted(set(config.n_estimators)):
        for samples in sorted(set(config.max_samples)):
            parameters = {
                "n_estimators": trees,
                "max_samples": samples,
                "contamination": "auto",
                "random_state": config.seed,
                "n_jobs": 1,
            }
            pipeline = Pipeline(
                [
                    (
                        "features",
                        ColumnTransformer([("speed", "passthrough", FEATURES)], remainder="drop"),
                    ),
                    ("detector", IsolationForest(**parameters)),
                ]
            )
            pipeline.fit(train[FEATURES])  # labels, IDs, scenarios and future data are excluded
            scores = -pipeline.score_samples(validation[FEATURES])
            threshold, metrics = choose_threshold(
                validation["is_anomalous"].to_numpy(), scores, config.max_false_positive_rate
            )
            search.append({**parameters, "threshold": threshold, **metrics})
            rank = (metrics["f1"], -metrics["fp"], -trees, -samples)
            if best is None or rank > best[0]:
                best = (rank, pipeline, threshold, parameters, metrics)
    return SelectedModel(
        pipeline=best[1],
        threshold=best[2],
        parameters=best[3],
        validation_metrics=best[4],
        search=search,
        used_record_ids=frozenset(train["record_id"]) | frozenset(validation["record_id"]),
        used_vessel_ids=frozenset(train["vessel_id"]) | frozenset(validation["vessel_id"]),
        validation_end=validation["observed_at"].max(),
    )


def evaluate(selected: SelectedModel, test: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Evaluate the frozen selection; never refit or adjust a threshold here."""
    validate_snapshots(test)
    if selected.used_record_ids & set(test["record_id"]) or selected.used_vessel_ids & set(
        test["vessel_id"]
    ):
        raise ValueError("Test observations or vessels overlap training/validation")
    if test["observed_at"].min() <= selected.validation_end:
        raise ValueError("Test observations must follow the validation period")
    actual = test["is_anomalous"].to_numpy()
    scores = -selected.pipeline.score_samples(test[FEATURES])
    predicted = (scores >= selected.threshold).astype(int)
    default_labels = raw_to_binary(selected.pipeline.predict(test[FEATURES]))
    report = {
        "selected_threshold": summarize(actual, predicted, scores),
        "default_isolation_forest_threshold": summarize(actual, default_labels, scores),
        "always_normal": summarize(actual, np.zeros(len(test), dtype=int)),
        "scenario_slices": {
            name: summarize(actual[mask], predicted[mask])
            for name in sorted(test["scenario"].unique())
            if (mask := (test["scenario"] == name).to_numpy()).any()
        },
    }
    predictions = test.copy()
    predictions["anomaly_score"] = scores
    predictions["predicted_anomaly"] = predicted
    predictions["default_predicted_anomaly"] = default_labels
    return report, predictions
