import hashlib
import json
from dataclasses import replace

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from coastal_security.vessel.data import (
    generate_snapshots,
    split_snapshots,
    validate_snapshots,
)
from coastal_security.vessel.model import (
    FEATURES,
    TrainingConfig,
    choose_threshold,
    evaluate,
    raw_to_binary,
    summarize,
    train_select,
)
from coastal_security.vessel.run import run_experiment

FAST = TrainingConfig(n_estimators=(16,), max_samples=(32,))


@pytest.fixture(scope="module")
def splits():
    return split_snapshots(generate_snapshots(1000, 42))


@pytest.fixture(scope="module")
def selected(splits):
    return train_select(splits["train"], splits["validation"], FAST)


def test_generator_is_repeatable_and_seed_changes_samples():
    first = generate_snapshots(1000, 42)
    pd.testing.assert_frame_equal(first, generate_snapshots(1000, 42))
    assert not first.equals(generate_snapshots(1000, 43))
    validate_snapshots(first)
    assert first.groupby("scenario")["is_anomalous"].nunique().eq(1).all()
    assert first["scenario"].nunique() == 4


def test_split_membership_is_order_independent_chronological_and_disjoint(splits):
    shuffled = split_snapshots(generate_snapshots(1000).sample(frac=1, random_state=7))
    assert [len(s) for s in splits.values()] == [600, 200, 200]
    ids = []
    for name, split in splits.items():
        pd.testing.assert_frame_equal(split, shuffled[name])
        ids.extend(split["record_id"])
    assert len(set(ids)) == 1000
    assert splits["train"]["observed_at"].max() < splits["validation"]["observed_at"].min()
    assert splits["validation"]["observed_at"].max() < splits["test"]["observed_at"].min()


@pytest.mark.parametrize(
    "column,value",
    [
        ("speed_knots", np.nan),
        ("speed_knots", np.inf),
        ("speed_knots", -1),
        ("speed_knots", 61),
        ("is_anomalous", -1),
        ("observed_at", "2025-01-01T00:00:00"),
        ("vessel_id", ""),
    ],
)
def test_invalid_snapshot_values_rejected(column, value):
    frame = generate_snapshots(1000)
    frame.loc[0, column] = value
    with pytest.raises(ValueError):
        validate_snapshots(frame)


def test_repeated_vessels_and_empty_tables_rejected():
    frame = generate_snapshots(1000)
    frame.loc[1, "vessel_id"] = frame.loc[0, "vessel_id"]
    with pytest.raises(ValueError, match="repeated-vessel"):
        split_snapshots(frame)
    with pytest.raises(ValueError, match="nonempty"):
        validate_snapshots(frame.iloc[:0])


def test_missing_positive_evaluation_class_is_explicit():
    frame = generate_snapshots(1000)
    frame.loc[800:, "is_anomalous"] = 0
    with pytest.raises(ValueError, match="test needs both classes"):
        split_snapshots(frame)


def test_fitting_receives_only_training_speed_and_no_labels(splits, monkeypatch):
    original = IsolationForest.fit
    seen = []

    def inspect_fit(self, X, y=None, **kwargs):
        seen.append(np.asarray(X).copy())
        assert y is None
        return original(self, X, y, **kwargs)

    monkeypatch.setattr(IsolationForest, "fit", inspect_fit)
    train_select(splits["train"], splits["validation"], FAST)
    assert len(seen) == 1
    np.testing.assert_array_equal(seen[0], splits["train"][FEATURES].to_numpy())


def test_training_labels_and_metadata_cannot_be_features(splits, selected):
    altered = splits["train"].copy()
    altered["is_anomalous"] = 1 - altered["is_anomalous"]
    altered["scenario"] = "label-like information that must not reach the model"
    other = train_select(altered, splits["validation"], FAST)
    assert other.threshold == selected.threshold
    np.testing.assert_array_equal(
        selected.pipeline.score_samples(splits["test"][FEATURES]),
        other.pipeline.score_samples(splits["test"][FEATURES]),
    )


def test_test_labels_cannot_change_predictions_or_selection(splits, selected, monkeypatch):
    before_threshold, before_params = selected.threshold, selected.parameters.copy()
    report, predictions = evaluate(selected, splits["test"])

    def forbidden_fit(*args, **kwargs):
        pytest.fail("Evaluation must never fit a model")

    monkeypatch.setattr(selected.pipeline, "fit", forbidden_fit)
    changed = splits["test"].copy()
    changed["is_anomalous"] = 1 - changed["is_anomalous"]
    changed_report, changed_predictions = evaluate(selected, changed)
    pd.testing.assert_series_equal(
        predictions["anomaly_score"], changed_predictions["anomaly_score"]
    )
    pd.testing.assert_series_equal(
        predictions["predicted_anomaly"], changed_predictions["predicted_anomaly"]
    )
    assert (
        report["selected_threshold"]["positives"]
        != changed_report["selected_threshold"]["positives"]
    )
    assert (selected.threshold, selected.parameters) == (before_threshold, before_params)


def test_overlap_and_wrong_time_are_rejected(splits, selected):
    with pytest.raises(ValueError, match="overlapping"):
        train_select(splits["train"], splits["train"], FAST)
    with pytest.raises(ValueError, match="overlap"):
        evaluate(selected, splits["validation"])
    changed = splits["test"].copy()
    changed["observed_at"] = splits["validation"]["observed_at"].to_numpy()
    with pytest.raises(ValueError, match="follow"):
        evaluate(selected, changed)


def test_isolation_forest_label_mapping():
    np.testing.assert_array_equal(raw_to_binary(np.array([-1, 1, -1])), [1, 0, 1])
    with pytest.raises(ValueError):
        raw_to_binary(np.array([0, 1]))


def test_threshold_selection_respects_budget_and_has_no_alert_fallback():
    actual = np.array([0, 0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.9, 0.8, 0.95])
    threshold, metrics = choose_threshold(actual, scores, 0)
    assert threshold == 0.95 and metrics["fp"] == 0 and metrics["tp"] == 1
    threshold, metrics = choose_threshold(actual, scores, 0.25)
    assert threshold == 0.8 and metrics["fp"] == 1 and metrics["tp"] == 2
    threshold, metrics = choose_threshold(np.array([0, 1]), np.array([0.9, 0.1]), 0)
    assert threshold > 0.9 and metrics["precision"] is None and metrics["recall"] == 0


def test_metrics_match_independent_sklearn_calculations(splits, selected):
    report, predictions = evaluate(selected, splits["test"])
    actual, predicted = predictions["is_anomalous"], predictions["predicted_anomaly"]
    metrics = report["selected_threshold"]
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
    assert (metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]) == (tn, fp, fn, tp)
    assert metrics["precision"] == pytest.approx(precision_score(actual, predicted))
    assert metrics["recall"] == pytest.approx(recall_score(actual, predicted))
    assert metrics["f1"] == pytest.approx(f1_score(actual, predicted))
    assert metrics["false_positive_rate"] == fp / (fp + tn)
    assert metrics["false_alerts_per_1000_observations"] == 1000 * fp / len(actual)
    assert sum(s["rows"] for s in report["scenario_slices"].values()) == len(actual)


def test_undefined_metrics_and_bad_inputs():
    result = summarize(np.array([0, 0]), np.array([0, 0]), np.array([0.2, 0.3]))
    assert result["precision"] is None and result["recall"] is None and result["f1"] is None
    assert result["roc_auc"] is None and result["average_precision"] is None
    with pytest.raises(ValueError):
        summarize(np.array([]), np.array([]))
    with pytest.raises(ValueError):
        choose_threshold(np.array([0, 1]), np.array([0.1, np.inf]), 0.05)
    with pytest.raises(ValueError):
        replace(FAST, max_false_positive_rate=float("nan")).validate()


def test_repeatable_bundle_hashes_lineage_and_new_pipeline_roundtrip(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    report = run_experiment(first, 1000, FAST)
    assert report == run_experiment(second, 1000, FAST)
    manifest = json.loads((first / "manifest.json").read_text())
    assert manifest["complete"] and manifest["provenance"]["operational_data"] is False
    assert manifest["source"]["source_sha256"]
    for name, entry in manifest["files"].items():
        assert hashlib.sha256((first / name).read_bytes()).hexdigest() == entry["sha256"]
        assert (first / name).read_bytes() == (second / name).read_bytes()
    prediction = pd.read_csv(first / "predictions.csv")
    membership = pd.read_csv(first / "splits.csv")
    assert set(prediction["record_id"]) == set(
        membership.loc[membership["split"] == "test", "record_id"]
    )
    # Only deserialize the artifact this test just trained in its own temporary directory.
    bundle = joblib.load(first / "model.joblib")
    scores = -bundle["pipeline"].score_samples(prediction[bundle["features"]])
    np.testing.assert_allclose(scores, prediction["anomaly_score"], rtol=1e-12)
    np.testing.assert_array_equal(scores >= bundle["threshold"], prediction["predicted_anomaly"])
    with pytest.raises(FileExistsError):
        run_experiment(first, 1000, FAST)
    assert (
        hashlib.sha256((first / "report.json").read_bytes()).hexdigest()
        == manifest["files"]["report.json"]["sha256"]
    )
