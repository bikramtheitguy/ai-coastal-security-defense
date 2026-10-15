import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from dashboard.results import (
    AssetError,
    binary_metrics,
    checked_asset,
    read_image,
    read_predictions,
)

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "dashboard"


@pytest.mark.parametrize(
    ("filename", "expected", "precision", "recall", "f1", "accuracy", "baseline"),
    [
        ("drone_threat_predictions_phase1.csv", (2, 8, 14, 176), 0.2, 0.125, 2 / 13, 0.89, 0.92),
        ("cyber_intrusion_predictions_phase1.csv", (0, 1, 4, 195), 0, 0, 0, 0.975, 0.98),
    ],
)
def test_preserved_predictions(filename, expected, precision, recall, f1, accuracy, baseline):
    path, _ = checked_asset(filename)
    metrics = binary_metrics(read_predictions(path))
    assert (metrics.tp, metrics.fp, metrics.fn, metrics.tn) == expected
    assert metrics.count == 200
    assert metrics.precision == pytest.approx(precision)
    assert metrics.recall == pytest.approx(recall)
    assert metrics.f1 == pytest.approx(f1)
    assert metrics.accuracy == pytest.approx(accuracy)
    assert metrics.negative_baseline_accuracy == pytest.approx(baseline)


@pytest.mark.parametrize(
    "content",
    [
        "",
        "actual,predicted\n",
        "actual,wrong\n0,1\n",
        "actual,predicted,extra\n0,1,2\n",
        "actual,predicted\n1,\n",
        "actual,predicted\n1,-1\n",
        "actual,predicted\n1,0.5\n",
        "actual,predicted\n0,maybe\n",
    ],
)
def test_reject_invalid_prediction_tables(tmp_path, content):
    path = tmp_path / "bad.csv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(AssetError):
        read_predictions(path)


def test_missing_prediction_file(tmp_path):
    with pytest.raises(AssetError, match="Cannot read predictions"):
        read_predictions(tmp_path / "missing.csv")


def test_zero_denominators_are_explicit():
    metrics = binary_metrics(pd.DataFrame({"actual": [0, 0], "predicted": [0, 0]}))
    assert metrics.precision is None
    assert metrics.recall is None
    assert metrics.f1 is None
    assert metrics.accuracy == 1


def test_all_saved_assets_match_manifest_and_images_decode():
    manifest = json.loads((ASSETS / "artifacts.json").read_text(encoding="utf-8"))
    for name, entry in manifest["assets"].items():
        path, digest = checked_asset(name)
        assert digest == entry["sha256"]
        if path.suffix == ".png":
            image = read_image(path)
            assert image.width > 0 and image.height > 0


def test_changed_asset_is_rejected(tmp_path):
    (tmp_path / "artifacts.json").write_text(
        json.dumps({"assets": {"test.csv": {"sha256": hashlib.sha256(b"old").hexdigest()}}}),
        encoding="utf-8",
    )
    (tmp_path / "test.csv").write_text("changed", encoding="utf-8")
    with pytest.raises(AssetError, match="differs from the saved baseline"):
        checked_asset("test.csv", tmp_path)


@pytest.mark.parametrize("manifest", [None, "not json", "{}", "[]"])
def test_missing_or_invalid_manifest(tmp_path, manifest):
    if manifest is not None:
        (tmp_path / "artifacts.json").write_text(manifest, encoding="utf-8")
    with pytest.raises(AssetError, match="Restore"):
        checked_asset("test.csv", tmp_path)


def test_asset_path_must_be_a_filename():
    with pytest.raises(AssetError, match="filename"):
        checked_asset("../data/vessel_data.csv")


def test_corrupt_image_is_rejected(tmp_path):
    path = tmp_path / "bad.png"
    path.write_bytes(b"not an image")
    with pytest.raises(AssetError, match="Cannot decode chart"):
        read_image(path)


@pytest.mark.parametrize(
    ("name", "id_column", "label", "rows", "positives"),
    [
        ("vessel_data.csv", "vessel_id", "is_anomalous", 500, 26),
        ("drone_data.csv", "frame_id", "is_threat", 1000, 81),
        ("cyber_data.csv", "log_id", "intrusion_attempt", 1000, 20),
    ],
)
def test_original_dataset_contract(name, id_column, label, rows, positives):
    manifest = json.loads((ASSETS / "artifacts.json").read_text(encoding="utf-8"))
    path = ROOT / "data" / name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["datasets"][name]["sha256"]
    frame = pd.read_csv(path)
    assert len(frame) == rows
    assert not frame.isna().any().any()
    assert frame[id_column].is_unique
    assert frame[label].isin([0, 1]).all()
    assert int(frame[label].sum()) == positives
