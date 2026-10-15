import csv
import hashlib
import io
import json

import pytest

from seed.workflow_fixtures import generate, write_pack


def test_fixture_bytes_are_repeatable_and_manifest_matches():
    files = generate()
    assert files == generate()
    assert files != generate(42)
    manifest = json.loads(files["manifest.json"])
    assert manifest["is_synthetic"] and manifest["seed"] == 20261015
    for name, meta in manifest["files"].items():
        assert hashlib.sha256(files[name]).hexdigest() == meta["sha256"]
        rows = list(csv.DictReader(io.StringIO(files[name].decode())))
        assert len(rows) == meta["rows"]
        assert all(row["is_synthetic"] == "True" for row in rows)
    vessels = list(csv.DictReader(io.StringIO(files["vessel_observations.csv"].decode())))
    assert {r["scenario"] for r in vessels} == {"routine", "speed_outlier", "ais_gap", "stale_position", "position_outlier"}
    assert {r["expected_review"] for r in vessels} == {"True", "False"}


def test_pack_preserves_existing_outputs(tmp_path):
    write_pack(tmp_path)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(FileExistsError):
        write_pack(tmp_path, 42)
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}
