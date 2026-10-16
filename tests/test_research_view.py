import pytest

from dashboard.research_view import OUTCOMES, review_rows, review_spec
from dashboard.results import checked_asset, read_predictions


@pytest.mark.parametrize(
    "kind,filename,counts",
    [
        ("drone", "drone_threat_predictions_phase1.csv", [2, 8, 14, 176]),
        ("cyber", "cyber_intrusion_predictions_phase1.csv", [0, 1, 4, 195]),
    ],
)
def test_review_outcomes_reconcile_to_preserved_predictions(kind, filename, counts):
    source = read_predictions(checked_asset(filename)[0])
    rows = review_rows(source)
    assert [int((rows.outcome == name).sum()) for name in OUTCOMES] == counts
    assert rows.saved_row.tolist() == list(range(1, 201))
    assert rows[["actual", "predicted"]].equals(source)
    spec = review_spec(rows, kind, "All outcomes")
    assert len(spec["data"]["values"]) == (200 if kind == "drone" else 4)
    if kind == "cyber":
        assert sum(cell["count"] for cell in spec["data"]["values"]) == 200
    # The review UI adds no invented geolocation, packet identity or model score.
    assert not {"latitude", "longitude", "confidence", "source_ip"} & set(rows.columns)
