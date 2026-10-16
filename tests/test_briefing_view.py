import json

from coastal_security.briefing import watch_briefing
from coastal_security.watch import generate_exercise
from dashboard.briefing_view import PAGE_SIZE, briefing_chart


def test_chart_uses_causal_intervals_numeric_time_and_explicit_legend_contract():
    result = watch_briefing(generate_exercise(), 95, None, state_iso="IN-OR")
    spec = briefing_chart(result, result["vessels"], "IST")
    json.dumps(spec, allow_nan=False)
    assert spec["encoding"]["x"]["scale"]["domain"] == [0, 95]
    assert "IST" in spec["encoding"]["x"]["title"]
    assert spec["encoding"]["color"]["legend"] is None
    assert spec["encoding"]["y"]["axis"]["labelOverlap"] is False
    assert spec["encoding"]["y"]["scale"]["paddingInner"] == 0.3
    outage = [
        r for r in spec["data"]["values"] if r["vessel_id"] == "SIM-004" and r["status"] == "Stale"
    ]
    assert len(outage) == 1
    assert outage[0]["start_minute"] == 40 and outage[0]["end_minute"] == 91
    assert outage[0]["current_freshness"] == "Fresh"
    assert outage[0]["evidence_id"] == "OBS-SIM-004-025"


def test_chart_page_does_not_add_hidden_vessels_or_mutate_evidence():
    result = watch_briefing(generate_exercise(), 60)
    page = result["vessels"][:PAGE_SIZE]
    spec = briefing_chart(result, page, "UTC")
    assert {r["vessel_id"] for r in spec["data"]["values"]} == {r["vessel_id"] for r in page}
    assert len(result["vessels"]) == 44
    assert "start_minute" not in result["vessels"][0]["continuity"][0]
