"""Receipt-causal vessel dossiers from validated synthetic watch snapshots."""

import copy
import math
from datetime import timedelta

from coastal_security.geography import digest, json_bytes
from coastal_security.globe import MAX_TRACK_GAP_MINUTES
from coastal_security.watch import END_MINUTE, STALE_MINUTES, START, VERSION, at, parse_time

DELAY_MINUTES = 5


def _distance_nm(a: dict, b: dict) -> float:
    lat1, lat2 = math.radians(a["latitude"]), math.radians(b["latitude"])
    dlat = lat2 - lat1
    dlon = math.radians(b["longitude"] - a["longitude"])
    hav = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 3440.065 * 2 * math.asin(math.sqrt(min(1.0, max(0.0, hav))))


def _continuity(rows: list[dict], start, end) -> list[dict]:
    """Reconstruct freshness using only information received at each boundary."""
    times = {start, end}
    for row in rows:
        for boundary in (
            parse_time(row["received_at"]),
            parse_time(row["observed_at"]) + timedelta(minutes=STALE_MINUTES),
        ):
            if start < boundary < end:
                times.add(boundary)
    intervals = []
    ordered = sorted(times)
    received_order = sorted(rows, key=lambda r: (parse_time(r["received_at"]), r["receipt_id"]))
    received_index, latest = 0, None
    for left, right in zip(ordered, ordered[1:], strict=False):
        while (
            received_index < len(received_order)
            and parse_time(received_order[received_index]["received_at"]) <= left
        ):
            row = received_order[received_index]
            if latest is None or parse_time(row["observed_at"]) > parse_time(latest["observed_at"]):
                latest = row
            received_index += 1
        # The strict > threshold changes immediately after its boundary. The
        # midpoint labels the positive-duration interval, not the boundary instant.
        probe = left + (right - left) / 2
        status = (
            "Awaiting first report"
            if latest is None
            else "Stale"
            if probe - parse_time(latest["observed_at"]) > timedelta(minutes=STALE_MINUTES)
            else "Fresh"
        )
        identity = latest["observation_id"] if latest else None
        if intervals and (intervals[-1]["status"], intervals[-1]["evidence_id"]) == (
            status,
            identity,
        ):
            intervals[-1]["end_utc"] = right.isoformat()
            intervals[-1]["minutes"] = (
                right - parse_time(intervals[-1]["start_utc"])
            ).total_seconds() / 60
        else:
            intervals.append(
                {
                    "start_utc": left.isoformat(),
                    "end_utc": right.isoformat(),
                    "minutes": (right - left).total_seconds() / 60,
                    "status": status,
                    "evidence_id": identity,
                }
            )
    return intervals


def vessel_history(
    state: dict, catalogue: list[dict], vessel_id: str, window_minutes: int | None = 60
) -> dict:
    """Derive a bounded dossier. Input is an internal, validated watch snapshot.

    The full accepted evidence is retained to explain carry-in freshness; window
    arrivals use receipt time and track samples use observation time. Neither
    admits a receipt after the selected watch clock. No uploaded data is accepted.
    """
    if (
        state.get("source") != "synthetic"
        or state.get("version") != VERSION
        or (
            window_minutes is not None
            and (type(window_minutes) is not int or window_minutes not in {30, 60})
        )
    ):
        raise ValueError("Expected a synthetic watch snapshot and a supported history window.")
    cursor = parse_time(state["as_of_utc"])
    if not START <= cursor <= at(END_MINUTE):
        raise ValueError("History clock is outside the exercise.")
    allowed = {r["vessel_id"] for r in state["positions"]} | set(state["unknown_vessels"])
    if vessel_id not in allowed:
        raise ValueError("Select a vessel in the current watch cohort.")
    vessel = next((r for r in catalogue if r["vessel_id"] == vessel_id), None)
    if vessel is None:
        raise ValueError("Vessel is missing from the synthetic catalogue.")
    start = (
        START if window_minutes is None else max(START, cursor - timedelta(minutes=window_minutes))
    )
    accepted = sorted(
        (
            copy.deepcopy(r)
            for r in state["observations"]
            if r["vessel_id"] == vessel_id
            and START <= parse_time(r["observed_at"]) <= parse_time(r["received_at"]) <= cursor
        ),
        key=lambda r: (parse_time(r["observed_at"]), r["observation_id"]),
    )
    if len(accepted) > 10_000:
        raise ValueError("History exceeds the exercise receipt bound.")
    ledger = [
        copy.deepcopy(r)
        for r in state["ledger"]
        if r.get("vessel_id") == vessel_id and start <= parse_time(r["received_at"]) <= cursor
    ]
    rows = [r for r in accepted if parse_time(r["observed_at"]) >= start]
    arrivals = [r for r in accepted if parse_time(r["received_at"]) >= start]
    segments, gaps, track = [], [], []
    segment = 0
    for index, row in enumerate(rows):
        if index:
            previous = rows[index - 1]
            minutes = (
                parse_time(row["observed_at"]) - parse_time(previous["observed_at"])
            ).total_seconds() / 60
            detail = {
                "from_id": previous["observation_id"],
                "to_id": row["observation_id"],
                "start_utc": previous["observed_at"],
                "end_utc": row["observed_at"],
                "minutes": minutes,
            }
            if 0 < minutes <= MAX_TRACK_GAP_MINUTES:
                segments.append(detail | {"distance_nm": _distance_nm(previous, row)})
            else:
                segment += 1
                gaps.append(detail)
        track.append(row | {"segment": segment})
    intervals = _continuity(accepted, start, cursor)
    durations = {
        status: sum(row["minutes"] for row in intervals if row["status"] == status)
        for status in ("Fresh", "Stale", "Awaiting first report")
    }
    latest = accepted[-1] if accepted else None
    age = (cursor - parse_time(latest["observed_at"])).total_seconds() / 60 if latest else None
    current = (
        "Awaiting first report" if age is None else "Stale" if age > STALE_MINUTES else "Fresh"
    )
    return {
        "schema_version": 1,
        "source": "synthetic",
        "exercise_version": VERSION,
        "vessel": copy.deepcopy(vessel),
        "as_of_utc": cursor.isoformat(),
        "window_start_utc": start.isoformat(),
        "window_duration_minutes": (cursor - start).total_seconds() / 60,
        "requested_window_minutes": window_minutes,
        "current_freshness": current,
        "last_known": latest,
        "last_observation_age_minutes": age,
        "accepted_evidence": accepted,
        "window_receipts": ledger,
        "window_observations": track,
        "continuity": intervals,
        "sampled_links": segments,
        "unlinked_intervals": gaps,
        "summary": {
            "accepted_arrivals": len(arrivals),
            "rejected_arrivals": sum(r["disposition"] == "Rejected" for r in ledger),
            "delayed_arrivals": sum(r["lag_minutes"] > DELAY_MINUTES for r in arrivals),
            "out_of_order_arrivals": sum(r["out_of_order"] for r in arrivals),
            "track_samples": len(rows),
            "sampled_link_distance_nm": sum(r["distance_nm"] for r in segments)
            if segments
            else None,
            "freshness_minutes": durations,
        },
        "definitions": {
            "freshness": (
                f"Last-known observation age > {STALE_MINUTES} minutes is stale; "
                "not sensor coverage."
            ),
            "timeline": (
                "Receipt-causal, half-open intervals. The strict stale threshold "
                "applies immediately after its boundary."
            ),
            "arrivals": (
                "Accepted/rejected receipts received in the inclusive selected window; "
                "delayed means lag > 5 minutes."
            ),
            "track": (
                "Accepted observation times in the inclusive window, received by the clock; "
                "gaps > 10 minutes stay open."
            ),
            "distance": (
                "Spherical sampled-link distance in nautical miles, "
                "not total travel or a navigational distance."
            ),
            "carry_in": (
                "Accepted evidence before the window explains last-known state "
                "and freshness at its start."
            ),
        },
    }


def history_export(history: dict, registry: dict) -> bytes:
    """Preserve the existing watch/replay/casebook export formats unchanged."""
    payload = copy.deepcopy(history) | {"registry": copy.deepcopy(registry)}
    payload["history_sha256"] = digest(json_bytes(payload))
    payload["hash_scope"] = (
        "Canonical JSON excluding history_sha256 and hash_scope; not a signature."
    )
    return json_bytes(payload)
