"""Offline globe context and receipt-aware display payloads, independent of the UI."""

import json
import math
from pathlib import Path

from coastal_security.geography import digest
from coastal_security.watch import parse_time

DATA = Path(__file__).resolve().parents[1] / "data/globe"
VESSEL_LAYER = "watch-vessels"
MAX_TRACK_GAP_MINUTES = 10


def load_land(directory: Path = DATA) -> tuple[dict, dict]:
    """Read only the fixed, bounded, checksum-registered public land asset."""
    try:
        manifest = json.loads((directory / "manifest.json").read_bytes())
        path = directory / "land.geojson"
        if path.stat().st_size > 500_000:
            raise ValueError("Globe geometry exceeds its rendering bound.")
        payload = path.read_bytes()
        if digest(payload) != manifest["sha256"] or len(payload) != manifest["bytes"]:
            raise ValueError("Globe geometry checksum differs from its manifest.")
        land = json.loads(payload)
        if land["type"] != "FeatureCollection" or not 1 <= len(land["features"]) <= 1000:
            raise ValueError("Invalid globe feature collection.")
        for feature in land["features"]:
            geometry = feature["geometry"]
            if geometry["type"] not in {"Polygon", "MultiPolygon"}:
                raise ValueError("Expected land polygons.")
            polygons = (
                [geometry["coordinates"]]
                if geometry["type"] == "Polygon"
                else geometry["coordinates"]
            )
            for polygon in polygons:
                for ring in polygon:
                    if len(ring) < 4 or ring[0] != ring[-1]:
                        raise ValueError("Land ring is not closed.")
                    for point in ring:
                        if len(point) != 2 or any(
                            isinstance(v, bool)
                            or not isinstance(v, int | float)
                            or not math.isfinite(v)
                            for v in point
                        ):
                            raise ValueError("Invalid land coordinate.")
                        if not -180 <= point[0] <= 180 or not -90 <= point[1] <= 90:
                            raise ValueError("Land coordinate outside geographic bounds.")
        return land, manifest
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(
            "Globe context unavailable. Use the preserved 2D map or restore data/globe."
        ) from exc


def vessel_marks(state: dict, selected: str | None) -> list[dict]:
    """Use accepted positions only. Presentation fields never replace source evidence."""
    result = []
    for position in state["positions"]:
        result.append(
            position
            | {
                "position": [position["longitude"], position["latitude"]],
                "colour": [255, 195, 105] if position["freshness"] == "Stale" else [107, 226, 200],
                "size": 30 if position["vessel_id"] == selected else 21,
                "summary": (
                    f"{position['vessel_id']} · {position['vessel_class']} · SYNTHETIC\n"
                    f"{position['freshness']} · {position['age_minutes']:.0f} min old\n"
                    f"Observed {position['observed_at']}\nReceived {position['received_at']}"
                ),
            }
        )
    return result


def track_segments(state: dict, selected: str | None) -> list[dict]:
    """Join short observed intervals; never bridge outages or include future receipts."""
    if selected is None:
        return []
    cursor = parse_time(state["as_of_utc"])
    observations = sorted(
        (
            p
            for p in state["observations"]
            if p["vessel_id"] == selected and parse_time(p["received_at"]) <= cursor
        ),
        key=lambda p: parse_time(p["observed_at"]),
    )
    segments = []
    for a, b in zip(observations, observations[1:], strict=False):
        gap = (parse_time(b["observed_at"]) - parse_time(a["observed_at"])).total_seconds() / 60
        if 0 < gap <= MAX_TRACK_GAP_MINUTES:
            segments.append(
                {
                    "path": [[a["longitude"], a["latitude"]], [b["longitude"], b["latitude"]]],
                    "from_id": a["observation_id"],
                    "to_id": b["observation_id"],
                }
            )
    return segments


def selected_vessel(event: object, allowed: set[str]) -> str | None:
    """Treat all client selection attributes as untrusted; resolve identity server-side."""
    if not isinstance(event, dict):
        return None
    selection = event.get("selection")
    if not isinstance(selection, dict) or not isinstance(selection.get("objects"), dict):
        return None
    objects = selection["objects"].get(VESSEL_LAYER)
    if not isinstance(objects, list) or len(objects) != 1 or not isinstance(objects[0], dict):
        return None
    identity = objects[0].get("vessel_id")
    return identity if isinstance(identity, str) and identity in allowed else None


def camera(preset: str, extent: list | tuple) -> dict:
    """Bound zoom below GlobeView's precision limit; no terrain/navigation claim."""
    if preset == "Odisha":
        return {"longitude": 86.9, "latitude": 19.85, "zoom": 6.6}
    if preset == "Filtered area":
        west, south, east, north = extent
        span = max(east - west, north - south, 0.1)
        return {
            "longitude": (west + east) / 2,
            "latitude": (south + north) / 2,
            "zoom": max(1.0, min(8.0, math.log2(250 / span))),
        }
    return {"longitude": 82.0, "latitude": 20.0, "zoom": 0.5 if preset == "Globe" else 1.1}


def graticule() -> list[dict]:
    return [{"path": [[lon, lat] for lat in range(-90, 91, 5)]} for lon in range(-180, 180, 15)] + [
        {"path": [[lon, lat] for lon in range(-180, 181, 5)]} for lat in range(-75, 76, 15)
    ]
