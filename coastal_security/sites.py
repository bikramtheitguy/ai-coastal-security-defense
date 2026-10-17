"""Fictional harbour and landing-centre records placed on public reference geometry.

Sites are invented exercise context. Each coordinate is copied from a vertex of
the selected state/UT's public reference boundary; no facility, client or
operational dataset is consulted. Records carry identities and provenance only:
no names, capacities, berths, traffic or security attributes.
"""

import json
import math
import re
from pathlib import Path

from coastal_security.geography import DATA, EXTENT, digest, json_bytes
from coastal_security.india import COASTAL_STATES, polygons

VERSION = "coastal-sites-v1"
SITE_TYPES = {"HBR": "Fictional harbour", "LDC": "Fictional landing centre"}
COASTLINE_KM = 8.0
SEPARATION_KM = 25.0
SEARCH_KM = 400.0
EARTH_KM = 6371.0088
ID_PATTERN = re.compile(r"SIM-(HBR|LDC)-([A-Z]{2})-0([12])")
FIELDS = {
    "site_id",
    "site_type",
    "exercise_state",
    "state_name",
    "coast",
    "longitude",
    "latitude",
    "source",
    "reference_vertex",
    "anchor_km",
    "coastline_km",
}
RULES = {
    "anchor": "Existing fictional exercise anchor for each coastal state/UT",
    "candidates": "Vertices of that state/UT's geoBoundaries reference boundary",
    "coastline_km": COASTLINE_KM,
    "separation_km": SEPARATION_KM,
    "order": "Nearest eligible vertex to the anchor first; ties by polygon, ring, vertex",
    "distance": "Great-circle on a sphere of radius 6371.0088 km; spacing aid only",
}
LIMITATIONS = (
    "Invented sites on generalised public geometry. Not real ports, harbours, landing "
    "centres, facilities or jurisdictions; a site may fall near a real place by "
    "coincidence. Not for navigation or operational planning."
)


def great_circle_km(a: list[float], b: list[float]) -> float:
    lat1, lat2 = math.radians(a[1]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, math.radians(b[0] - a[0])
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_KM * math.asin(math.sqrt(h))


def _segment_km(point: list[float], a: list[float], b: list[float]) -> float:
    """Local equirectangular distance; adequate for an 8 km proximity screen."""
    scale = math.cos(math.radians(point[1]))
    ax, ay = (a[0] - point[0]) * scale, a[1] - point[1]
    bx, by = (b[0] - point[0]) * scale, b[1] - point[1]
    dx, dy = bx - ax, by - ay
    length = dx * dx + dy * dy
    t = 0.0 if length == 0 else max(0.0, min(1.0, -(ax * dx + ay * dy) / length))
    return math.hypot(ax + t * dx, ay + t * dy) * math.pi / 180 * EARTH_KM


def coastline_km(point: list[float], segments: list[tuple]) -> float:
    nearby = [
        (a, b)
        for a, b in segments
        if min(a[1], b[1]) - 0.2 <= point[1] <= max(a[1], b[1]) + 0.2
        and min(a[0], b[0]) - 0.2 <= point[0] <= max(a[0], b[0]) + 0.2
    ]
    return min((_segment_km(point, a, b) for a, b in nearby), default=math.inf)


def generate_sites(admin: dict, coast: dict) -> dict:
    states = {f["properties"]["state_iso"]: f for f in admin["states"]}
    segments = [
        (a, b)
        for feature in coast["features"]
        for a, b in zip(
            feature["geometry"]["coordinates"], feature["geometry"]["coordinates"][1:], strict=False
        )
    ]
    sites = []
    for iso, (name, coast_name, longitude, latitude) in COASTAL_STATES.items():
        anchor = [longitude, latitude]
        candidates = sorted(
            (great_circle_km(anchor, point), p, r, v, point)
            for p, polygon in enumerate(polygons(states[iso]["geometry"]))
            for r, ring in enumerate(polygon)
            for v, point in enumerate(ring[:-1])  # closing point repeats the first
        )
        picks = []
        for distance, p, r, v, point in candidates:
            if distance > SEARCH_KM or len(picks) == 2:
                break
            shore = coastline_km(point, segments)
            if shore <= COASTLINE_KM and all(
                great_circle_km(point, chosen[4]) >= SEPARATION_KM for chosen in picks
            ):
                picks.append((distance, p, r, v, point, shore))
        if len(picks) != 2:
            raise ValueError(f"No eligible fictional site placement for {iso}.")
        for number, (code, (distance, p, r, v, point, shore)) in enumerate(
            zip(SITE_TYPES, picks, strict=True), start=1
        ):
            sites.append(
                {
                    "site_id": f"SIM-{code}-{iso[3:]}-0{number}",
                    "site_type": SITE_TYPES[code],
                    "exercise_state": iso,
                    "state_name": name,
                    "coast": coast_name,
                    "longitude": point[0],
                    "latitude": point[1],
                    "source": "synthetic",
                    "reference_vertex": {"polygon": p, "ring": r, "vertex": v},
                    "anchor_km": round(distance, 1),
                    "coastline_km": round(shore, 1),
                }
            )
    return {
        "version": VERSION,
        "source": "synthetic",
        "rules": RULES,
        "limitations": LIMITATIONS,
        "sites": sites,
    }


def validate_sites(data: dict, admin: dict) -> None:
    if data.get("version") != VERSION or data.get("source") != "synthetic":
        raise ValueError("Only the versioned fictional site records are supported.")
    states = {f["properties"]["state_iso"]: f for f in admin["states"]}
    sites = data.get("sites", [])
    if len(sites) != 2 * len(COASTAL_STATES):
        raise ValueError("Expected two fictional sites per coastal state/UT.")
    seen, per_state = set(), {}
    for site in sites:
        match = ID_PATTERN.fullmatch(str(site.get("site_id")))
        if set(site) != FIELDS or not match or site["site_id"] in seen:
            raise ValueError("Fictional site records require the fixed fields and identities.")
        seen.add(site["site_id"])
        iso = site["exercise_state"]
        code, suffix, number = match.groups()
        if (
            site["source"] != "synthetic"
            or iso not in COASTAL_STATES
            or iso[3:] != suffix
            or site["site_type"] != SITE_TYPES[code]
            or number != ("1" if code == "HBR" else "2")
            or (site["state_name"], site["coast"]) != COASTAL_STATES[iso][:2]
        ):
            raise ValueError("Fictional site identity does not match its exercise state.")
        index = site["reference_vertex"]
        try:
            vertex = polygons(states[iso]["geometry"])[index["polygon"]][index["ring"]][
                index["vertex"]
            ]
        except (IndexError, KeyError, TypeError) as exc:
            raise ValueError("Fictional site references a missing boundary vertex.") from exc
        point = [site["longitude"], site["latitude"]]
        if list(vertex) != point:
            raise ValueError("Fictional site coordinate differs from its reference vertex.")
        if not EXTENT[0] <= point[0] <= EXTENT[2] or not EXTENT[1] <= point[1] <= EXTENT[3]:
            raise ValueError("Fictional site outside the registered extent.")
        per_state[iso] = per_state.get(iso, 0) + 1
    if set(per_state.values()) != {2}:
        raise ValueError("Expected two fictional sites per coastal state/UT.")


def load_sites(admin: dict, directory: Path = DATA) -> dict:
    """Fail closed on a missing, corrupt or unregistered site fixture; no network."""
    try:
        registry = json.loads((directory / "registry.json").read_bytes())
        record = registry["assets"]["sites.json"]
        path = directory / "sites.json"  # fixed filename, never a registry path
        if path.stat().st_size > 200_000:
            raise ValueError("Fictional site fixture exceeds its size bound.")
        payload = path.read_bytes()
        if digest(payload) != record["sha256"] or len(payload) != record["bytes"]:
            raise ValueError("Integrity check failed for sites.json.")
        data = json.loads(payload)
        validate_sites(data, admin)
        return data
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(
            "Fictional site records unavailable or invalid. "
            "Restore the committed data/coastal_watch bundle."
        ) from exc


def sites_in_scope(
    data: dict, state_iso: str = "All coastal states/UTs", coast: str = "All coasts"
) -> list[dict]:
    """Same state/coast semantics as the vessel cohort; district focus is viewport only."""
    return [
        site
        for site in data["sites"]
        if state_iso in {"All coastal states/UTs", site["exercise_state"]}
        and coast in {"All coasts", site["coast"]}
    ]


def sites_export(data: dict, registry: dict, state_iso: str, coast: str) -> bytes:
    payload = {
        "version": data["version"],
        "source": "synthetic",
        "exercise_state": state_iso,
        "coast": coast,
        "sites": sites_in_scope(data, state_iso, coast),
        "rules": data["rules"],
        "limitations": data["limitations"],
        "registry": registry,
    }
    payload["records_sha256"] = digest(json_bytes(payload))
    payload["hash_scope"] = (
        "Canonical JSON before adding records_sha256 and hash_scope; "
        "integrity fingerprint, not signature."
    )
    return json_bytes(payload)
