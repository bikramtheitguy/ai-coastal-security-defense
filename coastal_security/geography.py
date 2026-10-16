"""Bounded, offline public coastline processing and integrity checks."""

import hashlib
import json
import math
from pathlib import Path

EXTENT = (67.0, 5.5, 96.0, 28.5)  # Indian coastal context; CRS84 longitude first
UPSTREAM_REVISION = "ca96624a56bd078437bca8184e78163e5039ad19"
UPSTREAM_SHA256 = "271f1c4c1908312bac6b29d158ea1356544beafc129f260005300913aa5ea283"
SOURCE_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    f"{UPSTREAM_REVISION}/geojson/ne_50m_coastline.geojson"
)
DATA = Path(__file__).resolve().parents[1] / "data/coastal_watch"


def json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def clip_segment(a, b, bounds=EXTENT):
    """Liang-Barsky clipping in geographic coordinates; not a distance operation."""
    west, south, east, north = bounds
    dx, dy = b[0] - a[0], b[1] - a[1]
    low, high = 0.0, 1.0
    for p, q in ((-dx, a[0] - west), (dx, east - a[0]), (-dy, a[1] - south), (dy, north - a[1])):
        if p == 0:
            if q < 0:
                return None
        else:
            ratio = q / p
            if p < 0:
                low = max(low, ratio)
            else:
                high = min(high, ratio)
            if low > high:
                return None
    points = [[round(a[0] + t * dx, 6), round(a[1] + t * dy, 6)] for t in (low, high)]
    return points if points[0] != points[1] else None


def coastline_fixture(payload: bytes) -> dict:
    if digest(payload) != UPSTREAM_SHA256:
        raise ValueError("Upstream coastline checksum differs from the approved revision.")
    source = json.loads(payload)
    features = []
    for index, feature in enumerate(source["features"]):
        geometry = feature["geometry"]
        if geometry["type"] not in {"LineString", "MultiLineString"}:
            raise ValueError("Unexpected upstream coastline geometry.")
        lines = (
            [geometry["coordinates"]]
            if geometry["type"] == "LineString"
            else geometry["coordinates"]
        )
        for part, line in enumerate(lines):
            segment = []
            pieces = []
            for a, b in zip(line, line[1:], strict=False):
                clipped = clip_segment(a, b)
                if clipped is None:
                    if segment:
                        pieces.append(segment)
                        segment = []
                elif segment and segment[-1] == clipped[0]:
                    segment.append(clipped[1])
                else:
                    if segment:
                        pieces.append(segment)
                    segment = clipped
            if segment:
                pieces.append(segment)
            for number, coordinates in enumerate(pieces):
                features.append(
                    {
                        "type": "Feature",
                        "properties": {
                            "id": f"NE50-{index}-{part}-{number}",
                            "source": "Natural Earth",
                            "synthetic": False,
                        },
                        "geometry": {"type": "LineString", "coordinates": coordinates},
                    }
                )
    result = {"type": "FeatureCollection", "bbox": list(EXTENT), "features": features}
    validate_coastline(result)
    return result


def validate_coastline(data: dict) -> None:
    if data.get("type") != "FeatureCollection" or data.get("bbox") != list(EXTENT):
        raise ValueError("Invalid coastline extent or container.")
    features = data.get("features", [])
    if not 1 <= len(features) <= 500:
        raise ValueError("Expected a bounded coastline fixture.")
    ids = set()
    points = 0
    for feature in features:
        identity = feature["properties"]["id"]
        if identity in ids or feature["geometry"]["type"] != "LineString":
            raise ValueError("Duplicate identity or invalid coastline geometry.")
        ids.add(identity)
        coordinates = feature["geometry"]["coordinates"]
        if len(coordinates) < 2:
            raise ValueError("A coastline line needs at least two positions.")
        for point in coordinates:
            if len(point) != 2 or any(not math.isfinite(v) for v in point):
                raise ValueError("Invalid coastline coordinate.")
            lon, lat = point
            if not EXTENT[0] <= lon <= EXTENT[2] or not EXTENT[1] <= lat <= EXTENT[3]:
                raise ValueError("Coastline coordinate outside the registered region.")
        points += len(coordinates)
    if points > 50_000:
        raise ValueError("Coastline exceeds the demo rendering bound.")


def load_bundle(directory: Path = DATA) -> tuple[dict, dict, dict, dict]:
    """Fail closed on missing, corrupt or mismatched checked-in assets; no network."""
    try:
        registry = json.loads((directory / "registry.json").read_bytes())
        contents = {}
        for name in ("coastline.geojson", "exercise.json", "india-reference.json"):
            record = registry["assets"][name]
            path = directory / name  # fixed filenames, never arbitrary registry paths
            if path.stat().st_size > 6_000_000:
                raise ValueError("Fixture exceeds its size bound.")
            payload = path.read_bytes()
            if digest(payload) != record["sha256"] or len(payload) != record["bytes"]:
                raise ValueError(f"Integrity check failed for {name}.")
            contents[name] = json.loads(payload)
        validate_coastline(contents["coastline.geojson"])
        return (
            contents["coastline.geojson"],
            contents["exercise.json"],
            contents["india-reference.json"],
            registry,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(
            "Coastal Watch assets unavailable or invalid. "
            "Restore the committed data/coastal_watch bundle."
        ) from exc
