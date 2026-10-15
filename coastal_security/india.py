"""Indian coastal context, with explicit reference-data and exercise boundaries."""

import math
import unicodedata

COASTAL_STATES = {
    "IN-GJ": ("Gujarat", "West coast", 68.8, 21.2),
    "IN-MH": ("Maharashtra", "West coast", 72.5, 18.0),
    "IN-GA": ("Goa", "West coast", 73.5, 15.2),
    "IN-KA": ("Karnataka", "West coast", 74.3, 13.5),
    "IN-KL": ("Kerala", "West coast", 75.6, 10.5),
    "IN-TN": ("Tamil Nadu", "East coast", 80.6, 12.0),
    "IN-AP": ("Andhra Pradesh", "East coast", 83.5, 16.7),
    "IN-OR": ("Odisha", "East coast", 86.6, 19.5),
    "IN-WB": ("West Bengal", "East coast", 88.5, 20.9),
    "IN-DH": ("Dadra and Nagar Haveli and Daman and Diu", "West coast", 72.5, 20.1),
    "IN-PY": ("Puducherry", "East coast", 80.2, 11.8),
    "IN-LD": ("Lakshadweep", "Island territories", 72.0, 11.0),
    "IN-AN": ("Andaman and Nicobar Islands", "Island territories", 93.0, 11.5),
}
GEOB_REVISION = "9469f09592ced973a3448cf66b6100b741b64c0d"
ADMIN_HASHES = {
    1: "4c63fe43294a391e8f2de4e9f86f3edb60f8688275b9fee90c61fb2aa0c26061",
    2: "d68db39cd3e2d0892af268e2b0454166368ce3b5b8a78fcda63069ec92a641db",
}


def plain_name(value: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", value) if not unicodedata.combining(c))


def polygons(geometry: dict) -> list:
    if geometry["type"] == "Polygon":
        return [geometry["coordinates"]]
    if geometry["type"] == "MultiPolygon":
        return geometry["coordinates"]
    raise ValueError("Expected public polygon geometry.")


def bounds(geometry: dict) -> list[float]:
    points = [point for polygon in polygons(geometry) for ring in polygon for point in ring]
    return [
        min(p[0] for p in points),
        min(p[1] for p in points),
        max(p[0] for p in points),
        max(p[1] for p in points),
    ]


def in_ring(point: list[float], ring: list) -> bool:
    x, y = point
    inside = False
    for a, b in zip(ring, ring[1:], strict=False):
        if (a[1] > y) != (b[1] > y):
            crossing = a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            if x < crossing:
                inside = not inside
    return inside


def contains(point: list[float], geometry: dict) -> bool:
    return any(
        in_ring(point, rings[0]) and not any(in_ring(point, hole) for hole in rings[1:])
        for rings in polygons(geometry)
    )


def interior_point(geometry: dict) -> list[float]:
    """Point within the widest scanline interval, respecting holes and multipart islands."""
    candidates = []
    for rings in polygons(geometry):
        ys = sorted({p[1] for p in rings[0]})
        for fraction in (0.25, 0.5, 0.75):
            if len(ys) < 2:
                continue
            index = min(len(ys) - 2, int((len(ys) - 1) * fraction))
            y = (ys[index] + ys[index + 1]) / 2
            xs = []
            for ring in rings:
                for a, b in zip(ring, ring[1:], strict=False):
                    if (a[1] > y) != (b[1] > y):
                        xs.append(a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1]))
            xs.sort()
            for left, right in zip(xs, xs[1:], strict=False):
                point = [(left + right) / 2, y]
                if (
                    right > left
                    and in_ring(point, rings[0])
                    and not any(in_ring(point, hole) for hole in rings[1:])
                ):
                    candidates.append((right - left, point))
    if not candidates:
        raise ValueError("No defensible interior point for reference polygon.")
    return max(candidates, key=lambda pair: pair[0])[1]


def geometry_checks(geometry: dict) -> None:
    for rings in polygons(geometry):
        for ring in rings:
            if len(ring) < 4 or ring[0] != ring[-1]:
                raise ValueError("Unclosed or degenerate reference polygon.")
            for point in ring:
                if len(point) != 2 or not all(math.isfinite(v) for v in point):
                    raise ValueError("Invalid reference coordinate.")
                if not 66 <= point[0] <= 99 or not 5 <= point[1] <= 38:
                    raise ValueError("Reference coordinate outside Indian geographic envelope.")


def scope_geometry(
    admin: dict, state_iso: str, district_id: str, coast: str = "All coasts"
) -> tuple[list, list, list]:
    states = [
        s
        for s in admin["states"]
        if (state_iso == "All coastal states/UTs" or s["properties"]["state_iso"] == state_iso)
        and (coast == "All coasts" or s["properties"]["coast"] == coast)
    ]
    selected_ids = {s["properties"]["state_iso"] for s in states}
    districts = [
        d
        for d in admin["districts"]
        if d["properties"]["state_iso"] in selected_ids
        and (district_id == "All reference districts" or d["properties"]["id"] == district_id)
    ]
    visible = districts if district_id != "All reference districts" else states
    boxes = [feature["bbox"] for feature in visible]
    if not boxes:
        raise ValueError("No geography for the selected scope.")
    extent = [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]
    pad = max(0.35, (extent[2] - extent[0]) * 0.12, (extent[3] - extent[1]) * 0.12)
    return states, districts, [extent[0] - pad, extent[1] - pad, extent[2] + pad, extent[3] + pad]
