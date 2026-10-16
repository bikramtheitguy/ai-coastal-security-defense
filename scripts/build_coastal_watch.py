"""Build public/synthetic assets from the pinned, locally supplied upstream file."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coastal_security.geography import (  # noqa: E402
    EXTENT,
    SOURCE_URL,
    UPSTREAM_REVISION,
    UPSTREAM_SHA256,
    coastline_fixture,
    digest,
    json_bytes,
)
from coastal_security.india import ADMIN_HASHES, GEOB_REVISION  # noqa: E402
from coastal_security.watch import SEED, VERSION, generate_exercise  # noqa: E402


def build(source: Path, admin_source: Path, output: Path) -> None:
    if output.exists():
        raise ValueError(
            "Output already exists; choose a new directory and compare before replacing."
        )
    coast = json_bytes(coastline_fixture(source.read_bytes()))
    exercise = json_bytes(generate_exercise())
    admin = admin_source.read_bytes()
    reference = json.loads(admin)
    if reference["revision"] != GEOB_REVISION or len(reference["states"]) != 13:
        raise ValueError("Unexpected administrative reference bundle.")
    registry = {
        "schema_version": 1,
        "crs": "OGC:CRS84",
        "coordinate_order": "longitude, latitude",
        "units": {"coordinates": "decimal degrees", "speed": "knots", "age": "minutes"},
        "extent": list(EXTENT),
        "assets": {
            "india-reference.json": {
                "source_type": "public",
                "provider": "geoBoundaries",
                "revision": GEOB_REVISION,
                "upstream_sha256": ADMIN_HASHES,
                "source_urls": [
                    f"https://github.com/wmgeolab/geoBoundaries/blob/{GEOB_REVISION}/releaseData/gbOpen/IND/ADM{level}/geoBoundaries-IND-ADM{level}_simplified.geojson"
                    for level in (1, 2)
                ],
                "sha256": digest(admin),
                "bytes": len(admin),
                "license": "State polygons CC BY 2.5 IN; district database ODbL 1.0",
                "notices": reference["notices"],
            },
            "coastline.geojson": {
                "source_type": "public",
                "provider": "Natural Earth",
                "dataset": "ne_50m_coastline",
                "dataset_version": "4.1.0",
                "revision": UPSTREAM_REVISION,
                "source_url": SOURCE_URL,
                "upstream_sha256": UPSTREAM_SHA256,
                "version_url": f"https://github.com/nvkelso/natural-earth-vector/blob/{UPSTREAM_REVISION}/50m_physical/ne_50m_coastline.VERSION.txt",
                "license": "Public domain",
                "license_url": "https://www.naturalearthdata.com/about/terms-of-use/",
                "source_checked_on": "2026-10-15",
                "observation_time": None,
                "scale": "1:50 million",
                "sha256": digest(coast),
                "bytes": len(coast),
                "transformation": (
                    "Clip line segments to registered extent using Liang-Barsky; "
                    "retain disconnected parts; round to six decimals. No inferred "
                    "land polygon or operational boundaries."
                ),
                "limitations": (
                    "Generalised historical coastline; not navigation, precise coastal"
                    " distance or legal boundaries. Source checking date is not survey"
                    " date."
                ),
            },
            "exercise.json": {
                "source_type": "synthetic",
                "provider": "Project generator",
                "version": VERSION,
                "seed": SEED,
                "generator": "coastal_security/watch.py",
                "license": "MIT",
                "license_url": "https://github.com/bikramtheitguy/ai-coastal-security-defense/blob/main/LICENSE",
                "sha256": digest(exercise),
                "bytes": len(exercise),
                "limitations": (
                    "Invented vessels, receipt times, outages and exercise area. No "
                    "real AIS, clients, ports, security facilities or operational "
                    "incidents."
                ),
            },
        },
    }
    output.mkdir(parents=True)
    for name, payload in (
        ("coastline.geojson", coast),
        ("exercise.json", exercise),
        ("india-reference.json", admin),
        ("registry.json", json_bytes(registry)),
    ):
        (output / name).write_bytes(payload)
    print(
        f"Wrote public coastline ({len(coast)} bytes) "
        f"and synthetic exercise ({len(exercise)} bytes)."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--admin", type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.admin, args.output)
