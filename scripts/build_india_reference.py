"""Derive licensed Indian coastal-state/district context from pinned public files."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coastal_security.geography import digest  # noqa: E402
from coastal_security.india import (  # noqa: E402
    ADMIN_HASHES,
    COASTAL_STATES,
    GEOB_REVISION,
    bounds,
    contains,
    geometry_checks,
    interior_point,
    plain_name,
)


def build(state_file: Path, district_file: Path, output: Path) -> None:
    if output.exists():
        raise ValueError("Choose a new output file; existing reference data are preserved.")
    inputs = []
    for level, path in ((1, state_file), (2, district_file)):
        data = path.read_bytes()
        if digest(data) != ADMIN_HASHES[level]:
            raise ValueError("Public boundary hash differs from pinned LFS source.")
        inputs.append(json.loads(data))
    states, districts, unmatched = [], [], []
    for feature in inputs[0]["features"]:
        source = feature["properties"]
        iso = source["shapeISO"]
        if iso not in COASTAL_STATES:
            continue
        geometry_checks(feature["geometry"])
        states.append(
            {
                "type": "Feature",
                "properties": {
                    "id": source["shapeID"],
                    "name": COASTAL_STATES[iso][0],
                    "source_name": source["shapeName"],
                    "state_iso": iso,
                    "coast": COASTAL_STATES[iso][1],
                },
                "geometry": feature["geometry"],
                "bbox": bounds(feature["geometry"]),
            }
        )
    for feature in inputs[1]["features"]:
        geometry_checks(feature["geometry"])
        point = interior_point(feature["geometry"])
        parents = [
            s
            for s in states
            if s["bbox"][0] <= point[0] <= s["bbox"][2]
            and s["bbox"][1] <= point[1] <= s["bbox"][3]
            and contains(point, s["geometry"])
        ]
        if len(parents) != 1:
            unmatched.append(
                {
                    "id": feature["properties"]["shapeID"],
                    "name": feature["properties"]["shapeName"],
                    "candidate_coastal_states": len(parents),
                }
            )
            continue
        source = feature["properties"]
        districts.append(
            {
                "type": "Feature",
                "properties": {
                    "id": source["shapeID"],
                    "name": plain_name(source["shapeName"]),
                    "source_name": source["shapeName"],
                    "state_iso": parents[0]["properties"]["state_iso"],
                    "association": (
                        "interior-point reference association; not official parent registry"
                    ),
                },
                "geometry": feature["geometry"],
                "bbox": bounds(feature["geometry"]),
                "reference_point": point,
            }
        )
    data = {
        "version": "india-coastal-reference-v1",
        "crs": "OGC:CRS84",
        "revision": GEOB_REVISION,
        "states": sorted(states, key=lambda s: s["properties"]["name"]),
        "districts": sorted(
            districts, key=lambda d: (d["properties"]["state_iso"], d["properties"]["name"])
        ),
        "source_counts": {
            "states": len(inputs[0]["features"]),
            "districts": len(inputs[1]["features"]),
        },
        "excluded_from_coastal_reference": unmatched,
        "notices": {
            "states": (
                "DataMeet India community / Election Commission of India via "
                "geoBoundaries; CC BY 2.5 IN "
                "https://creativecommons.org/licenses/by/2.5/in/"
            ),
            "districts": (
                "Contains information from geoBoundaries / Pathways Data Pvt. Ltd."
                " / lgdirectory.gov.in, made available under ODbL 1.0 "
                "https://opendatacommons.org/licenses/odbl/1-0/. This derived "
                "district database and association fields are offered under ODbL "
                "1.0."
            ),
            "vintage": (
                "ADM1 metadata reports 2011 but includes later administrative "
                "units; temporal currency is unverified. ADM2 metadata represents "
                "2021 and advertises 736 units, while the pinned simplified file "
                "contains 735. No current official jurisdiction claim."
            ),
            "transformation": (
                "Coastal states selected by source ISO. Districts associated using"
                " an interior point and exact polygon containment; unmatched units"
                " retained in exclusion ledger. Coordinates and polygon topology "
                "are unmodified. District list includes inland districts within "
                "coastal states."
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "states": len(states),
                "districts": len(districts),
                "excluded": len(unmatched),
                "bytes": output.stat().st_size,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--districts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.states, args.districts, args.output)
