# Fictional harbour and landing-centre records

## Purpose and boundary

Roadmap item 4 called for fictional harbour/landing-centre records with
conspicuous synthetic identifiers, without client coordinates or operational
features. This increment adds that shore-side context to India Coastal Watch.
It does not link vessels to sites, infer arrivals or departures, or add any
alert rule. Every existing count, flag, export and investigation is unchanged.

The records are invented. They are not real ports, fishing harbours, landing
centres, facilities or jurisdictions. No port list, client map, Drive dataset
or facility database was consulted. A site can fall near a real place by
coincidence; it does not represent that place.

## Records

`data/coastal_watch/sites.json` holds 26 records, two per coastal state/UT:

| Field | Meaning |
| --- | --- |
| `site_id` | `SIM-HBR-<state>-01` (fictional harbour) or `SIM-LDC-<state>-02` (fictional landing centre) |
| `site_type`, `exercise_state`, `state_name`, `coast` | Exercise grouping, matching the vessel cohort filters |
| `longitude`, `latitude` | OGC:CRS84 decimal degrees, copied unchanged from the reference vertex |
| `reference_vertex` | Polygon, ring and vertex index in that state/UT's geoBoundaries geometry |
| `anchor_km`, `coastline_km` | Placement diagnostics, rounded to 0.1 km |
| `source` | Always `synthetic` |

There are deliberately no names, capacities, berths, traffic, ownership,
security or operational attributes.

## Placement rule

For each coastal state/UT, `coastal_security/sites.py`:

1. starts from that state/UT's existing fictional exercise anchor in
   `coastal_security/india.py`;
2. orders the vertices of its geoBoundaries reference boundary by
   great-circle distance from the anchor, ties broken by polygon, ring and
   vertex index;
3. accepts a vertex only within 8 km of the Natural Earth 1:50 million
   coastline, and at least 25 km from the site already chosen;
4. keeps the first two accepted vertices: the harbour, then the landing centre.

Using the state/UT's own boundary keeps each site inside its exercise cohort by
construction. The coastline screen keeps sites off inland borders: without it,
Puducherry's second candidate fell inland. Distances use a spherical Earth
(radius 6371.0088 km) and are a spacing aid, not a survey measure.

## Integrity and reproducibility

`scripts/build_coastal_watch.py` writes `sites.json` and registers its SHA-256,
size, generator, version, derivation and licence in `registry.json`. Rebuilding
from the pinned Natural Earth file reproduces the other three committed
fixtures byte for byte; the registry gains only the `sites.json` entry.
`load_sites` fails closed on a missing, altered or unregistered fixture, and
checks every coordinate against its referenced boundary vertex.

Embedded registries make every existing evidence export include the new asset
entry. Export structure and hash rules are otherwise unchanged.

Vertex coordinates derive from geoBoundaries state polygons (DataMeet India
community / Election Commission of India, CC BY 2.5 IN); see
[NOTICE](../data/coastal_watch/NOTICE.md). The generator is MIT.

## In the app

- **Fictional sites** layer toggle, on by default. Lavender diamonds on the 2D
  map; lavender dots on the globe. Identities are labelled landward when 12 or
  fewer sites are in scope. Globe sites are display-only: picking remains
  limited to validated vessel identities.
- **Fictional harbour and landing-centre records** expander under the review
  queue: the scoped table, placement rule and a hash-verifiable JSON download.
- Scope follows the coast and state/UT filters. District focus changes the
  viewport only.

## Validation

- `tests/test_sites.py`: reproducibility, registry entry, fixed fields, coastline
  and separation rules, rejection of altered records, fail-closed loading, scope,
  export hash, Vega-Lite schema and globe picking contract.
- `tests/test_watch_app.py`: Odisha table, caption, toggle and unchanged metrics.
- `browser_tests/test_sites_browser.py`: real Chromium at desktop and 390 px
  width: map labels, records, download hash, toggle and no horizontal overflow.

## Not included

- Vessel-to-site association, arrival/departure events or dwell rules.
- Real port or landing-centre data. Any public source would need its own
  licence, currency and accuracy review before use, and is outside this
  increment.
