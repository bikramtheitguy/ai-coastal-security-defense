# Spatial UI and public-data roadmap

## Implemented in this increment

The Streamlit application uses a spatial-inspired interface: layered navy surfaces, quiet borders, rounded cards, restrained blue accents, readable high-contrast text and consistent numeric typography. Assets and fonts are local/system-provided. There is no external font, analytics, map-tile or animation service. Native Streamlit controls keep their semantics and keyboard behaviour; the layout supports narrow viewports and reduced-motion preferences. Cosmetic CSS is isolated in `dashboard/spatial_style.py`; theme tokens live in `.streamlit/config.toml`.

The new evaluation page orders the workflow as scope → outcomes → method comparison → evidence → protocol and export. Meaning is conveyed through text, labels and counts as well as colour. No glowing severity indicators, confidence gauges or “live” badges imply evidence the system does not have. Existing experiment plots are preserved as historical images.

## Data boundary

Public GitHub demonstrations use verified public sources or clearly labelled fictional synthetic data. Client-provided references remain outside the public repository until dataset-specific rights and approval are established. Synthetic data must never be described as unavailable real intelligence, recovered observations or verified operational records. There are no Lovable integrations: the coastal website remains an external showcase hyperlink only. No bkdigital.tech assets or settings are involved.

## Verified candidate sources, not yet integrated

| Source | Suitable use | Important limitation |
| --- | --- | --- |
| [Natural Earth terms](https://www.naturalearthdata.com/about/terms-of-use/) and [data creation](https://www.naturalearthdata.com/about/data-creation/) | Public-domain small-scale geographic context | Generalised at 1:10 million and smaller scales; not navigation, precise shore distance or port geometry. Review boundary portrayal for intended use |
| [NOAA AccessAIS](https://www.coast.noaa.gov/digitalcoast/tools/ais.html) | Historical traffic fixtures and external behaviour evaluation within stated U.S. coastal coverage | Not an Indian live surveillance feed or incident-label source; review the selected dataset's metadata and terms before ingestion |
| Seeded project generators | Missing vessel scenarios, outages and human review exercises | Fictional, labelled, versioned and reproducible; no real-world performance claim |

Source descriptions were checked against their providers on 15 October 2026. Neither public dataset is downloaded or wired into this increment.

## Next geospatial increment

1. Add a data registry with origin, licence URL, version, checksum, observation/ingestion time, units, coordinate reference system, coverage and synthetic status.
2. Ingest a bounded approved geographic fixture; validate coordinate order/ranges, geometry and duplicate identifiers. Prefer original GIS geometries to tracing PDF images.
3. Add a geographic context view with selectable layers, source/freshness legend, synchronized replay and an explicit empty/stale state. Keep unknown observations distinct from zero activity.
4. Add fictional harbour/landing-centre records where public reuse cannot be verified, with conspicuous synthetic identifiers. Do not copy client coordinates or operational features.
5. Validate map alignment, distance calculations, performance, keyboard navigation, colour contrast and narrow-screen behaviour before release.

## Credible operational-workflow roadmap

Preserve a traceable chain: observation → data-quality check → explainable flag → human review → recorded disposition. Later increments need delayed/out-of-order messages, vessel-class and weather context, persistent authenticated reviews, server-side authorization, audit retention, evidence export, incident summaries with citations and abstention, and monitoring of alert workload. Document proposed capabilities separately until tested. National-scale capacity requires explicit workload targets, load and recovery tests, independent security review and authorized field validation.
