# Spatial UI and public-data roadmap

## Implemented in this increment

The Streamlit application uses a spatial-inspired interface: layered navy surfaces, quiet borders, rounded cards, restrained blue accents, readable high-contrast text and consistent numeric typography. Assets and fonts are local/system-provided. There is no external font, analytics, map-tile or animation service. Native Streamlit controls keep their semantics and keyboard behaviour; the layout supports narrow viewports and reduced-motion preferences. Cosmetic CSS is isolated in `dashboard/spatial_style.py`; theme tokens live in `.streamlit/config.toml`.

The new evaluation page orders the workflow as scope → outcomes → method comparison → evidence → protocol and export. Meaning is conveyed through text, labels and counts as well as colour. No glowing severity indicators, confidence gauges or “live” badges imply evidence the system does not have. Existing experiment plots are preserved as historical images.

India Coastal Watch extends that theme with public geographic context, layer controls, source/receipt quality and last-known positions. National-view vessel labels are suppressed to reduce overlap; state cohorts of 12 or fewer show identifiers, with hover details and the inspector available at every scope. The [watch comparison](COASTAL_WATCH_COMPARISON.md) records the current increment.

## Data boundary

Public GitHub demonstrations use verified public sources or clearly labelled fictional synthetic data. Client-provided references remain outside the public repository until dataset-specific rights and approval are established. Synthetic data must never be described as unavailable real intelligence, recovered observations or verified operational records. There are no Lovable integrations: the coastal website remains an external showcase hyperlink only. No bkdigital.tech assets or settings are involved.

## Public-source decisions

| Source | Suitable use | Important limitation |
| --- | --- | --- |
| [Natural Earth terms](https://www.naturalearthdata.com/about/terms-of-use/) and [data creation](https://www.naturalearthdata.com/about/data-creation/) | Integrated: pinned 1:50 million public-domain coastline | Generalised context; not navigation, precise shore distance or port geometry |
| [geoBoundaries references and notices](../data/coastal_watch/NOTICE.md) | Integrated: 13 coastal states/UTs and 229 historical district references | Separate licences, mixed vintages and spatial parent associations; not current official jurisdiction |
| [NOAA AccessAIS](https://www.coast.noaa.gov/digitalcoast/tools/ais.html) | Historical traffic fixtures and external behaviour evaluation within stated U.S. coastal coverage | Not an Indian live surveillance feed or incident-label source; review the selected dataset's metadata and terms before ingestion |
| Seeded project generators | Missing vessel scenarios, outages and human review exercises | Fictional, labelled, versioned and reproducible; no real-world performance claim |

The original source review was on 15 October 2026; Coastal Watch's pinned geography was reviewed on 16 October. NOAA AIS remains a candidate and is not integrated. Source-check dates do not establish observation freshness.

## Geospatial progress and next steps

1. Implemented: pinned registry with origin, licence, version, hashes, units, coordinate reference system and synthetic status; separate observed/received timestamps in the traffic contract.
2. Implemented: bounded public geography, geometry checks and byte-integrity validation. Display copies adapt polygon orientation without rewriting the source bundle.
3. Implemented: selectable layers, freshness legend, exercise clock, explicit unknown/stale states and evidence export. Counts follow coast/state cohorts; district selection changes viewport only.
4. Implemented: 26 fictional harbour/landing-centre records with conspicuous `SIM-HBR`/`SIM-LDC` identities, placed on public reference-boundary vertices and carrying no operational attributes. See [fictional sites](FICTIONAL_SITES.md). No client coordinates or facility data are used.
5. Continuing release gates: map alignment, narrow layouts and browser errors. Before real-data use, add independent geographic validation, accessibility review, measured performance and geodesic event evaluation.

## Credible operational-workflow roadmap

Preserve a traceable chain: observation → data-quality check → explainable flag → human review → recorded disposition. Later increments need delayed/out-of-order messages, vessel-class and weather context, persistent authenticated reviews, server-side authorization, audit retention, evidence export, incident summaries with citations and abstention, and monitoring of alert workload. Document proposed capabilities separately until tested. National-scale capacity requires explicit workload targets, load and recovery tests, independent security review and authorized field validation.
