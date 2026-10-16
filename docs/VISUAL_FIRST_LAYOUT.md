# Graphics before controls and evidence

The five main analytical pages now put their visual below the page heading
and synthetic-data label. Native Streamlit containers establish the reading
order; there is no CSS reordering or duplication of charts.

| Page | Previous first content | Current first visual |
| --- | --- | --- |
| Drone observation classification | Metrics, limitations and prediction table expander | Interactive saved-prediction board |
| Cybersecurity classification experiment | Metrics, limitations and prediction table expander | Interactive detection matrix |
| Synthetic vessel scenario replay | Vessel/time controls and metrics | Replay clock, position plot and speed plot |
| Vessel trajectory evaluation | Scope filters and held-out metrics | Detection and false-alert workload comparison |
| India Coastal Watch | Geographic filters, clock, metrics and layer controls | Full-width India globe, or selected 2D map |

All filters, evidence tables, reviewer forms, exports, provenance, original
research charts and metric definitions remain available. The globe inspector
is below the map and retains the investigation handoff. The map stays visible
above the Watch picture, Intake quality, Investigations and Sources & export
tabs, so their evidence shares the same geographic context. Camera presets
follow the globe. The comparison chart likewise remains above the evaluation
tabs and responds to the same scenario and rule filters.

Replay retains its app-level playback controls and fragment timer. A single
`st.empty()` slot replaces the two charts on each fragment tick, including
the empty-vessel state, preventing accumulation outside the fragment.

The visual change adds no geography to drone or cyber records, no live feeds,
and no new model claims. Existing saved predictions and public/synthetic
geographic sources are unchanged. Lovable remains a hyperlink only;
project relocation remains on hold.

## Verification

The browser suite checks all five pages at 1365 × 900 and 390 × 844:

- Charts begin in the first viewport, below the page heading.
- Controls, metrics and evidence follow the visual in document order.
- Charts remain within the viewport, without horizontal page overflow.
- The globe uses its full content container width.
- Two replay charts remain after autoplay, pause, seeking and end-of-replay.

Existing checks still cover real globe pixels and picking, Odisha review
handoff and recovery, scope/layer filters, research outcomes, benchmark
evidence, both JSON export workflows, reset and original artifact rendering.
Console errors and external requests remain failures. CI publishes
`visual-first-<page>-<width>.jpg` previews with the usual trace artifacts.
