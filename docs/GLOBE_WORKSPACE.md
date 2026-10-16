# India Coastal Watch: interactive globe and research review

This increment enhances the Python Streamlit application. It does not connect to or
change either Lovable project. Traffic remains a deterministic synthetic exercise;
no live AIS, operational inventory, map subscription or authenticated case service
is introduced.

## What changes

| Before | This increment |
|---|---|
| App opens on a saved drone experiment | App opens on India Coastal Watch |
| Geographic watch uses a fixed Mercator projection | Native interactive globe with rotation, zoom, India/Odisha/filter presets and vessel picking |
| Vessel inspection requires a dropdown lookup | Pick a vessel or use the accessible dropdown; the server resolves its current evidence |
| Watch and review are separate navigation steps | Review button focuses the selected flagged vessel or existing case, retaining explicit human review |
| Research pages lead with large white PNGs | Drone observation review board and cyber outcome matrix, with selection, row inspection and CSV export |
| Static charts are the primary research presentation | Original charts remain available in labelled expandable sections; their files and hashes are unchanged |

All six modules, the 2D map, state/UT/district filters, intake ledger, rule explanations,
evaluation, replay, casebook import/export, human decisions and saved artifacts remain.
Historical model limitations and their measured false alerts/misses are unchanged.
The review panels do not fabricate geolocation for prediction rows lacking source IDs.

## Rendering choice

The preliminary plan considered CesiumJS. Inspection of the pinned Streamlit 1.66.0
frontend and [chart API](https://docs.streamlit.io/develop/api-reference/charts/st.pydeck_chart)
confirmed native `GlobeView`, stable view state and server-side selection callbacks.
This implementation uses pydeck 0.9.3, already in the lockfile and now declared as a
direct dependency. It adds no frontend build, provider account, terrain service or
runtime CDN dependency. `map_provider=None` disables external basemap tiles.

[deck.gl GlobeView](https://deck.gl/docs/api-reference/core/globe-view) is experimental.
This is a bounded research globe, with zoom limited below its high-zoom precision
limits. The existing 2D map is an explicit fallback, accessible independently of globe
asset availability or GPU support. There is no promise of terrain, satellite imagery,
navigation accuracy or operational capacity.

The map uses public-domain Natural Earth 1:110m land for global context plus the existing
1:50m India coastline. The new land asset has a separate integrity manifest under
`data/globe`; adding it does not change the existing exercise/casebook source hashes.
The geometry is generalised; it is not a coastline survey or a political boundary map.
Existing administrative references retain their licences and historical-vintage caveats.

## Evidence contract

- The existing `snapshot` function remains the sole source of vessel positions and counts.
- Only receipts available at the selected clock appear. Unknown locations stay unknown.
- Amber means an observation more than 15 minutes old, not a threat classification.
- Vessel glyphs identify records; glyph orientation is not a measured heading.
- Selected-vessel history connects accepted samples at most ten minutes apart. These
  segments are visual connections between observations, not measured intervening paths.
  Reporting gaps are not bridged or animated as known travel.
- Client selection is an identity hint only. The server validates the vessel against
  the current cohort and ignores browser-supplied coordinates, speeds and evidence.
- Camera movement stays in the browser; selection triggers the usual Streamlit rerun.
  Explicit reset/preset and geographic-filter changes reset the camera; time changes do not.
- Opening the investigation view does not create a case or record a decision. Existing
  acknowledgement and rationale forms still control every case action.

Research review rows are numbered by preserved CSV file order. These are not the
original training/test row IDs. The drone board shows 2 detected positives, 8 false
alerts, 14 misses and 176 correct negatives. The cyber matrix shows 0, 1, 4 and 195
respectively. Both total 200 and retain the existing precision/recall/F1 displays.

## Odisha walkthrough and validation

1. Open the app, select Odisha in State / Union Territory, and use the Odisha camera preset.
2. At exercise minute 60, select the amber SIM-004 glyph. The inspector must show a
   35-minute-old observation from 06:25 UTC, received at 06:26 UTC.
3. Choose **Review this vessel**, enter fictional review details and acknowledge the
   evidence to create a case through the existing workflow.
4. Set the replay clock to minute 95. SIM-004 is fresh with a 5-minute-old observation
   from 07:30 UTC. The 25→90 minute reporting gap remains open in its history.
5. Existing frozen evidence stays frozen. Capture new evidence explicitly; freshness
   alone does not close the case. Verify the JSON and handover downloads as before.
6. Switch to 2D map and compare counts, selected identity and evidence. Exercise island
   filters and the known Lakshadweep district-data gap.
7. Test both research review panels, empty outcome selections and filtered CSV exports;
   expand the original charts to verify they remain accessible.

Tests cover source integrity, receipt time, outage segmentation, untrusted picking,
native globe serialization, prediction reconciliation and the existing regressions.
Chromium QA must verify rendered land, vessel picking, review handoff, replay,
rotation/zoom and narrow layouts. Screenshots are collected as CI evidence.
Do not treat a non-empty WebGL canvas or passing Python tests alone as visual verification.

### Deck.gl expression regression

Chromium run `38853758843` at `a163d24` failed during JSON conversion, before the
vessel layer could render. In `dashboard/globe_view.py`, `icon_atlas=ICON` passed a
bare `data:image/svg+xml;base64,...` string to pydeck. pydeck 0.9.3 treats unquoted
strings as expressions and emitted `iconAtlas: "@@=data:image/svg+xml;base64,..."`.
The Deck.gl JSON converter removed `@@=` and parsed `data:` as JavaScript, producing
`Unexpected ":" at character 4`. The same implicit conversion affected the
`width_units`, `radius_units` and `size_units` constants (`@@=pixels`).

The fix uses `pydeck.types.String` for the atlas, icon name and pixel units. Data
accessors such as `get_position="position"` intentionally remain expressions.
This follows [pydeck's literal-string contract](https://deckgl.readthedocs.io/en/latest/layer.html#passing-string-constants).
Serialization tests cover all four camera presets at both stale and fresh Odisha
timestamps, including optional district layers, and reject expressions in constant
properties. Browser QA continues to fail on every console error, page error and
external request; none is filtered or suppressed. Picking is tested against the
rendered vessel glyph on desktop and through a touch-enabled narrow viewport.

## Boundaries and follow-up

This change improves presentation and exploration, not model validity or data coverage.
The 44 synthetic vessels do not establish national-scale performance. Real-data
integration, coastline-aware route generation, imagery licensing, authentication,
durable cases and calibrated detection remain separately reviewed work.
