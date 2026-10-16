# Platform concepts and implementation sequence

Reviewed 16 October 2026. These are independent design adaptations for the Python/Streamlit repository, not endorsements, provider integrations or feature-parity claims. No platform code, imagery, branding or proprietary feed was copied.

| Reference | Useful concept | This increment | Subsequent work |
|---|---|---|---|
| [MarineTraffic](https://support.marinetraffic.com/en/articles/13641583-how-can-i-track-a-vessel-s-historical-route) | Vessel details, timestamped past track and event timeline | Vessel inspector, observation history, UTC and IST clocks | Voyage segmentation and evidence-backed event timeline |
| [VesselFinder](https://www.vesselfinder.com/historical-ais-data) | Historical positions, playback, voyage and port events | Receipt-aware synthetic replay and rejected-record ledger | Approved historical AIS adapter; port events from licensed polygons |
| [Skylight](https://support.skylight.global/en_US/what-are-events) | Behaviour events separated from detections | Rule explanations linked to exact synthetic evidence | Proximity/dwell candidates, independent detection correlation |
| [Global Fishing Watch](https://api-doc.globalfishingwatch.org/our-apis/documentation/docs/v3/general-api-doc/data-caveats) | Apparent fishing activity and coverage caveats | Unknown and stale states; no intent or violation inference | Apparent-activity aggregates after separate real-data validation |
| [Flightradar24](https://fr24api.flightradar24.com/docs/endpoints/overview) | Aircraft identity, altitude, speed and historical tracks | Architecture reference only; no aircraft feed in this increment | Synthetic civilian aircraft layer and separate aviation schema |
| [OSINT Framework](https://osintframework.com/) | Categorised source discovery with access requirements | Checked source registry and provenance documentation | Curated source catalogue with review date, access and reuse terms |
| [OSIRIS](https://www.osirisai.live/docs) | Selectable map layers, consistent feed schemas, regional context | Layer toggles, India geographic scope, evidence export | Independent feed status, contextual hazards and linked incident evidence |

## India-first data design

Use west/east coasts and island territories, all 13 coastal states/UTs, and explicitly dated administrative references. Treat LGD codes, Census codes, postal codes and polygon feature IDs as different identifier systems. District aliases require a versioned crosswalk; point proximity does not establish jurisdiction. A civil district outline cannot define seaward police responsibility.

Drive material is assessed read-only outside this repository. Raw private files, internal inventories, credentials and client maps are not bundled. Supplied administrative extracts are candidates pending exact provenance, reuse and reconciliation. The committed map uses separately verified public Natural Earth and geoBoundaries sources only. Fictional vessels and events have synthetic identities and a deterministic generator.

## Event and model requirements

1. Keep observed, received, processed and analyst-review timestamps distinct. Replay only evidence available at its clock. Missing AIS/ADS-B does not establish intentional disabling.
2. Preserve the original observation and its source identity. Validate coordinate ranges, units, duplicate identity and timestamp causality before scoring. Retain rejected evidence with reasons.
3. Future proximity events need geodesic distances, bounded time alignment, duration and separate evidence for both objects. Do not bridge outages by drawing invented paths. Encounters are candidates, not proof of illicit transfer.
4. Keep evidence quality, completeness, event likelihood and analyst priority separate. Do not combine unrelated observations into arbitrary threat percentages. Do not borrow a demo risk-weight table as a calibrated model.
5. Evaluate by vessel/time-disjoint held-out data, false alerts per vessel-day, precision/recall, detection delay, event-duration error and missing-data robustness. Synthetic success cannot establish real-world performance.
6. Aircraft need aircraft/flight identifiers, altitude reference and units, position source and timestamp. They must not masquerade as AIS vessels. A shared timeline does not establish that nearby aircraft and vessels are related.
7. AI summaries must cite supplied evidence, distinguish observations from inference and abstain when evidence is missing. No automated dispatch, interception or operational command authority is implied.

## Integration boundaries

Use approved adapters with server-side credentials, bounded requests, caching, timeouts and explicit unavailable states. Do not scrape a public UI to bypass a data contract. [VesselFinder terms](https://www.vesselfinder.com/terms) and each provider's API agreement govern actual use. [Global Fishing Watch API terms](https://globalfishingwatch.org/our-apis/documentation/docs/license-rate-limits) specify noncommercial use and attribution; individual datasets may have additional restrictions. Source visibility is not a redistribution licence.

For CCTV, use generated test scenes or media whose owner has specifically authorised use. Do not ingest unsecured discovered cameras. Public news is contextual evidence with links and timestamps, not verified incident ground truth. Avoid operational deployment locations and restricted infrastructure datasets in the public POC.

Lovable remains an external showcase link only. The GitHub application is not connected to either Lovable project.
