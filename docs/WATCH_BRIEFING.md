# India watch briefing

## Plan before implementation

Build on merged PR20 (`0b9e4c94eb4379f757f0faceaca21a48b8d44117`). A reviewer
can already inspect one vessel, replay its history and preserve a case. The next
gap is a fleet handover: what changed, what remains unknown, and which source
records explain the change across a selected Indian coast/state/class cohort?

Add a lazy **Watch briefing** tab. Preserve the globe-first layout, every earlier
page, investigation form and export schema. Use only the existing verified
public-geography bundle and versioned synthetic Indian exercise. No new service,
package, credential, model, client data or Lovable connection. Relocation is on hold.

1. Lead with receipt-causal freshness lanes, 12 vessels per page, using the
   existing history calculation. Default to vessels with current flags, unknown
   positions, a freshness change, stale duration or intake exceptions. Offer all
   vessels explicitly; display the shown/eligible/cohort counts.
2. Compare the last 30/60 minutes or full exercise with the same cohort at both
   endpoints. Rewinding recomputes both, excluding future receipts. Changes and
   intake counts cover `(baseline, as-of]`; baseline receipts are carry-in evidence.
3. Keep distinct current freshness, historical stale duration, current rule flags
   and delivery-quality exceptions. Never invent a risk score. Group rows by
   current stale, awaiting, rule flag, historical gap, intake issue, then routine;
   this is display order, not a validated operational priority model.
4. Link the selected row to existing vessel history and, where eligible, the
   existing investigation form. Navigation never creates a case or review.
5. Export a deterministic JSON evidence pack and plain-text handover. Include
   filters, baseline/current snapshots, definitions, source hashes and receipt IDs.
   SHA-256 verifies content identity, not authorship. No persistent shared state.

## Metric contract

| Measure | Meaning |
| --- | --- |
| Continuity review now | Distinct stale or awaiting-first-report vessels at as-of |
| Vessels with stale time | Distinct vessels with positive stale duration in the window, including recovered vessels |
| Intake exceptions | Distinct window receipts rejected, delayed >5 min or out of order; categories may overlap |
| Endpoint change | Freshness at baseline versus as-of; does not enumerate transient changes |
| Rule change | Compare `(vessel_id, rule)` at endpoints, not changing observation/flag IDs |
| Receipt increment | Received = accepted + rejected in `(baseline, as-of]` |

Current flags disappearing are labelled "No longer flagged" and carry the reason
that the rule no longer applies; they do not mean an incident was resolved. No
case status is inferred from clock time. District selection remains map focus
only. State membership is a fictional exercise assignment, not jurisdiction.

## Acceptance scenarios

- Odisha at minute 60, full history: SIM-004 has 20 stale minutes; SIM-006 has
  6 stale minutes despite being fresh now; SIM-008 changes from awaiting to fresh.
- Odisha at minute 95: SIM-004 is fresh, with 51 past stale minutes retained.
- Delayed SIM-006 observations never displace a newer last-known position.
- No-receipt/zero-duration windows have explicit empty states. Clock rewind cannot
  leak future arrivals or recovery. A receipt at baseline is not counted twice.
- Regular updates to an unchanged area rule are continuing flags, not new alerts.
- State/coast/class changes reset out-of-scope selection; national view is bounded
  and all 13 coastal state/UT cohorts remain available.
- JSON and text agree with source evidence; no original exporter changes.
- Desktop/mobile charts fit; keyboard controls, globe, original research panels,
  history, casebook and every existing browser assertion remain functional.

## Source interpretation

The [USCG AIS FAQ](https://www.navcen.uscg.gov/ais-frequently-asked-questions)
distinguishes reported position data from static/voyage information and calls
for data accuracy checks. It is a design reference, not an Indian feed or a
source of thresholds. The project's 15-minute freshness rule is demo policy.
Reviewed 16 October 2026; the fixture remains explicitly synthetic.

## Before and after

| Capability | PR20 | This increment |
| --- | --- | --- |
| Temporal review | Single-vessel history | Preserved; adds fleet continuity and endpoint comparison |
| Watch queue | Current rule flags | Preserved; adds grouped unknown, stale, changed and delivery-exception evidence |
| Handover | Case-specific text | Preserved; adds cohort briefing text and JSON with evidence |
| Navigation | Map -> history / investigation | Adds briefing -> same existing workflows |

Further work: approved observation adapters and source-health contracts;
licensed environmental context; authenticated persistent cases; independently
evaluated contextual events; capacity and resilience testing. These remain
separate increments, not implied by this exercise briefing.
