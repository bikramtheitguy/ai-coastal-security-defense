# Investigation and handover increment

## Decision and scope

Build an additive **Investigations** tab inside India Coastal Watch. The audience is a reviewer demonstrating how received observations become a documented assessment. The central question is: what did the analyst know, what changed, and why was a case kept open or closed? Preserve all six modules and the existing watch rules, geography, filters and source data.

Baseline: merged PR #15, commit `e4a383bf79858a38ea4574699fa31a46f57b6532`. Build on a separate branch and submit a reviewable PR; deployment remains a separate approved step. Lovable is an external hyperlink only.

## Implemented contract to build and verify

1. Open a case from a current flag. Keep one case per vessel per exercise; changing report IDs must not silently create duplicate cases. A case is a demonstration review folder, not a confirmed incident or an episode detector.
2. Freeze vessel evidence using receipts available at the selected exercise time. Later captures append to the case. Never rewrite earlier evidence when the map clock changes. Refuse backward or duplicate captures.
3. Display observation-to-receipt timing and a capture comparison: observation age, accepted/rejected reports, flag changes and reference IDs. Receipt lag and stale positions remain distinct.
4. Require a fictional reviewer label, rationale and acknowledgement for each capture/decision. State machine: New -> In review -> Closed — benign / Escalated — demo; either disposition can reopen to In review. Capturing new evidence preserves status and asks for review; no automatic closure or escalation.
5. Export the casebook as bounded JSON and a deterministic plain-text handover. Restore only after validating every event and independently reproducing every evidence snapshot against the bundled synthetic fixture. Preview first; import only into an empty casebook. Never overwrite or merge histories implicitly.
6. Keep records private to each Streamlit session and out of shared caches/files/databases. The downloadable casebook is the recovery mechanism. It is not durable authenticated storage; labels and timestamps are self-asserted on import, and hashes detect corruption rather than prove authorship.

## Measurement and UI

The existing four watch cards remain scoped to map filters and the exercise clock. Casebook counts are separately labelled across all saved vessels and clocks: total cases, awaiting review (New/In review), and locally escalated cases. Those are workflow counts, not threat severity or model precision. Map filters do not hide saved cases. Review the latest capture, with both exercise UTC/IST and actual recording UTC visible. Plain-text summaries separate observations from analyst statements.

Use the existing spatial theme, a bounded queue, one selected case, evidence comparison, a receipt timeline, explicit actions and exports. Empty, invalid import, duplicate and capacity states must be actionable. No extra runtime dependencies, network requests or commercial/provider data are needed.

## Acceptance scenario

At minute 60, open SIM-004 from its stale-report flag. Capture its 35-minute-old last observation. Start review with a rationale. At minute 95, capture received recovery evidence; retain both snapshots and the prior review history, and do not close automatically. Close as benign with a new rationale, export, and restore into a fresh session. The counts, evidence hashes, decision history and handover must round-trip. A changed coordinate, altered rule explanation, invalid transition, duplicate JSON key, oversized file or future receipt inserted into evidence must be rejected, including when an attacker recalculates the outer hash. An unrelated browser session must remain empty.

Validate with domain tests, Streamlit AppTest and Chromium desktop/narrow-layout tests, including the six existing views. Review screenshots and update the POC/comparison before handoff. Do not claim production readiness.

## Next increments after this

1. Authenticated durable case service: server-side authorization, database migrations, optimistic concurrency, retention and backup/restore tests.
2. Approved historical AIS replay with source coverage and licensing, followed by independent real-data detection evaluation.
3. Calibrated trajectory events with benign counterexamples and event-level false-alert measurement.
4. Evidence-citing AI assistance only after measuring this deterministic handover baseline.

## Runtime references

[Streamlit Session State](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state) is tied to a WebSocket session; browser reload can reset it. [File uploader](https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader) supplies uploaded bytes; this increment must validate the content itself and must not trust an extension or MIME type. The project never loads pickle from an upload.
