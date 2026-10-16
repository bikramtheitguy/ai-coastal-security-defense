# Investigation and handover increment

## Decision and scope

Build an additive **Investigations** tab inside India Coastal Watch. The audience is a reviewer demonstrating how received observations become a documented assessment. The central question is: what did the analyst know, what changed, and why was a case kept open or closed? Preserve all six modules and the existing watch rules, geography, filters and source data.

Baseline: merged PR #15, commit `e4a383bf79858a38ea4574699fa31a46f57b6532`. Build on a separate branch and submit a reviewable PR; deployment remains a separate approved step. Lovable is an external hyperlink only.

## India-first scope

India is the primary product scope. Odisha and the adjoining Bay of Bengal are the reference scenario for development and acceptance, with extension across Indian coastal states, union territories and islands. This increment preserves the existing geographic selection and startup defaults; it does not silently change the supported baseline. The SIM-004 reporting-gap investigation uses the existing Odisha exercise.

Future map-workspace work should make Odisha the initial area of interest and retain explicit nationwide and island selection. Use Indian administrative context with source vintage, UTC/IST timestamps, kilometres and nautical miles where relevant. Broader global context is secondary, not a requirement for the next Indian demonstration. Never infer authoritative jurisdictions, operational inventories or sensor coverage from synthetic fixtures.

The expanded maritime-platform brief is a roadmap input. Its map, source adapters, aviation, sensor correlation, persistent authenticated cases, AI and forecasting requirements are not claims about this increment. Research, access verification and separate acceptance checks remain necessary for each phase.

## Implementation contract

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

1. India map workspace: default Odisha/Bay of Bengal area, vessel search and selection, saved non-sensitive areas, map-visible evidence and synchronised playback. Review reference-platform evidence and interaction requirements before choosing a map component; preserve the current offline reference view.
2. Indian observation and source contracts: explicit observed/received/processed times, provider rights and geographic coverage, bounded adapters and degraded states. Add an approved historical AIS source only after access and reuse verification; otherwise keep a clearly labelled fixture. Evaluate trajectory events using benign counterexamples and event-level false-alert measurement.
3. Authenticated durable case service: server-side authorization, database migrations, optimistic concurrency, retention and backup/restore tests. Plan migration from the current versioned portable casebook without silently rewriting its history or implying authentication of old labels.

Subsequent phases can add synthetic civilian aviation, environmental context, sensor-association contracts, evidence-citing AI and evaluated forecasts. Each needs its own provenance, uncertainty and acceptance criteria. The current deterministic handover supplies a baseline for later AI evaluation.

## Runtime references

[Streamlit Session State](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state) is tied to a WebSocket session; browser reload can reset it. [File uploader](https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader) supplies uploaded bytes; this increment must validate the content itself and must not trust an extension or MIME type. The project never loads pickle from an upload.
