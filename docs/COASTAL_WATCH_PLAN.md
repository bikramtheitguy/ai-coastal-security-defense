# Coastal Watch: implementation plan

## Architectural decision

Build a sixth, additive Streamlit view for an analyst conducting a fictional coastal watch. The main questions are: which vessels have current observations, which positions are stale or unknown, what evidence supports a flag, and which incoming records cannot be trusted? Keep all five existing views.

The current project has reproducible experiments, synthetic replay and evaluation, but no geographic context or receipt-time-aware ingestion. A more credible map requires trustworthy observation handling before persistent case management or AI summaries. This increment uses public physical coastline context and a separate invented traffic exercise; it does not represent official surveillance coverage.

## Implement now

1. Bundle a bounded Natural Earth 1:50 million coastline fixture. Pin the upstream revision; record licence, version, source and derived hashes, coordinate order and transformation. Render locally without tile requests or credentials.
2. Generate 44 fictional vessels across 13 Indian coastal states/UTs, including the original eight-vessel Odisha exercise. Use plausible position increments and separate observation/receipt timestamps. Include outages, late/out-of-order delivery, duplicates and invalid coordinates. Use deterministic generation and explicit synthetic identifiers.
3. Validate incoming receipts, retain rejected records with reasons, deduplicate only accepted observation identities, and evaluate the visible state using receipts available at the chosen exercise clock. A late report must not move the latest position backwards.
4. Add a spatial watch view with a fixed geographic extent, public coastline, selectable synthetic observation trails, last-known positions, an invented exercise area, freshness metrics and a vessel inspector. Do not interpolate missing positions or draw continuous tracks across outages.
5. Explain active stale-report and exercise-area flags with evidence IDs. Export the current evidence snapshot and registry with complete provenance. This is a review aid, not hostile-intent inference or an incident dispatch system.
6. Validate arithmetic, temporal causality, rejection handling, boundary cases, reproducibility and asset integrity. Run real Streamlit/Chromium desktop and narrow-layout checks in CI because local socket creation is restricted. Inspect screenshots before handoff.

## Metric contract

- Scope: selected coast, exercise state/UT, vessel class and exercise clock; all surface counts reconcile to this cohort. District focus changes map extent only and is labelled accordingly. Synthetic state assignment is not jurisdiction.
- Observed vessels: catalogue vessels with at least one accepted report received by the clock.
- Stale positions: latest observation age exceeds 15 minutes, regardless of receipt freshness.
- Unknown: catalogue vessels with no accepted report; not represented as zero activity or map positions.
- Review flags: one current stale-report flag per stale vessel and one current exercise-area flag per fresh in-area vessel. Historical entry/exit or intent are not inferred.
- Receipt ledger: received = accepted + rejected. Out-of-order and delayed are explanatory properties of accepted reports, not extra receipt counts.

## Delivery and acceptance

Isolated source snapshot based on main485357e, additive branch and reviewable pull request. No main update or deployment is implied by this build request. UI/defaults, filters, layers, no-report state, source metadata and JSON export must work. All existing tests remain passing. Include a previous/current capability comparison, new screenshots and recorded validation results.

## Subsequent increments (proposed)

1. Authenticated incident workspace with durable storage, server-side permissions, review history, retention and recovery tests.
2. Approved historical AIS adapter with schema validation, jurisdiction/coverage metadata and a separate real-data evaluation suite.
3. Evidence-citing incident summaries with human review and abstention; assess whether an LLM improves utility over deterministic summaries before adding it.
4. Independent image/video validation and authorized security-event adapters; explicit capacity and failure targets before any national-scale claim.

Lovable remains a hyperlink only. Client maps and operational locations are excluded from these public fixtures.

## India and platform-reference refinements

Add dated public state/UT and district outlines, coast/state cohort selection, district map focus, explicit island reference gaps and IST display alongside canonical UTC timestamps. Maintain third-party licences and a reproducible geographic derivation. Mixed-vintage district associations are not a current official LGD crosswalk.

See [platform concepts](PLATFORM_CONCEPTS.md) for the seven requested references and the separation of implemented work from subsequent aircraft, incident, hazard and AI capabilities. Drive sources are reviewed read-only and remain outside this repository; a metadata/content assessment and notebook are saved separately in the local workspace.
