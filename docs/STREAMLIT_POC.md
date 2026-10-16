# Streamlit coastal analytics POC

This POC combines India-focused geographic watch, reproducible vessel replay, human evidence review and an independent trajectory evaluation lab in Python/Streamlit. Geography comes from dated public sources; traffic and operational scenarios are invented. It is a software demonstration, not an operational coastal-security system.

## Current proposal: vessel history and reporting continuity

Add a Vessel history workspace inside Coastal Watch, using the existing Indian
synthetic exercise and public geography. It explains gaps, recovery, delayed
receipts and windowed sampled-link distance without inventing travel through
missing observations. The [implementation plan and comparison](VESSEL_HISTORY.md)
define the metric contract, Odisha walkthrough and validation requirements.
This increment is proposed on a separate branch; hosted deployment requires
merge and separate live verification.

## Merged baseline: globe and graphics-first layouts

India Coastal Watch becomes the default workspace, with a rotatable globe, India/Odisha
camera presets, selectable synthetic vessels, receipt-aware history and an investigation
handoff. The existing 2D map and all six modules remain available. Drone and cyber
experiments gain interactive review panels while preserving their metrics, source files
and original charts. See the [globe comparison, architecture and walkthrough](GLOBE_WORKSPACE.md).
PR19 merged at `4c4de199d53450e7c454ef720a0699cef3e7967f` on 16 October 2026.
Its five graphics-first pages and existing exports were verified on the hosted
app at desktop and narrow widths after an authorized restart. That hosted check
does not establish deployment of the new vessel-history proposal.

## Investigation and handover baseline: PR #16

Coastal Watch includes a session-local casebook with frozen evidence captures, human review transitions, observation-to-receipt timing, a deterministic handover brief, and validated JSON recovery. PR #16 was merged at `131fcaa052525b1935e1aa49512206366a7b3e32`; its Odisha workflow and validated download/restore were verified on the hosted app on 16 October 2026. See the [plan](INVESTIGATIONS_PLAN.md), [side-by-side capability comparison and demo](INVESTIGATIONS_COMPARISON.md), and [validation record](INVESTIGATIONS_VALIDATION.md).

## Geographic watch baseline: PR #15

The sixth view, **Coastal Watch**, adds 13 coastal states/UTs, 229 historical reference district polygons, and 44 fictional vessels. Coast and state/UT filters scope vessel counts; district selection focuses the map only. The watch preserves observation and receipt times, shows stale and unknown positions, quarantines invalid receipts, and exports evidence with reproducibility hashes.

Use **All coasts** for the overview, **Odisha** for the original eight-vessel exercise, and **Island territories** for the island cohorts. At minute 60 the all-coast picture has 44 observed vessels, one stale position, three current review flags and four rejected receipts. Move the clock to 150 to inspect stale last-known positions. Flags are illustrative review candidates, not evidence of unlawful activity.

The public fixtures are pinned and checked before rendering, require no credentials or external map tiles, and carry their own data licences. Mixed boundary vintages and the missing Lakshadweep district association remain explicit. The app does not infer offshore jurisdiction from a civil district polygon.

See the [implementation plan and metric contract](COASTAL_WATCH_PLAN.md), [previous/current comparison and demonstration](COASTAL_WATCH_COMPARISON.md), [public-source notices](../data/coastal_watch/NOTICE.md), and [seven-platform concept roadmap](PLATFORM_CONCEPTS.md). PR #15 was merged at `e4a383bf79858a38ea4574699fa31a46f57b6532` and its hosted functionality was verified on 16 October 2026.

The [Coastal Watch validation record](COASTAL_WATCH_VALIDATION.md) records passing cross-platform and real-browser checks. It is separate from hosted deployment verification.

## Preserved evaluation and spatial UI increment

The preceding five-view application includes **Vessel Evaluation**, with 48 calibration and 48 held-out test voyages across six families, including benign manoeuvres and interruptions. It compares calibrated rules, fixed replay thresholds and no-alert baseline; displays precision, recall, misses and false alerts per vessel-hour; exposes matching ledgers; and exports both complete synthetic splits. See [protocol and reproducibility](TRAJECTORY_EVALUATION.md).

The spatial theme adds layered navy surfaces, rounded panels and clear hierarchy without external assets. Existing replay controls and all historical experiments remain available. Coastal Watch now supplies geographic context; persistent authenticated operations remain proposed in the [design and data roadmap](DESIGN_AND_DATA_ROADMAP.md).

PR #13 was approved and merged on 15 October 2026 at `4a1630a25d533a415948035563608db783fe00e2`. PR #14 subsequently added evaluation and the spatial theme; this increment starts from main `485357e97feecc9ed39a72ceb88009197af41910`. Historical validation records describe those revisions. The separate 16 October hosted verification confirmed Coastal Watch; it does not establish deployment of subsequent proposals. The sections below preserve the replay specification.

The delivered application remains `dashboard/app.py`. The three historical research views and offline vessel benchmark are preserved. Lovable is an **external showcase hyperlink only**: no code changes, backend calls, shared data, synchronisation or deployment connection. The separate unmerged ICCC proposal in PR #10 is retained and is not a dependency or replacement application.

## Demonstration brief

- Audience: project reviewers, developers and researchers assessing a reproducible monitoring workflow.
- Question: what observations were available at a chosen time, which simple rules triggered, and how can a reviewer inspect and disposition them?
- Scope: four fictional vessels, 91 observations, 24 five-minute time slots on 1 January 2026, 00:00–01:55 UTC.
- Default: replay paused at 01:00 with all vessels selected, showing four vessels, zero stale last positions and three cumulative flags.
- Data: `coastal_security/replay.py`, fixed seed `20261015`; committed CSV and manifest in `data/replay/`. No external datasets, credentials, APIs, map tiles or model files are used by the new view.

## Implemented behavior

Select **Scenario Replay** in the existing module selector. Play/pause, step five minutes, seek to a replay time, and filter vessels. Playback advances one step per second while the view is active; it does not simulate background ingestion or catch up after a disconnected browser. The replay stops at 01:55. The seek control is a destination selector; the displayed replay clock is authoritative during playback.

Positions appear on a longitude/latitude scatter plot, with axes fitted to the visible observations and tooltips identifying each observation. Axis ranges change with replay time and vessel selection; the plot does not preserve geographic aspect ratio. A second chart shows reported speed over time. The table identifies each last observation and its age relative to the replay clock. These plots have no navigation basemap. Speed lines may span missing reports; they are not evidence of continuous observations.

| Measure | Definition | Interpretation |
| --- | --- | --- |
| Vessels with reports | Distinct selected vessels with a report at or before the cursor | Observed coverage in this fictional scenario |
| Stale last positions | Selected vessels whose last report is more than 15 minutes old at the cursor | Data freshness, not threat severity |
| Rule flags to inspect | Unique rule events for selected vessels at or before the cursor, including historical gaps | Cumulative events; not unresolved incident count or detection accuracy |

Rule calculations use only observations at or before the cursor. Changing the vessel filter changes the plots, table, counts and available flag evidence together. With no vessels selected, an explicit empty state replaces the charts. Full-scenario downloads deliberately contain future replay times and are labelled accordingly.

## Transparent rule specification

| Rule | Trigger | Evidence and limits |
| --- | --- | --- |
| Reported speed change | Absolute difference between consecutive reported speeds ≥8 kn | Two observation IDs. Not acceleration and not calibrated for vessel type |
| Position jump | Great-circle distance / elapsed hours >60 kn | Two observation IDs. Spherical distance in nautical miles; check sensor/data quality first |
| Reporting gap | More than 15 minutes since a report without another observation | Last observation ID; stable episode ID persists after reporting resumes. First whole second past the threshold is recorded |

The thresholds are illustrative and fixed. They have not been selected using representative maritime data. A gap, speed change or position jump does not establish hostile intent. There are no confidence scores, real-world false-alert estimates or ML performance claims for these rules.

The scenario injects one sustained reported speed change in DEMO-002, one corrupted position in DEMO-003 (which creates an outward and return jump), and five omitted reports in DEMO-004. DEMO-001 is an ordinary track. At the end there are four rule events. These designed examples are regression fixtures, not an independent evaluation set.

## Human review exercise

Pause playback, select a flag, inspect its explanation and source rows, enter a fictional reviewer label and rationale, and acknowledge reviewing the evidence. The allowed sequence is **New → In review → Dismissed / Escalated (demo)**; either closed status can return to In review. A rationale and evidence acknowledgement are required for every transition.

The session keeps an append-only sequence of review events with previous/current status, label, rationale, replay time and actual UTC recording time. JSON export includes the scenario identity and CSV hash. Seeking backwards hides future flags but intentionally retains session review history; the UI labels that distinction. Reset clears replay and reviews, and losing the Streamlit session can also lose them.

This is **not authentication, authorisation, persistent incident management or a tamper-proof audit log**. Reviewer labels are self-asserted. Escalation changes a local demonstration status only. No dispatch, notification or enforcement occurs. Use fictional text only.

## Reproduce and demonstrate

```bash
uv sync --locked --no-dev
uv run --locked --no-dev streamlit run dashboard/app.py
```

Choose Scenario Replay. For a repeatable walkthrough:

1. Seek to minute 0: four vessels, no stale positions, no flags.
2. Seek to minute 40: the DEMO-002 speed change appears; the gap is exactly 15 minutes and is not yet stale.
3. Seek to minute 45: DEMO-004 is stale; two cumulative flags are visible.
4. Seek to minute 55: DEMO-004 has resumed reporting; the historical gap flag remains.
5. Seek to minute 60: inspect DEMO-003's position jump. Record an In review status with a synthetic rationale, then dismiss or escalate the demo case.
6. Play to minute 115 and pause: four cumulative events. Export review history and the fixture manifest.
7. Filter to DEMO-001: no flags. Clear the selection to exercise the empty state. Reset to clear reviews.
8. Return to each historical module and confirm its original chart and metrics remain available.

Regenerate the fixture pack into a **new** directory:

```bash
uv run --locked --no-dev python -m coastal_security.replay --output artifacts/replay-check
```

The generator refuses to overwrite an existing directory. Tests compare regenerated CSV bytes and manifest values to the committed pack. No application runtime dependencies were added; the existing locked runtime supports this view. The separate [browser QA environment](BROWSER_QA.md) uses hash-locked Playwright and pytest tooling.

## Architecture and increment process

`generate_scenario → validate_observations → cursor-bounded observations → rules / freshness → Streamlit plots and evidence → session review → optional user download`

Domain calculations live in `coastal_security/replay.py`; presentation and session state live in `dashboard/replay_view.py`. Historical artifact verification remains in `dashboard/results.py`; offline training remains in `coastal_security/vessel/`. Each path has separate claims and validation.

For every future increment: define a small acceptance scenario, record data provenance, implement domain logic separately from UI, add meaningful failure/boundary tests, exercise Streamlit, update this POC and the comparison, run CI, and review before merging/deploying. Preserve existing views and dataset history. Use synthetic data until approved representative sources and redistribution terms are available.

## Prioritised next work and readiness gaps

| Priority | Increment | Acceptance gate |
| --- | --- | --- |
| P0 | Review and release this Streamlit replay increment | Windows/Linux CI, real-browser checks and screenshot review; maintainer approval before merge, then a separately verified hosted revision |
| P1 | Independent trajectory evaluation suite | Separate scenario families, grouped temporal holdouts, rule calibration, false-alert rate by vessel-hour, missed-event counts; no tuning on final evaluation |
| P1 | Drone and cyber reproducible pipelines | Documented synthetic generators, train-only preprocessing, saved split IDs/scores, baselines and class-sensitive metrics; genuine imagery needed before claiming computer vision |
| P2 | Approved public maritime adapters | Explicit licence/approval, source and ingestion timestamps, UTC/unit/schema validation, deduplication, quarantine and reproducible recorded fixtures |
| P2 | Richer Streamlit geospatial context | Approved layers, coordinate-system and distance tests, missing/stale data visualization; validate target hardware and network requirements |
| P2 | Persistent review service behind Streamlit | Authenticated identities, server-side scopes, durable audited transitions, privacy/retention, backup/restore and negative authorization tests |
| P3 | AI-assisted summaries | Approved evidence only, citations and abstention, prompt-injection testing and mandatory human review; never independently authorise actions |
| P3 | Multi-jurisdiction scale assessment | Workload and latency targets, tenancy, monitoring, recovery exercises, capacity tests and independent security review |

The replay P0 was merged and its hosted UI checked. The independent P1 synthetic event evaluation is implemented in this proposal; real-data and unseen-family validation remain outstanding. Other P1–P3 capabilities remain future work. Current legacy model limitations remain unchanged. National-scale deployment and operational readiness are unverified.

See [before/after comparison](STREAMLIT_POC_COMPARISON.md), [validation record](STREAMLIT_VALIDATION.md), [architecture](ARCHITECTURE.md) and [installation](INSTALLATION.md).
