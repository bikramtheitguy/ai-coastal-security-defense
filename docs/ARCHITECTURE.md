# Architecture

## Current supported demo

Streamlit UI -> manifest/checksum verification -> local CSV or PNG -> validated labels and recomputed metrics -> rendered research results.

The three preserved research views follow that path. The fourth view follows: fixed-seed synthetic tracks -> schema validation -> cursor-bounded observations -> rules/freshness -> local charts and evidence -> session review -> optional user download.

The fifth view evaluates independently seeded synthetic voyages, compares detection methods and exports matching evidence. The sixth view follows: pinned public geographic fixtures + synthetic receipt stream -> integrity/contract validation -> clock-bounded acceptance or quarantine -> last-known vessel picture -> current rule evidence and snapshot export.

- `dashboard/app.py`: presentation and module selection.
- `dashboard/results.py`: file integrity, CSV contract, image decoding and pure metric calculations.
- `dashboard/artifacts.json`: preserved baseline asset identities.
- `dashboard/replay_view.py`: replay controls, charts, evidence and session review UI.
- `coastal_security/replay.py`: deterministic generation, UTC/coordinate contracts, transparent rules, review transitions and fixture export CLI.
- `data/replay/`: generated CSV and manifest; regeneration identity checked by tests.
- `coastal_security/trajectory.py` and `dashboard/evaluation_view.py`: synthetic trajectory evaluation and its presentation.
- `coastal_security/watch.py`: deterministic traffic, receipt causality, quarantine, freshness, scoped snapshots and evidence exports.
- `coastal_security/geography.py` and `india.py`: pinned-source integrity, bounded geometry processing, coast/state selection and display-only polygon orientation.
- `dashboard/watch_view.py`: India map, layers, cohort controls, vessel inspector and intake/source panels.
- `scripts/build_india_reference.py`, `build_coastal_watch.py` and `data/coastal_watch/`: reproducible public-geography derivation, synthetic generation and third-party notices.
- `data/`, `models/`, `notebooks/`: historical research inputs and artifacts; training is outside the supported runtime.
- No model deserialization, database, API, ingestion worker, identity service or external data feed is used.

The public demo serves synthetic operational artifacts and explicitly licensed public geographic references. It must not become a container for non-public operational data.

Replay uses local charts without external map tiles. Domain logic is separate from Streamlit session state. Review events are volatile and self-attributed, not an identity or audit service. See the [POC definitions and acceptance walkthrough](STREAMLIT_POC.md).

Coastal Watch's receipt validator is an offline exercise processor, not a live ingestion service. It retains observation time separately from receipt time, explains rejected records, and refuses mismatched fixture bytes. Display copies adapt polygon winding for D3 without rewriting source geometry. Historical district associations are context, not official current jurisdiction. Snapshot SHA-256 values verify content identity, not authorship or tamper-proof audit. See the [watch metric contract](COASTAL_WATCH_PLAN.md) and [platform concept sequence](PLATFORM_CONCEPTS.md).

## Current offline vessel benchmark

`coastal_security/vessel/data.py` generates and validates fictional snapshots and creates chronological, vessel-disjoint partitions. `model.py` fits the feature pipeline only on training data, chooses the model/threshold on validation data, and evaluates the frozen selection on test data. `run.py` writes a new experiment bundle with IDs, scores, metrics, pipeline, model card and provenance hashes. `__main__.py` supplies the CLI.

This is a separate offline path using the research dependency group. It never rewrites dashboard assets, fetches external data, or deserializes model files. The saved new pipeline supports reproducibility; no model registry, serving API, drift monitoring or operational promotion is implemented. See the [evaluation protocol](VESSEL_BENCHMARK.md).

## Proposed architecture, not implemented

Approved/synthetic source adapters -> validation/quarantine -> normalized observations and spatial storage -> domain analysis -> evidence-linked alerts -> human review -> recorded disposition.

Proposed internal boundaries:

| Boundary | Responsibilities |
| --- | --- |
| contracts and ingestion | Typed schemas, units, UTC event/ingestion times, source provenance, retries, deduplication, checkpoints |
| maritime and geospatial | Extend the implemented synthetic replay with approved layers, evaluated contextual rules, CRS/unit correctness and spatial storage |
| vision | Approved media, annotations, detector inference, tracking, uncertainty and scene-isolated evaluation |
| security events | Normalized approved logs, correlation, alert evaluation and analyst feedback |
| model lifecycle | Train-only preprocessing, untouched evaluation, data/split/code hashes, registry, promotion and rollback |
| incident review | Extend the session exercise with authenticated review, persistent evidence/status/rationale and protected audit history |
| AI assistance | Permission-aware evidence retrieval, cited draft summaries, abstention and mandatory human review |
| API and platform | Versioned contracts, OIDC, server-side authorization, migrations, telemetry and recovery |
| quality | Unit, integration, geospatial, authorization, model and UI tests |

Start with a modular Python application and independent training/background jobs. Add distributed infrastructure only when measured load justifies it. Public demos must work without provider credentials. No AI component should independently authorize operational actions.

## Separate ICCC proposal

PR #10 contains a separate unmerged FastAPI/Next.js ICCC proposal. It is preserved independently and is not a dependency or replacement for this Streamlit project. Earlier test results are recorded in the [comparison](STREAMLIT_POC_COMPARISON.md#pr-10-context-preserved-separately); they are not evidence that Streamlit implements those capabilities. Any future reuse needs explicit scope and independent review.

## Related Lovable prototype

The [coastal dashboard showcase](https://icccodishacshq.lovable.app/) is a separately maintained website. The only relationship to this Streamlit repository is this external hyperlink. It is not the application developed here.

Do not connect repositories, synchronise code, call its backend, share data/credentials, change its settings or deploy to it. Earlier hypothetical integration proposals are withdrawn from this project's architecture. The link does not establish operational approval, national-scale performance or a safe destination for real incident submissions.

Future data adapters and model services, if approved, belong to this independent Python/Streamlit architecture. `bkdigital.tech` is outside this project scope.

## Decisions for this baseline

- Keep historical assets and research code unchanged.
- Use an explicit Python 3.13 dependency lock with separate runtime/dev/research groups.
- Export hashed runtime requirements because Community Cloud consumes a requirements file.
- Verify asset bytes and labels before displaying metrics.
- Preserve undefined metric values instead of presenting unsupported zeroes.
- Keep model-quality defects visible; do not convert a passing UI test into a detection-quality claim.
