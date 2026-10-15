# Architecture

## Current supported demo

Streamlit UI -> manifest/checksum verification -> local CSV or PNG -> validated labels and recomputed metrics -> rendered research results.

- `dashboard/app.py`: presentation and module selection.
- `dashboard/results.py`: file integrity, CSV contract, image decoding and pure metric calculations.
- `dashboard/artifacts.json`: preserved baseline asset identities.
- `data/`, `models/`, `notebooks/`: historical research inputs and artifacts; training is outside the supported runtime.
- No model deserialization, database, API, ingestion worker, identity service or external data feed is used.

The public demo serves synthetic artifacts only. It must not become a container for non-public operational data.

## Proposed architecture, not implemented

Approved/synthetic source adapters -> validation/quarantine -> normalized observations and spatial storage -> domain analysis -> evidence-linked alerts -> human review -> recorded disposition.

Proposed internal boundaries:

| Boundary | Responsibilities |
| --- | --- |
| contracts and ingestion | Typed schemas, units, UTC event/ingestion times, source provenance, retries, deduplication, checkpoints |
| maritime and geospatial | Timestamped tracks, contextual rules, anomaly evaluation, CRS/unit correctness, PostGIS queries and replay |
| vision | Approved media, annotations, detector inference, tracking, uncertainty and scene-isolated evaluation |
| security events | Normalized approved logs, correlation, alert evaluation and analyst feedback |
| model lifecycle | Train-only preprocessing, untouched evaluation, data/split/code hashes, registry, promotion and rollback |
| incident review | Evidence, status transitions, reviewer rationale and audit history |
| AI assistance | Permission-aware evidence retrieval, cited draft summaries, abstention and mandatory human review |
| API and platform | Versioned contracts, OIDC, server-side authorization, migrations, telemetry and recovery |
| quality | Unit, integration, geospatial, authorization, model and UI tests |

Start with a modular Python application and independent training/background jobs. Add distributed infrastructure only when measured load justifies it. Public demos must work without provider credentials. No AI component should independently authorize operational actions.

## Separate ICCC proposal

PR #10 contains unmerged FastAPI/Next.js/PostGIS scaffolding and tests. It needs an independent security and runtime review before selective adoption. Its claims and test files are not evidence that this Streamlit baseline implements those capabilities.

## Decisions for this baseline

- Keep historical assets and research code unchanged.
- Use an explicit Python 3.13 dependency lock with separate runtime/dev/research groups.
- Export hashed runtime requirements because Community Cloud consumes a requirements file.
- Verify asset bytes and labels before displaying metrics.
- Preserve undefined metric values instead of presenting unsupported zeroes.
- Keep model-quality defects visible; do not convert a passing UI test into a detection-quality claim.
