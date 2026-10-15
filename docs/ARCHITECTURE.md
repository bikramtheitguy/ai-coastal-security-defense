# Architecture

## Current supported demo

Streamlit UI -> manifest/checksum verification -> local CSV or PNG -> validated labels and recomputed metrics -> rendered research results.

- `dashboard/app.py`: presentation and module selection.
- `dashboard/results.py`: file integrity, CSV contract, image decoding and pure metric calculations.
- `dashboard/artifacts.json`: preserved baseline asset identities.
- `data/`, `models/`, `notebooks/`: historical research inputs and artifacts; training is outside the supported runtime.
- No model deserialization, database, API, ingestion worker, identity service or external data feed is used.

The public demo serves synthetic artifacts only. It must not become a container for non-public operational data.

## Current offline vessel benchmark

`coastal_security/vessel/data.py` generates and validates fictional snapshots and creates chronological, vessel-disjoint partitions. `model.py` fits the feature pipeline only on training data, chooses the model/threshold on validation data, and evaluates the frozen selection on test data. `run.py` writes a new experiment bundle with IDs, scores, metrics, pipeline, model card and provenance hashes. `__main__.py` supplies the CLI.

This is a separate offline path using the research dependency group. It never rewrites dashboard assets, fetches external data, or deserializes model files. The saved new pipeline supports reproducibility; no model registry, serving API, drift monitoring or operational promotion is implemented. See the [evaluation protocol](VESSEL_BENCHMARK.md).

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

## Related Lovable prototype

The [command-centre dashboard](https://icccodishacshq.lovable.app/) is a separate Lovable application with a React/TanStack interface, Supabase-backed workflows and MapLibre maps. Its intended hierarchy spans national, state, district and station views. This is distinct from both the Streamlit artifact viewer and the unmerged PR #10 proposal.

There is currently no integration between that application and this repository's Python models. The link is provided for prototype review, not as evidence of operational approval, national-scale performance or a safe destination for real incident submissions.

A proposed integration would send versioned, validated analysis results through an authenticated API into a human review queue. Each result should carry a source identifier, observation time, model/version, score semantics, limitations and evidence references. The UI should distinguish simulated, approved live and stale data. Model outputs must not independently verify reports or authorise actions.

Before integrating, create a separate read-only demonstration with synthetic fixtures; verify server and database permissions for every role and jurisdiction; define privacy and retention boundaries; and demonstrate load, recovery and audit requirements. Production deployment and approved maritime feeds require separate decisions and validation.

## Decisions for this baseline

- Keep historical assets and research code unchanged.
- Use an explicit Python 3.13 dependency lock with separate runtime/dev/research groups.
- Export hashed runtime requirements because Community Cloud consumes a requirements file.
- Verify asset bytes and labels before displaying metrics.
- Preserve undefined metric values instead of presenting unsupported zeroes.
- Keep model-quality defects visible; do not convert a passing UI test into a detection-quality claim.
