# Coastal command centre proof of concept

A synthetic workflow prototype for reviewing coastal reports, resource readiness, tasking, field acknowledgements and evidence. It uses FastAPI, a static Next.js interface and SQLite or PostgreSQL/PostGIS.

**Status: under review. Use disposable local synthetic data only.** The source review found authorization and MFA defects. Existing tests do not establish operational security, official endorsement or national-scale readiness. Start with the [POC charter](docs/POC_CHARTER.md), [capability inventory](docs/CAPABILITY_MATRIX.md) and [review findings](docs/REVIEW_FINDINGS.md).

## One demonstration flow

Synthetic report → human verification → explained resource recommendation → authorized order → field acknowledgement → resolution → evidence → reviewed after-action report.

The code contains incident and command workflows, readiness calculations, personnel/assets, a map, intelligence examples, audit events, administration and multilingual keyword-based assistance. These features are retained. Maritime feeds and operational entities are simulated; the chatbot is not a general LLM or validated translation system. The [acceptance script](docs/VALIDATION.md) distinguishes intended behavior from remaining defects.

## Run locally

Follow [installation](docs/INSTALLATION.md) for generated demo secrets, loopback access, Docker or local Python/Node setup. Do not use real personnel, coordinates, reports or credentials. Browser geolocation and real report submissions are unnecessary for the synthetic exercise.

From this directory after configuring a fresh demo environment:

```bash
docker compose up --build
```

Open http://127.0.0.1:8000. The application seeds synthetic accounts on an empty database; use your configured demo password. Stop with `docker compose down`; existing volumes remain. Never reset a database containing work you need to retain.

## Related projects

The repository root contains the [Streamlit research demo and reproducible vessel benchmark](../README.md). Those models are not connected to this application's rule engines.

The separate [Lovable coastal dashboard](https://icccodishacshq.lovable.app/) provides interface and workflow concepts. Its source, runtime and database remain separate. See the [concept mapping](../docs/LOVABLE_CONCEPT_MAPPING.md). Hosted reporting forms must not receive real or test incidents as part of this local demonstration.

The new [synthetic incident exchange contract](../docs/INCIDENT_EXCHANGE.md) validates invented jurisdiction membership, provenance timestamps and coordinate quality while preserving a citizen-selected destination. It does not write to either application or authorize tasking.

## Documentation

| Start here | Detail |
| --- | --- |
| [POC charter](docs/POC_CHARTER.md) | Scope, boundaries and evidence gates |
| [Capability inventory](docs/CAPABILITY_MATRIX.md) | Source present, partial and proposed functionality |
| [Architecture](docs/ARCHITECTURE.md) | Current components and future interfaces |
| [Installation](docs/INSTALLATION.md) | Isolated setup and safe test configuration |
| [Validation](docs/VALIDATION.md) | Repeatable scenario and exact evidence requirements |
| [Review findings](docs/REVIEW_FINDINGS.md) | Security, ML, provenance and readiness gaps |
| [Development process](docs/DEVELOPMENT.md) · [Backlog](docs/ROADMAP.md) | How each change keeps code, tests and documents aligned |
| [Incident workflow](docs/INCIDENT_WORKFLOW.md) · [Command workflow](docs/COMMAND_WORKFLOW.md) | State transitions and human decisions |
| [Chatbot](docs/CHATBOT_DESIGN.md) · [Limitations](docs/KNOWN_LIMITATIONS.md) | Language behavior and unsupported claims |
| [API](docs/API.md) · [RBAC](docs/RBAC_MATRIX.md) · [Data dictionary](docs/DATA_DICTIONARY.md) · [Tables](docs/DATA_DICTIONARY_TABLES.md) | Code-derived reference and domain semantics |
| [Synthetic dataset](docs/SYNTHETIC_DATASET.md) · [Test scenarios](docs/TEST_SCENARIOS.md) | Fixture generation and existing exercises |
| [Cybersecurity](docs/CYBERSECURITY.md) | Controls and outstanding hardening |

## Validation and attribution

The previous README reported September test runs. Those historical counts do not validate the current revision. Use the exact-commit CI results and [validation record](docs/VALIDATION.md); do not convert a source review or successful build into a security certification.

The [repository license](../LICENSE) governs repository-owned material. Map data attribution belongs to OpenStreetMap contributors and OpenSeaMap; third-party usage rights must be reviewed separately. Maps are not for navigation. Official programme references, emergency text and translated replies require current authoritative and native-speaker review before public use.
