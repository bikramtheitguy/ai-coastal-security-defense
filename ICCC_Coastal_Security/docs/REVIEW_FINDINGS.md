# POC technical review

The ICCC implementation is a substantial synthetic workflow prototype. It should remain under review until authorization, MFA and validation gaps close. This assessment is based on PR #10 source commit `f24bf50180e19ef8ccfafcb40ea5f8696485fa03`, inspected on 15 October 2026. It is not a penetration test of a deployed service.

## Priority findings

| ID | Priority and finding | Evidence | Required acceptance |
| --- | --- | --- | --- |
| SEC-01 | P1: object/jurisdiction checks are missing on incident and conversation paths | `routers/incidents.py:get_incident` returns a loaded record after a capability check; `routers/chat.py` conversation handlers similarly lack object scoping. A synthetic function-body probe returned an unrelated-station incident. | Centralize scope checks; test list/detail/update/evidence/chat with same and different station/district users, including missing scope |
| SEC-02 | P1: order cancellation lacks issuer/jurisdiction validation | `routers/command.py` admits CANCELLED using TASK_ASSETS/ORDERS_ISSUE alone. A synthetic probe admitted an unrelated-station cancellation with persistence stubbed. | Define delegated cancellation policy and enforce it on the target order; deny unrelated users without state change |
| SEC-03 | P1: MFA reenrollment disables an existing factor immediately | `routers/auth.py:mfa_enroll` replaces the secret and commits `mfa_enabled=False`; reproduced with a synthetic user and unchanged function body | Reauthenticate, retain the active factor until a pending replacement is confirmed; test interrupted and unauthorized reenrollment |
| WF-01 | P1: manual dispatch can contradict field-confirmation claims | `routers/incidents.py:do_transition` allows C5 when an asset is assigned; `services/orders.py` describes field EN_ROUTE as the confirmation gate | Define manual confirmation with attributable evidence or require the field transition; test that a SENT order alone cannot trigger a dispatch promise |
| SEC-04 | P1: demo exposure and known fallback secrets | Configuration and Compose include public fallback signing/database/demo values; original Compose bound all interfaces | Local-only defaults, generated secrets, explicit demo mode and fail-closed non-demo startup; keep legitimate demo functionality |
| SEC-05 | P2: OTP attempts and token time need tests | Invalid OTP branch increments failures without the password branch's lockout handling; JWT code uses naive UTC datetime timestamps | Verify lockout for OTP and token lifetime across host timezones, with regression tests |
| OPS-01 | P2: audit/restore limits | Local hash chain, local evidence and JSON restore; no independent audit anchor or demonstrated DR | Test restore integrity and concurrent appends; state that full-chain rewrite or suffix removal needs an external anchor to detect |
| QA-01 | P1: current-commit validation absent at initial review | Existing tests and historical pass counts, but no Actions run for the reviewed head | Execute isolated backend, frontend and browser CI; link exact runs; add missing negative cases |
| DEP-01 | P1: dependency audit failed | CI reported affected PyJWT, python-multipart and Starlette versions, plus npm advisories including critical MapLibre; see [dependency findings](DEPENDENCY_REVIEW.md) | Compatible patched upgrades, regenerated locks and complete regression/audit runs; no suppressed checks |

SEC-01 to SEC-03 probes extracted unchanged function bodies and used synthetic doubles. They establish source-logic defects, not HTTP/database exploit results. WF-01 is a source-path finding; its integration regression test remains to be written. No operational records were accessed.

## Methodology and false alerts

The ICCC vessel engine applies heuristic rules; readiness and resource ranking use configured weights. These scores are not probabilities. No measured operational precision, recall, false-alert rate or response-time improvement is established. Evaluate per-detector and per-scenario errors, including missed detections and alerts per vessel-hour, before proposing any operational threshold.

The repository's separate synthetic vessel benchmark has held-out evaluation and traceable artifacts. Its speed-only snapshot results cannot validate ICCC track rules. The drone research consumes supplied object labels/confidence rather than images, while the cyber research consumes dummy packet records. Neither is integrated into this application. Future validation must split by time and entity, prevent preprocessing/threshold leakage, freeze the test set, report baselines and class-specific errors, and require analyst review.

## Provenance and attribution

The synthetic geography and station examples are illustrative. Named agencies, public-source reference counts and emergency text in older documents are not evidence of endorsement or current official integration. Maintain a source register with owner, URL/version, license, approval, retrieval date, coordinate accuracy, retention and redistribution terms before importing data. Review OSM/OpenSeaMap attribution and service-use terms before hosting a showcase; public tiles are not navigation charts.

The root license does not automatically license third-party maps, imagery, models or datasets. Dependency license inventory, image provenance, native-language review and approved source agreements remain open. Do not add restricted records to make a demonstration appear realistic.

## Production readiness

Identity federation, consistently enforced object authorization, secret management, validated models, approved feeds, geospatial indexes/query limits, queue/backpressure handling, service health metrics, load testing, immutable audit retention, backup/restore drills and multi-site recovery remain gaps. A passing demo suite establishes behavior for its fixtures only. See [ROADMAP.md](ROADMAP.md) for the sequence of work and [ARCHITECTURE.md](ARCHITECTURE.md) for proposed boundaries.
