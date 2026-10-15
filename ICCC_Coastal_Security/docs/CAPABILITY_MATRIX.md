# Capability inventory

Reviewed source: PR #10 commit `f24bf50180e19ef8ccfafcb40ea5f8696485fa03`, 15 October 2026. **Source present** means implementation was inspected; it does not mean all behavior, security or deployment claims are validated. Current results belong in [VALIDATION.md](VALIDATION.md).

| Area | Existing implementation and source | Evidence and remaining boundary |
| --- | --- | --- |
| Vessel monitoring | Synthetic tracks, simulator, threshold rules and alert escalation in `backend/app/services/analytics.py`, `simulator.py` | No real AIS/radar ingestion; false-alert and miss rates on real traffic unknown |
| Geospatial picture | MapLibre UI, PostGIS view initialization, approximate coastline, reference layers and stale-state display | No authoritative ENC, navigational routing or national capacity test |
| Readiness | Personnel/asset computations and ranked resource suggestions in `services/readiness.py`, `recommend.py` | Heuristic weights; estimated straight-line travel does not account for navigability or sea conditions |
| Citizen assistance | Keyword/lexicon pipeline and templated responses in `services/chatbot/`; original message retained | Five-language draft lexicons; neither neural translation nor a general LLM; audio storage without transcription |
| Incident and command | C0–C8 states, orders, field acknowledgements, citizen updates, outcome and AAR | API/browser happy-path tests exist; jurisdiction and manual-dispatch gaps remain |
| Personnel and assets | Management, qualifications, maintenance, crew assignment, patrol planning and UAV mission console | Synthetic workflow features; UAV tasking is not image/video object detection or physical control |
| Intelligence | Watchlists, vessel history and need-to-know checks | Synthetic data; endpoint-level negative coverage must be completed |
| Cybersecurity monitoring | Failed login/session/audit events and exercise scenarios | Application event views, not live network IDS, SOC or SIEM integration |
| Identity and access | Local JWT plus server sessions, role matrix, jurisdiction helpers and optional TOTP | Object checks are inconsistent; MFA reenrollment issue is a blocker |
| Evidence and audit | Local uploads, hashes, custody records, hash-chained audit, JSON backup/restore | No external immutable anchor, certified custody, HA storage or validated DR capability |
| API and packaging | FastAPI API, static Next.js export, Compose PostgreSQL/PostGIS and local SQLite | Single-node POC; shared demo credentials and fallback secrets require containment |
| Tests | pytest API suites and Playwright workflows in `tests/` | Historical README counts are not current-commit evidence; new CI must be inspected |
| Drone research in repository root | Offline Random Forest on supplied labels/confidence | No image detector; no connection to ICCC UAV views |
| Cyber research in repository root | Offline classifier on dummy packet records | No live sensor; no connection to ICCC security events |
| Vessel research in repository root | Reproducible synthetic speed benchmark in `coastal_security/vessel` | Train/validation/test isolation exists; speed snapshots are not vessel trajectories or an operational detector |

## Proposed capabilities

Approved feed adapters, geospatial replay, validated CV inference, model serving, governed AI incident assistance, federated identity, cross-site coordination, model monitoring and resilient deployment remain proposed. The [target architecture](ARCHITECTURE.md) describes interfaces rather than claiming an integration exists.

## Feature preservation review

For each PR, compare the previous and proposed revision for routes, actions, roles, filters, exports, supported inputs and persisted records. Record unchanged features as well as intentional behavior changes. Test at least one successful authorized action and one denied unauthorized action for each access-control change. Never count an unconditional error or a hidden button as a completed security fix.

The Lovable application has its own feature inventory and source revision. Do not infer its coverage from this table or treat a change here as a change to that application.
