# Test Scenario Catalogue

## 1. Exercise scenarios (inject from Analytics & System Health › Exercise Scenarios, or `POST /api/scenarios/{KEY}`)
Requires `SCENARIO_RUN` (ADGP, ICCC Supervisor, System Administrator). Everything injected is flagged EXERCISE / SIMULATED and audited.
All 16 keys are exercised by `tests/backend/test_platform.py::test_all_scenarios_inject`.

| # | Key | What happens | What to observe |
|---|---|---|---|
| 1 | `AIS_LOST` | A registered trawler stops transmitting AIS 35 min ago; radar keeps tracking | AIS-lost + radar-without-AIS alerts; fusion observation with contradiction "radar active while AIS silent" |
| 2 | `DARK_VESSEL` | Unidentified radar contact 7.5 NM off a vulnerable landing point, inbound 9 kn | Dark-vessel alert, hollow red contact on COP › Dark / Unidentified Targets |
| 3 | `RESTRICTED_ZONE` | A motorised boat moves inside a simulated restricted zone | Restricted-zone entry alert (HIGH) |
| 4 | `LOITERING` | 36 min of back-filled track within 0.3 NM | Loitering alert with rule parameters |
| 5 | `RENDEZVOUS` | Unidentified craft stops 0.12 NM from a trawler offshore | Paired rendezvous alerts; vessel risk rises |
| 6 | `UAV_CONFIRMATION` | Nearest UAV launched to the dark contact; EO/IR observation recorded | UAV-without-identity alert; fusion adds UAV source and "declared fishing but no fishing gear" style contradictions |
| 7 | `FLC_CAMERA_FAILURE` | An FLC CCTV goes offline; CCTV source DEGRADED | Station surveillance component drops; map icon struck through; Data Source Health shows fallback |
| 8 | `BOAT_BREAKDOWN` | A mission-ready (preferably patrolling) boat reports a CRITICAL engine defect | Patrol aborted; station/district/state readiness and recommendations change immediately |
| 9 | `COMMUNICATION_FAILURE` | A station's VHF base fails and primary link goes offline; one boat's telemetry freezes | Comms component drops; boat turns GREY / STALE ("last known"); recommendation warns to confirm by VHF |
| 10 | `CYCLONE_WARNING` | Simulated cyclone alert on all districts; fishing suspended | Top-bar weather chip RED; leadership "Cyber / Weather Threat"; recommendation shows warning text |
| 11 | `CYBER_INCIDENT` | Brute-force burst from a documentation-range IP + suspicious admin action | Cybersecurity view events; investigate/close workflow |
| 12 | `DISTRESS_CALL` | Odia WhatsApp-style report of a sinking boat with GPS and 6 POB | Immediate L1 incident in the verification queue |
| 13 | `MISSING_VESSEL` | Hindi report of an overdue gillnetter whose AIS went silent 6 h ago | L2 missing-boat incident linked to the registered vessel |
| 14 | `MEDICAL_EVACUATION` | English report: unconscious fisherman, chest pain, 12 NM east of Paradip, 8 POB | L1 medical incident with place-offset location |
| 15 | `SUSPICIOUS_LANDING` | Bengali night report of persons landing from an unknown boat ("smuggling" allegation) + radar contact | L3 **possible** suspicious landing (allegation not accepted as fact) + night-approach alert, community report in intelligence |
| — | `RESTORE_BASELINE` | Clears injected failures (cameras, comms, weather, exercise defects) | Readiness returns to baseline |

## 2. Automated test coverage (§53)

| Requirement | Automated test(s) |
|---|---|
| Page loading | `e2e/ui.spec.mjs` — every view of groups 01–09 (supervisor) and admin/cyber/audit views; asserts no JavaScript runtime errors |
| Login / logout | `test_platform::test_login_logout_and_landing`, `e2e … login, role-based landing, logout and invalid credentials` |
| Role restrictions | `test_role_restrictions` (12 cases), `test_denials_are_audited`, `test_admin_cannot_grant_intel_to_ineligible_role`, `test_intel_hidden_from_cop_without_need_to_know`, `test_jurisdiction_iic_cannot_task_other_station`, `e2e … role restrictions` |
| Map | `test_cop_snapshot_contents` (all object/layer types), `e2e … live nautical map renders operational layers offline and opens context panels` |
| Personnel search | `test_personnel_search_semantics_and_matrix`, `e2e … personnel and asset search` |
| Asset search | `test_asset_search`, e2e search test |
| Readiness propagation | `test_maintenance_propagates_to_station_recommendation_and_leadership`, `test_personnel_leave_reduces_crew_readiness`, `test_readiness_is_explainable` |
| Asset maintenance impact | same as above (station score, recommendation exclusion, leadership metric, tasking refusal) |
| Incident creation | acceptance tests; `test_suspicion_is_not_accepted_as_fact`; manual create in persistence test |
| Chatbot conversation | `test_acceptance`, `test_information_queries`, `test_myboat_requires_matching_mobile` |
| Multilingual input | `test_multilingual_classification` (Hindi, Bengali, Telugu, romanised Hindi, English) + Odia acceptance flows |
| Human takeover | `test_human_takeover_and_handoff` (bot silence, translated templates, forbidden status templates, MRCC handoff) |
| Supervisor tasking | acceptance tests; `test_operational_alert_and_personnel_tasking` |
| Field acknowledgement | acceptance tests; `test_unable_requires_reason_and_frees_asset` |
| Admin personnel CRUD | `test_admin_personnel_crud` (create, edit, transfer, qualification, training, archive, audit before/after) |
| Admin asset CRUD | `test_admin_asset_crud` (+ master-data edit refused for status-only roles) |
| Audit trail | chain verification in several tests; action-specific assertions throughout |
| Scenario injection | `test_all_scenarios_inject`, `e2e … scenario injection from the exercise console` |
| Data persistence & restart recovery | `test_persistence::test_restart_recovery_and_restore` (new app instance on same DB; no re-seed; backup → restore) |
| Security | `test_failed_login_lockout_and_unlock`, `test_mfa_enrolment_and_login`, `test_backup_and_audit_chain`, `test_evidence_upload_rejects_bad_type` |
| **Final acceptance (§54/§55)** | `tests/backend/test_acceptance.py` (API, all 22 steps) and `tests/e2e/acceptance.spec.mjs` (real browser, four sessions: citizen, operator, supervisor, boat master) |

Results at the time of writing: backend 47/47 on SQLite and 47/47 on PostgreSQL 16 + PostGIS 3.4; browser 8/8.

## 3. Manual demonstration checklist
1. Log in as each demo role and confirm the landing view matches RBAC_MATRIX.md.
2. Inject `BOAT_BREAKDOWN`; open Readiness › State and follow the station drop to the reason.
3. Inject `COMMUNICATION_FAILURE`; confirm the grey/stale boat on the map and the recommendation warning.
4. Turn off network access; reload the COP; confirm the offline banner and that all operational layers remain.
5. Run the five-minute demonstration in README.md.
