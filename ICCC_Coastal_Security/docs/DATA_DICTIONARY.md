# Data Dictionary — semantics

Every column of every table is listed in [DATA_DICTIONARY_TABLES.md](DATA_DICTIONARY_TABLES.md) (generated from the model).
This document defines the **meaning** of the operational terms; the same definitions are implemented in `backend/app/services/readiness.py`.

## 1. Personnel status (distinct, not interchangeable)

| Term | Definition (as computed) |
|---|---|
| **Posted** | Active record with a Marine Police Station posting. |
| **Present** | Posted and physically reporting: duty status `ON_DUTY`, `STANDBY` or `DEPLOYED`. (`OFF_DUTY`, `LEAVE`, `MEDICAL_LEAVE`, `TRAINING`, `ABSENT` are not present.) |
| **On duty** | Present with status `ON_DUTY` or `DEPLOYED`. |
| **Deployed** | Present and committed to a patrol / response (`DEPLOYED`). |
| **Available** | Present, *not* deployed (`ON_DUTY`/`STANDBY`) and medically fit with a valid clearance — can be tasked now. |
| **Sea ready** | Present, medically fit, and holding valid `SWIMMING` + `SEA_SURVIVAL` qualifications. May be deployed (already at sea). |
| **Qualified (X)** | Holds qualification X with `valid_until` ≥ today. Boat master = `BOAT_CREW` + `NAVIGATION`. |
| **Qualified boat crew (available)** | Available ∧ sea ready ∧ `BOAT_CREW` valid. |
| **Unavailable** | Posted but neither available nor deployed (leave, training, medical, absent, off duty, unfit). |

Duty status values: `ON_DUTY`, `STANDBY`, `OFF_DUTY`, `DEPLOYED`, `LEAVE`, `MEDICAL_LEAVE`, `TRAINING`, `ABSENT`.
Qualification codes: `BOAT_CREW`, `NAVIGATION`, `MARINE_VHF`, `UAV_PILOT`, `SWIMMING`, `SEA_SURVIVAL`, `SAR`, `FIRST_AID`, `NIGHT_OPS`, `WEAPONS`, `CYBER_IT`.

## 2. Asset readiness chain

| Stage | Definition |
|---|---|
| **Exists** | Active in the asset register (not archived). |
| **Operational** | Operational status `OPERATIONAL`/`DEGRADED`, no open CRITICAL defect, availability not `MAINTENANCE`/`DEFECTIVE`/`GROUNDED`. |
| **Available** | Operational, not held in `RESERVE`, and not committed to an order (`TASKED`/`EN_ROUTE`/`ON_SCENE`). A patrolling asset is available (diverting it is flagged). |
| **Mission ready** | Exists ∧ operational ∧ available ∧ fuel/battery ≥ threshold ∧ qualified crew (boats: required number of present, sea-ready `BOAT_CREW` members including a qualified master; UAV: qualified pilot) ∧ communications (VHF + GPS; UAV: link + GNSS) ∧ safety equipment complete ∧ telemetry not stale. |

Advisory (non-blocking) checks: certification validity, maintenance due date, major defects (−5 points each).
Score = weighted share of passing checks (operational 30, available 15, fuel 15, crew 20, comms 10, safety 5, certification 5).
Colour: GREEN ≥ 85, AMBER ≥ 60, RED below; a not-mission-ready asset is never GREEN; non-operational is RED; stale telemetry is GREY.

Availability values: `AVAILABLE`, `DEPLOYED`, `MAINTENANCE`, `DEFECTIVE`, `GROUNDED`, `RESERVE`.
Mission status values: `IDLE`, `PATROLLING`, `TASKED`, `EN_ROUTE`, `ON_SCENE`, `RETURNING`.

## 3. Station / district / state readiness
Station score = weighted mean of available components (default weights, editable in Administration › Risk Weights / System Configuration):

| Component | Weight | Measure |
|---|---|---|
| Boats | 35 % | mission-ready boats ÷ max(minimum required, 75 % of non-reserve holding) |
| Crew | 25 % | available sea-ready personnel ÷ station minimum |
| Communications | 15 % | VHF base (50), backup VHF test in date (20), primary link (20), backup link (10) |
| Surveillance | 15 % | operational sensors + cameras ÷ total |
| UAV | 10 % | a mission-ready UAV exists (only if the station holds UAVs) |

Components that do not apply are excluded and the weights renormalised. District = mean of its stations; State = mean of districts.
Every score is accompanied by a severity-ranked list of reasons (e.g. "FIB-12T-03 not mission-ready — under maintenance",
"Backup VHF test overdue").

## 4. Provenance fields (on assets, vessels, alerts, observations, incidents, personnel, places, weather)
`source`, `source_ts` (time at source), `received_ts`, `confidence` (0–1), `verification` (`UNVERIFIED` / `SYSTEM` / `HUMAN_VERIFIED` / `DISPUTED`),
`classification` (data classification label), `data_owner`. **Freshness** is computed: LIVE ≤ 60 s, RECENT ≤ `STALE_AFTER_SECONDS`, else STALE.

## 5. Incident record (§29)
Incident ID, detection/source time (`detected_at`), alert time, verification time, classification, location (+ confidence GPS / REPORTED /
APPROXIMATE / UNKNOWN), source, risk, confidence, human verification, assigned agency, MPS, personnel, asset, dispatch / launch / arrival
times, response notes, evidence, outcome, closure, After-Action Review. Lifecycle codes C0–C8 — see INCIDENT_WORKFLOW.md.

## 6. Evidence
Evidence ID, kind (PHOTO / VIDEO / VOICE / SCREENSHOT / TRACK / UAV / CCTV_REF / NOTE), source, created time (capture), uploaded time,
officer, incident, SHA-256 of the stored bytes (integrity placeholder), chain-of-custody status (COLLECTED / SEALED / TRANSFERRED /
RELEASED) and an append-only custody log.

## 7. Orders
Order type (OPERATIONAL_ALERT / PERSONNEL_TASKING / ASSET_MOVEMENT / INCIDENT_RESPONSE), priority (FLASH / IMMEDIATE / PRIORITY / ROUTINE),
issuer and rank, recipients, instruction, validity, destination, status and a timestamped transition list; acknowledgements are stored separately.

## 8. Vessel analytics & risk
Alert types: AIS_LOST, DARK_VESSEL, IDENTITY_MISMATCH, ABNORMAL_SPEED, UNUSUAL_COURSE, LOITERING, RESTRICTED_ZONE, NIGHT_APPROACH,
RENDEZVOUS, REPEATED_VISITS, ROUTE_DEVIATION, SENSITIVE_PROXIMITY, WATCHLIST, RADAR_NO_AIS, UAV_NO_ID. Vessel risk = 1 − Π(1 − wᵢ) over
active alert types (weights in `risk.weights`), ×0.7 multiplier on the complement for designated TOIs; HIGH ≥ 0.6, MEDIUM ≥ 0.3.
"Of concern" (used by proximity, night-approach and rendezvous rules) = no verified identity, AIS-fitted but silent, or a designated TOI —
small craft without AIS are normal and are not flagged for that alone.
