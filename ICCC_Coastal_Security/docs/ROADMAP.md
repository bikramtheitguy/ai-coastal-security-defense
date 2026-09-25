# Future Integration Roadmap

Phasing is indicative; each integration requires the owning agency's approval, data-sharing agreements and security accreditation.
Items marked *(verify)* depend on external systems whose current interfaces must be confirmed with the owner before design.

## Phase 1 — Harden the demonstrator (0–3 months)
* Replace illustrative master data with authoritative station, personnel and asset registers (one-off import + Administration workflows).
* Native-speaker validation of the Odia, Hindi, Bengali and Telugu lexicon and replies; collect anonymised real message samples.
* Reverse proxy with TLS/HSTS, named accounts, demo accounts disabled, secrets in a vault, CERT-In-empanelled VAPT.
* PostgreSQL PITR backups, restore drills, runbooks; move evidence to WORM object storage.
* Pilot at one district ICCC with parallel paper process.

## Phase 2 — Live maritime picture (3–9 months)
| Feed | Approach |
|---|---|
| AIS | Terrestrial/satellite AIS via approved provider or national feed *(verify)*; Kafka ingest service → `vessel_track_points` (TimescaleDB hypertable) |
| Coastal radar chain | Track feed through the owning agency's gateway *(verify)*; correlation in the fusion service |
| NABHMITRA / VCSS transponders | Position & SOS messages through the Department of Fisheries / ISRO system *(verify interface and data-sharing terms)* |
| Vessel registry | National fishing-vessel registration database *(verify)* for identity checks and "My Boat" OTP |
| UAV | Ground-control-station telemetry + video references; EO/IR detections as alerts |
| CCTV at FLCs | VMS event/health API (status first, video references second) |
| Weather | IMD / INCOIS bulletins and warnings *(verify feeds and licensing)* replacing simulated weather |
| Charts | Authorised ENC service (S-57/S-101 via WMS/vector tiles) in the reserved layer slot; licensed static chart with verified georeferencing |

Architecture: Kafka topics per feed → Go/Rust ingest & analytics services → PostGIS/TimescaleDB → existing API (see ARCHITECTURE.md §5).

## Phase 3 — Coordination & public channels (6–12 months)
* **ERSS-112**: MHA describes ERSS-112 as receiving distress over channels including chatbot and WhatsApp at State PSAPs — explore
  bi-directional incident exchange with Odisha's ERSS rather than a parallel citizen channel *(verify with MHA / State PSAP)*.
* **MRCC / MRSC (Indian Coast Guard)**: structured SAR handoff and status sync replacing the recorded-only handoff *(verify protocol)*.
* WhatsApp Business API (approved BSP), SMS fallback (shortcode), IVR voice with speech-to-text in Odia; native mobile app with offline SOS.
* Approved neural translation / LLM provider behind the chatbot interface (data residency, original always preserved, human-in-the-loop unchanged).

## Phase 4 — Scale & intelligence (12+ months)
* Government IdP (SSO, MFA enforcement, PKI tokens); attribute-based access for intelligence compartments.
* SIEM/SOC integration; UEBA on privileged activity.
* ML models for anomaly detection trained on validated historical tracks, evaluated against the rule engine before replacement.
* Multi-site HA (Kubernetes, Patroni), DR site, materialised readiness, event-sourced audit to external WORM.
* Inter-state / national MDA data exchange under applicable sharing frameworks *(verify)*.

## Success measures to agree with the Wing
Alert-to-verification time, verification-to-dispatch time, share of incidents with GPS-quality location, share of assets with fresh telemetry,
readiness data accuracy (UNABLE rate), false-alert rate per detector, citizen report completion rate by language.
