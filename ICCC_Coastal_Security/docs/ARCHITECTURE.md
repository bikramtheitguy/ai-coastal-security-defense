# Architecture

## 1. Guiding principle
Operational workflow first, screens second. Every component serves the chain
**Map → Situation → Readiness → Incident → Decision → Response → Evidence → Learning**. The home screen is the map; charts and
analytics are drill-downs, not the landing page.

## 2. Component view

```
 Browser (ICCC workstation / video wall / laptop / tablet)          Citizen phone (web / QR / simulated WhatsApp)
 ┌──────────────────────────────────────────────────────┐          ┌─────────────────────────────┐
 │ Next.js 15 static app (React 19, TypeScript)          │          │ /citizen  mobile-first chat │
 │  • MapLibre GL live nautical map (offline-capable)    │          └──────────────┬──────────────┘
 │  • 10 workspace groups, context panel, event feed     │                         │ /api/public/*
 └───────────────────────┬──────────────────────────────┘                         │ (rate-limited, no login)
                         │ HTTPS (TLS at reverse proxy)  /api/* with Bearer JWT    │
 ┌───────────────────────▼─────────────────────────────────────────────────────────▼───────────┐
 │ FastAPI application (single deployable in the POC; clean internal service boundaries)         │
 │  routers/  auth · cop · readiness · personnel · assets · incidents · intel · chat · command ·   │
 │            system (analytics, health, audit, backup, scenarios, search) · admin                │
 │  deps.py   authentication, session idle timeout, permission & jurisdiction checks (audited)    │
 │  services/ readiness engine · resource recommendation · vessel analytics · multi-source fusion │
 │            chatbot (pipeline, lexicon, dialogue) · incident lifecycle + AAR · orders ·          │
 │            simulator (movement, telemetry) · scenarios · serializers                           │
 │  audit.py  tamper-evident SHA-256 hash-chained audit log                                        │
 └───────────────────────┬─────────────────────────────────────────────────────────────────────┘
                         │ SQLAlchemy 2 (portable types)
 ┌───────────────────────▼──────────────────────┐
 │ PostgreSQL 16 + PostGIS 3.4 (compose)         │   SQLite (single-machine demo / tests)
 │  gis_assets / gis_vessels / gis_incidents     │
 │  views; vessel_track_points hypertable-ready  │
 └───────────────────────────────────────────────┘
```

## 3. Key design decisions

| Decision | Rationale |
|---|---|
| **Readiness computed on read** from relational data (no stored scores) | Guarantees propagation: a boat put under maintenance immediately changes station/district/state scores, recommendations and leadership metrics. Scores are always explainable because the reasons are produced by the same computation. |
| **Explicit semantics** (posted / present / on duty / deployed / available / sea-ready / qualified; exists / operational / available / mission-ready) | These are not interchangeable operationally; each is a separate computed flag (`services/readiness.py`). |
| **Provenance on every operational entity** (`ProvenanceMixin`) | Source, source time, received time, confidence, verification, classification, owner; freshness is computed and STALE values are shown as last-known, never as current truth. |
| **Advisory AI only** | Recommendation, analytics and chatbot outputs are labelled and cannot change operational state. Tasking requires `TASK_ASSETS`; dispatch confirmation (C5) needs a field-confirmed EN ROUTE on an authorised order. |
| **Deterministic, explainable rule engines** for analytics and language understanding | Auditable, offline, no data egress. ML/LLM components can be added behind the same interfaces (see ROADMAP). |
| **Static-exported frontend served by the API** | One container / one process for the POC; no Node runtime in production. |
| **No text glyphs on the map** | Map labels need a font server; using canvas-drawn icons + tooltips keeps the map fully functional offline. |
| **Simulation engine inside the app, state in the DB** | Restart recovery is automatic; the simulator can be disabled and replaced by real feeds without code changes elsewhere. |
| **Soft delete everywhere** | Personnel/assets/master data are deactivated/archived; history remains auditable. |

## 4. Request flow example — tasking a boat
1. Supervisor opens incident → `GET /api/incidents/{id}/recommendation` → `services/recommend.py` loads assets with crew and
   defects, runs `asset_readiness()` per candidate, filters to mission-ready, scores and returns ranked + excluded lists with reasons,
   weather and provenance.
2. Supervisor chooses an asset → `POST /api/orders` → `deps.require_any` checks `TASK_ASSETS`, `ensure_station` checks jurisdiction,
   `services/orders.create_order` re-checks mission readiness (override requires an audited reason), stores the recommendation
   snapshot for the AAR, marks the asset TASKED, records supervisor review and an incident timeline event → audit row appended to hash chain.
3. Field unit → `POST /api/orders/{id}/transition` (recipient check) → EN ROUTE sets the asset moving, deploys crew, confirms
   dispatch (C5) and sends the verified citizen update → simulator moves the asset each tick → ON SCENE → COMPLETED → asset returns.

## 5. Service boundaries for evolution
The `services/` modules have no HTTP dependencies and communicate through the database, making them candidates for extraction:

| Candidate service | Why extract | Possible language |
|---|---|---|
| Track ingest & analytics (AIS/radar/UAV) | High-rate streams; CPU-bound geometry | Go or Rust consumer on Kafka |
| Fusion / correlation | Stateful windowed joins | Rust / Flink-style stream processor |
| Chatbot language service | Swap in approved NMT / LLM providers | Python (model serving) |
| Readiness | Read-heavy; can be cached/materialised | Stay Python; materialise via events |

**Event streaming readiness:** state changes already produce `EventFeed` rows and audit entries at single choke points
(`services/common.feed`, `audit.audit`). Publishing the same payloads to Kafka topics (`iccc.alerts`, `iccc.orders`, `iccc.incidents`,
`iccc.tracks`) is an additive change in those two functions.

## 6. Map & chart strategy
* Base: public OSM raster + OpenSeaMap seamarks (POC; **not for navigation**) — URLs configurable to an internal tile server.
* Offline: approximate schematic coastline from the backend (labelled not official).
* ENC: a layer slot is reserved in the layer model; an authorised S-57/S-101 service (e.g. via a WMS/vector-tile gateway)
  can be added as an additional source without redesign.
* Static chart: served as an image, deliberately not georeferenced until calibration is verified.

## 7. Deployment topology (target)
Reverse proxy (TLS, HSTS, WAF) → 2+ app replicas (stateless; simulator disabled on all but one or replaced by ingest services) →
PostgreSQL HA (Patroni) + PostGIS + TimescaleDB → object store for evidence (WORM) → Kafka for feeds. Kubernetes-ready: the image
is stateless apart from `DATA_DIR` (move evidence/backups to object storage), health endpoint `/api/public/info`.
