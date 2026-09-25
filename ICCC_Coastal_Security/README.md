# AI-Enabled Integrated Coastal Security & Maritime Domain Awareness Platform — Proof of Concept

**Prepared for the Coastal Security Wing, Odisha Police — operational demonstrator.**

> **ALL DATA IN THIS SYSTEM IS SIMULATED / POC DATA.** No real personnel, deployments, credentials, official boundaries,
> Government approvals or operational intelligence are represented. The live map uses public OpenStreetMap / OpenSeaMap tiles
> and is **NOT FOR NAVIGATION**. Marine Police Station names in the dataset are illustrative coastal localities, not the
> official list or official locations.

This is a working, integrated operational decision-support application, not a set of mock-ups. It is built around one chain:

**Live Nautical COP → Readiness → Live Force & Asset Status → AI Maritime Public Assistant → Rank/Role Access → Incident Command → Response → Evidence → Learning**

It answers ten operational questions directly:

| Question | Where it is answered |
|---|---|
| What is happening along the coast right now? | Landing page: Live Nautical Map (≈75 % of the workspace) + context panel + event feed |
| Where are our personnel and marine assets? | Map layers, Assets & Operations, Personnel › Current Deployment |
| Are they ready to respond? | Readiness (State → District → MPS → Asset/Personnel) with explained reasons |
| Exactly which personnel are qualified and available? | Personnel › Qualifications matrix + preset questions (e.g. night-patrol-qualified at Astaranga) |
| Which assets are operational and mission-ready? | Asset cards show *exists → operational → available → mission-ready* checklists |
| Which incidents / anomalies need attention? | Incident Command › Verification Queue, Active Alerts, Maritime Intelligence |
| What is the nearest suitable response resource? | Resource recommendation (distance, readiness, qualified crew, fuel, comms) — *AI RECOMMENDATION — HUMAN AUTHORISATION REQUIRED* |
| Who is authorised to decide and task it? | Rank + role + posting + jurisdiction RBAC; only supervisory roles hold `TASK_ASSETS` |
| What action has been taken? | Timestamped order transitions, incident timeline, audit trail |
| What was the outcome and what can be learned? | Outcome, evidence (hashed), auto-generated After-Action Review with improvement points |

## Quick start

**Docker (PostgreSQL + PostGIS):**
```bash
cp .env.example .env        # change SECRET_KEY, POSTGRES_PASSWORD, DEMO_PASSWORD
docker compose up --build   # then open http://localhost:8000
```

**Without containers (SQLite, single process):**
```bash
./start_scripts/run_local.sh      # creates .venv, installs deps, builds the frontend, starts on :8000
```

Log in with one of the demo accounts shown on the login page (shared demo password = `DEMO_PASSWORD`, default `Demo@2026`).
Citizens use the public channel at **`/citizen/`** (no login). Full instructions: [docs/INSTALLATION.md](docs/INSTALLATION.md).

### Five-minute demonstration
1. Open `/citizen/` on a phone-sized window → choose **ଓଡ଼ିଆ** → type `ଆମ ଡଙ୍ଗାର ଇଞ୍ଜିନ ବନ୍ଦ ହୋଇଯାଇଛି, ଡଙ୍ଗା ଭାସି ଯାଉଛି` → share location → `୫ ଜଣ ଅଛୁ`.
2. Log in as `operator.iccc` → Incident Command › Verification Queue → open the new L2 incident → read the original Odia and the canonical English → **Verify & take ownership**.
3. Log in as `supervisor.iccc` → the incident → **Task resource** → review the ranked, explained recommendation → select → **Authorise & send order**.
4. Log in as `master.fib04` (or the master of the chosen boat) → Field Unit Console → Acknowledge → Accept → EN ROUTE → watch it move on the map → ON SCENE → COMPLETED.
5. Operator records the outcome (C6) → supervisor preserves track logs as evidence and closes (C7) → read the After-Action Review. The citizen chat shows only verified updates.

The same flow is automated in `tests/backend/test_acceptance.py` (API) and `tests/e2e/acceptance.spec.mjs` (real browser, four sessions).

## Project structure
```
ICCC_Coastal_Security/
├── frontend/                Next.js 15 + React 19 + TypeScript + MapLibre GL (static export)
├── backend/app/             FastAPI: routers (API), services (readiness, recommendation, analytics, fusion,
│                            chatbot, incidents, orders, simulator, scenarios), RBAC, audit, security
├── database/init/           PostGIS initialisation (TimescaleDB-ready notes)
├── seed/                    Deterministic synthetic dataset generator (SIMULATED / POC DATA)
├── docker/                  Application Dockerfile (+ optional extra CA certs for proxied networks)
├── docs/                    Architecture, workflows, data dictionary, RBAC, security, limitations, roadmap …
├── tests/backend/           pytest API suite incl. end-to-end acceptance (runs on SQLite and PostgreSQL)
├── tests/e2e/               Playwright browser tests incl. four-session acceptance demonstration
├── assets/static_nautical_chart/   Offline static chart slot (placeholder — no chart was supplied)
├── start_scripts/           run_local.sh, dev_server.sh, run_tests.sh, gen_docs.py
├── docker-compose.yml
└── .env.example
```

## Documentation
| Document | Contents |
|---|---|
| [INSTALLATION.md](docs/INSTALLATION.md) | Docker and local installation, configuration, troubleshooting |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Components, data flow, service boundaries, evolution path |
| [DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md) · [tables](docs/DATA_DICTIONARY_TABLES.md) | Semantics (posted/present/available/sea-ready…, exists/operational/available/mission-ready, provenance) and every column |
| [API.md](docs/API.md) | Every endpoint (generated); interactive docs at `/docs` |
| [RBAC_MATRIX.md](docs/RBAC_MATRIX.md) | Roles × permissions, jurisdiction and need-to-know rules (generated) |
| [SYNTHETIC_DATASET.md](docs/SYNTHETIC_DATASET.md) | What is simulated, how it is generated, deliberate demo states |
| [CYBERSECURITY.md](docs/CYBERSECURITY.md) | Controls implemented, gaps, production hardening, backup/DR, manual fallback |
| [CHATBOT_DESIGN.md](docs/CHATBOT_DESIGN.md) | Language pipeline, intents, priorities, human-in-the-loop guarantees |
| [INCIDENT_WORKFLOW.md](docs/INCIDENT_WORKFLOW.md) | C0–C8 lifecycle, who may do what, citizen notifications, evidence, AAR |
| [COMMAND_WORKFLOW.md](docs/COMMAND_WORKFLOW.md) | Command desk, movement orders, acknowledgements, recommendation logic |
| [TEST_SCENARIOS.md](docs/TEST_SCENARIOS.md) | The 15 exercise scenarios + automated test coverage map |
| [KNOWN_LIMITATIONS.md](docs/KNOWN_LIMITATIONS.md) | What this POC does not do — read before any demonstration |
| [ROADMAP.md](docs/ROADMAP.md) | Future integration path to a production operational platform |

## Test status (last run 2026-09-25)
* `pytest tests/backend` — **47 passed** on SQLite and **47 passed** on PostgreSQL 16 + PostGIS 3.4.
* `playwright test` (tests/e2e) — **8 passed**, including the full four-session UI acceptance scenario, every workspace view loading without JavaScript runtime errors, role restrictions and the offline map.
* `docker compose up` verified: seeded on first start, simulator running, PostGIS spatial views populated.

Run everything with `./start_scripts/run_tests.sh`.

## Facts referenced in the application and their verification status
The application shows a small number of real-world facts to citizens or operators. Each was checked against public sources on
**2026-09-24/25**, but **must be re-verified by the Coastal Security Wing before any public use**:

| Fact | Where used | Sources consulted | Status |
|---|---|---|---|
| **112** is India's single emergency number (ERSS) | Citizen chat safety messages | [MHA — ERSS](https://www.mha.gov.in/en/commoncontent/emergency-response-support-system-erss), [112.gov.in](https://112.gov.in/), [Odisha Police — ERSS](https://odishapolice.gov.in/main/?q=node/4103) | Corroborated (primary sources) |
| **1554** is the Indian Coast Guard maritime SAR helpline | Citizen chat safety messages | [Indian Coast Guard — Search and Rescue](https://indiancoastguard.gov.in/search-and-rescue) (search listing), [PIB via GlobalSecurity, Apr 2025](https://www.globalsecurity.org/wmd/library/news/india/2025/india-250404-india-pib03.htm), [Vartha Bharati](https://english.varthabharati.in/karavali/indian-coast-guard-gets-new-emergency-helpline-number) | Corroborated; the ICG page itself could not be opened from the build environment |
| **NABHMITRA** is the ISRO-developed app of the Vessel Communication and Support System (VCSS) for fishing vessels | "My Boat", data-source registry (marked NOT INTEGRATED) | [Dept of Fisheries (X post)](https://x.com/FisheriesGoI/status/1941088548930666633), [Vikaspedia](https://en.vikaspedia.in/viewcontent/agriculture/fisheries/advisories-for-fisheries-sector/nabhmitra-app-for-indian-fishermen?lgn=en), [ICSF](https://icsf.net/newss/india-indigenous-transponders-become-lifeline-for-fishermen-during-cyclone-dana/) | Corroborated |
| Odisha has **6 coastal districts** and **18 Marine Police Stations** | Dataset shape only | [Odisha Police — Coastal security](https://odishapolice.gov.in/main/?q=node/163) (search snippet), [Prameya News](https://www.prameyanews.com/18-coastal-police-stations-in-odisha-to-strengthen-coastal-security), [Deccan Chronicle](https://www.deccanchronicle.com/nation/guardians-of-the-blue-frontier-odishas-marine-police-fortify-coastal-defences-1914180) | Counts corroborated. **The official station list could not be retrieved** (the Odisha Police site was blocked from the build environment), so station names/locations are illustrative |
| VHF **Channel 16** is the international distress channel | Citizen safety messages | International maritime practice (ITU Radio Regulations) | Not separately re-verified in this build |

Weather, vessel, AIS, radar, UAV, CCTV, personnel and asset data are **all synthetic**. The chatbot's Odia, Hindi, Bengali and Telugu
lexicon and reply templates are POC drafts and **must be reviewed by native speakers** before public use.

## Licence
See the repository `LICENSE`. Map data © OpenStreetMap contributors; seamarks © OpenSeaMap (public tile services, subject to their usage policies).
