# Installation Guide

The POC runs either as a two-container Docker Compose stack (PostgreSQL + PostGIS, recommended) or as a single local
process on SQLite (no containers). Kubernetes is **not** required.

## 1. Docker Compose (recommended)

Prerequisites: Docker Engine 24+ with the Compose plugin; ~2 GB disk; ports 8000 free.

```bash
cd ICCC_Coastal_Security
cp .env.example .env
# edit .env: SECRET_KEY (python3 -c "import secrets;print(secrets.token_urlsafe(48))"), POSTGRES_PASSWORD, DEMO_PASSWORD
docker compose up --build -d
docker compose logs -f app     # wait for "Seeded synthetic POC dataset" and "Uvicorn running"
```
Open **http://localhost:8000**. The database is seeded automatically on first start only (existing data is never overwritten).

| Task | Command |
|---|---|
| Stop | `docker compose down` (data kept in named volumes) |
| Reset all data | `docker compose down -v && docker compose up -d` |
| Database shell | `docker compose exec db psql -U iccc -d iccc` |
| PostGIS views | `SELECT * FROM gis_assets LIMIT 5;` (also `gis_vessels`, `gis_incidents`) |
| Logical backup | `docker compose exec db pg_dump -U iccc iccc > iccc_$(date +%F).sql` |

**Behind a TLS-inspecting proxy** (common on Government networks): put the proxy root CA as `docker/certs/<name>.crt`
(git-ignored) — it is trusted during `npm ci`/`pip install` and at runtime. If Docker Hub rate-limits base images, set
`PYTHON_IMAGE` / `NODE_IMAGE` in `.env` to a registry mirror (e.g. `mirror.gcr.io/library/python:3.11-slim`).
If the proxy only listens on the host's loopback, build with `docker build --network host -f docker/app.Dockerfile -t iccc_coastal_security-app .`
then `docker compose up -d --no-build`.

## 2. Local run without containers (SQLite)

Prerequisites: Python 3.11+, Node.js 20+ (22 tested), npm.

```bash
cd ICCC_Coastal_Security
./start_scripts/run_local.sh          # creates .venv, installs deps, builds frontend once, serves on 127.0.0.1:8000
```
Data (SQLite file, evidence files, backups) is written to `ICCC_Coastal_Security/data/`. Delete that folder to reseed.
Background mode: `./start_scripts/dev_server.sh start|stop|restart [--fresh]`.

Frontend development with hot reload (backend on :8000 in another terminal):
```bash
cd frontend && NEXT_DEV_SERVER=1 NEXT_PUBLIC_API_BASE=http://localhost:8000 npm run dev   # http://localhost:3000
```

## 3. Configuration reference (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | SQLite in `DATA_DIR` | e.g. `postgresql+psycopg://user:pass@host:5432/iccc` |
| `SECRET_KEY` | POC placeholder (flagged in Cyber view) | JWT signing key — **must be changed** |
| `DEMO_PASSWORD` | `Demo@2026` | Password for the synthetic demo accounts (seed time only) |
| `SESSION_IDLE_MINUTES` | 30 | Idle timeout (server + client) |
| `JWT_TTL_MINUTES` | 480 | Absolute token lifetime |
| `MAX_FAILED_LOGINS` / `LOCKOUT_MINUTES` | 5 / 15 | Account lockout policy |
| `PBKDF2_ITERATIONS` | 210000 | Password hashing work factor |
| `SIM_ENABLED` | true | Run the movement/analytics simulator |
| `SIM_TICK_SECONDS` / `SIM_TIME_FACTOR` | 3 / 20 | Simulator cadence and time acceleration |
| `SEED_ON_START` / `SEED_RANDOM` | true / 20260924 | Seed an empty database; deterministic seed |
| `STALE_AFTER_SECONDS` | 300 | When telemetry is shown as STALE / GREY |
| `MAP_TILE_URL` / `SEAMARK_TILE_URL` | OSM / OpenSeaMap | Point at an internal tile server or authorised chart service |
| `DATA_DIR` | `./data` | Evidence files, citizen media, backups, SQLite DB |
| `CORS_ORIGINS` | `http://localhost:3000` | Only needed for the Next.js dev server |

## 4. Running the tests
```bash
./start_scripts/run_tests.sh                    # backend (SQLite) + Playwright browser suite
SKIP_E2E=1 ./start_scripts/run_tests.sh         # backend only
TEST_DATABASE_URL=postgresql+psycopg://u:p@localhost:5432/test ./start_scripts/run_tests.sh   # backend on PostgreSQL
```
The Playwright suite starts its own server on port 8100 with a throw-away database and the simulator disabled.
It uses the Chromium that Playwright manages (`npx playwright install chromium` on a new machine).

## 5. Offline operation
The application works with no internet: the map falls back to an approximate schematic coastline (clearly labelled) and
all operational layers, readiness, chat and workflows are served locally. Public tiles return automatically when reachable.

## 6. Troubleshooting
| Symptom | Fix |
|---|---|
| "Frontend not built" at `/` | `cd frontend && npm ci && npx next build` |
| Login says account locked | Wait `LOCKOUT_MINUTES` or unlock in Administration › Users |
| Map shows only coastline | Public tiles unreachable (expected offline) — operational layers still render |
| Session ends unexpectedly | Idle timeout (`SESSION_IDLE_MINUTES`) or an administrator revoked the session |
| `database is locked` on SQLite under load | Use the PostgreSQL stack for multi-user demonstrations |
