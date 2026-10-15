# Local POC installation

Run this prototype in a disposable local environment with synthetic data. Known authorization/MFA findings prevent treating it as a shared operational service. Preserve existing data and history; use a new data directory or Compose project for a fresh exercise.

## Docker Compose

Prerequisites: Docker with Compose and enough space for Python/Node images. The application uses PostgreSQL/PostGIS and a static frontend build.

From `ICCC_Coastal_Security`, copy `.env.example` to `.env`. Set independently generated values for `SECRET_KEY`, `POSTGRES_PASSWORD` and `DEMO_PASSWORD`; for example use `python -c "import secrets; print(secrets.token_urlsafe(48))"` for each. These are local demo credentials; do not reuse any real account password.

```bash
docker compose -p iccc-poc-demo up --build
```

Open http://127.0.0.1:8000. The default port bind is loopback. Seeding occurs on an empty database; use your configured demo password for the displayed synthetic accounts. Changing the environment after seeding does not reset those account passwords.

Stop with `docker compose -p iccc-poc-demo down`; named volumes are retained. For another clean exercise use a new project name and a free APP_PORT. Remove volumes only after confirming they contain disposable test data. Do not copy shared-network deployment commands from an unverified demo.

## Local Python and Node

The existing scripts target Python 3.11+ and Node 22 with Bash on Linux/WSL. They are separate from the repository root's Python 3.13/uv research environment.

```bash
cd ICCC_Coastal_Security
export SECRET_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(48))')"
export DEMO_PASSWORD="$(python3 -c 'import secrets; print(secrets.token_urlsafe(24))')"
./start_scripts/run_local.sh
```

The script creates a local virtual environment, installs development requirements, builds the static frontend and binds to 127.0.0.1:8000. Data defaults to this directory's `data/`. A local `.env` file is read by Compose; the Bash script does not automatically load it. Set environment variables explicitly.

For frontend development, start the API separately, then run from `frontend/`:

```bash
npm ci
NEXT_DEV_SERVER=1 NEXT_PUBLIC_API_BASE=http://127.0.0.1:8000 npm run dev
```

Use the Python executable from that environment for the [tests](VALIDATION.md). Native Windows support for these Bash launchers has not been verified; the root Streamlit instructions provide a separate Windows-supported research demo.

## Configuration

| Variable | Meaning |
| --- | --- |
| SECRET_KEY / POSTGRES_PASSWORD / DEMO_PASSWORD | Generated local secrets; code still contains fallback demo values, so explicit configuration is required |
| APP_PORT / APP_BIND_HOST | Compose host port and bind address; default 8000 / 127.0.0.1 |
| DATA_DIR / DATABASE_URL | Local files and database; select disposable locations |
| SIM_ENABLED / SIM_TIME_FACTOR | Synthetic simulation switch and movement acceleration |
| SEED_ON_START / SEED_RANDOM | Empty-database synthetic seeding and deterministic generator seed |
| SESSION_IDLE_MINUTES / JWT_TTL_MINUTES | Local session idle and absolute-token lifetime |
| CORS_ORIGINS | Explicit frontend development origin; no wildcard for authenticated use |
| MAP_TILE_URL / SEAMARK_TILE_URL | Public or approved internal map services; attribution and usage terms still apply |
| TEST_DATABASE_URL / ICCC_ALLOW_TEST_DB_RESET | Destructive test-schema reset target and explicit opt-in; see validation guide |

## Troubleshooting and limits

A missing frontend requires `npm ci` and `npm run build` in `frontend/`. Unreachable tiles should show the labelled schematic fallback; this is not an authoritative chart. SQLite contention under concurrent writes needs investigation or the local PostgreSQL stack, not a production-readiness claim.

For an approved TLS-inspecting proxy, the existing Docker build accepts CA certificates in `docker/certs/`; do not disable certificate verification. No Kubernetes, live AIS key, paid AI provider or operational dataset is needed for this POC.
