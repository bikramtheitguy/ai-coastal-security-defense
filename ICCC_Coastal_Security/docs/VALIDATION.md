# POC validation and demonstration

Every result must identify the tested commit, environment, command and CI/artifact link. Record unavailable checks as unverified. The source snapshot reviewed on 15 October 2026 was `f24bf50180e19ef8ccfafcb40ea5f8696485fa03`; the earlier README's September pass counts are historical author-reported results.

## Automated checks

The repository workflow `.github/workflows/iccc-poc.yml` runs the existing backend suites with SQLite and a disposable PostgreSQL/PostGIS service, plus frontend TypeScript/build and Playwright browser tests. It has no production secrets or deployment step. The research/demo workflow also tests the new synthetic incident contract.

From the ICCC directory, after installing development dependencies:

```bash
python -m pytest tests/backend -q
python start_scripts/gen_docs.py
cd frontend
npm ci
npm run typecheck
npm run build
cd ../tests/e2e
npm ci
npx playwright install chromium
npm test
```

For browser tests, `PYTHON` must point at the backend environment's absolute Python executable. Browser setup uses a disposable directory under `/tmp`; run these Bash-based scripts on Linux/WSL. Do not point them at an existing server or retained records.

For PostgreSQL tests, use only a newly created local database named with the `iccc_test_` prefix. Set `TEST_DATABASE_URL` and explicitly set `ICCC_ALLOW_TEST_DB_RESET=1`. The fixture drops and recreates the public schema; the guard rejects other hosts/names. Neither this guard nor its opt-in makes an existing database safe to erase. CI creates its own service database.

## Synthetic incident exchange

From the repository root with the baseline environment:

```bash
uv run --locked python -m coastal_security.incidents --input data/synthetic_incident_exchange.json
uv run --locked pytest tests/test_incident_contract.py
```

Expected output: two synthetic records, one per fictional target state, human review required and persisted=false. It validates data without sending it to ICCC or Lovable.

## Fifteen-minute local acceptance exercise

| Step | Actor and action | Evidence to retain |
| --- | --- | --- |
| 1 | Start fresh local synthetic seed; open citizen view in a separate session | Commit/configuration, visible synthetic label, no external operational connection |
| 2 | Submit the fixture-style engine-failure example with invented location/person count; avoid browser GPS | Original text, interpretation marked for review, provisional incident ID and timestamps |
| 3 | Operator verifies and takes ownership | Actor, C2 transition and citizen update; operator cannot issue an unauthorized order |
| 4 | Supervisor reviews suitability and authorizes an order | Ranked and excluded resources with reasons, chosen asset, decision actor and order ID |
| 5 | Field user acknowledges/accepts and reports EN_ROUTE | Order transitions and attributable dispatch confirmation; SENT alone must not promise dispatch |
| 6 | Field user records ON_SCENE and completion; operator records outcome | Ordered timeline, original versus received times, recorded outcome |
| 7 | Authorized reviewer preserves synthetic track evidence and closes | Hash verification, custody history, C7 and reviewed AAR |
| 8 | Repeat wrong-jurisdiction read/mutation and stale/unavailable-source cases | Denial without state change; freshness/error shown; preserved authorized workflow |

Existing implementations are exercised by `tests/backend/test_acceptance.py` and `tests/e2e/acceptance.spec.mjs`. The negative jurisdiction and manual-dispatch cases above are additional acceptance requirements and remain blockers until implemented.

## Evidence status

| Check | Status at foundation review |
| --- | --- |
| Original PR10 source integrity | 110 added files matched Git blob hashes; 45 Python files parsed |
| Three focused authorization/MFA probes | Source-logic failures reproduced with synthetic doubles; not HTTP integration tests |
| New synthetic incident contract | 33 local tests passed on Python 3.13.2; no external services or writes |
| Destructive test database guard | 6 local tests passed without a database connection |
| Documentation and Python checks | 68 relative file links resolved; 48 Python files parsed; root Ruff checks passed |
| Full local ICCC runtime suites | Unverified: required packages unavailable and package-index DNS access failed |
| New CI at updated PR head | Inspect Actions; do not assume success from workflow existence |
| Actual hosted Lovable deployment | Separate system; read-only source reference, not validated by these tests |
| Production security, scale and live detection quality | Not established |

A release record must also include capability parity, dependency/provenance review, reviewed screenshots and known remaining gaps. The completed code-revision CI evidence follows.

## CI evidence for the implemented code

Code revision `92380289331f128fe168552f4094eac1a733fd25` was checked on 15 October 2026 (GitHub's PR merge checkout incorporates main `cb3e3078336b0ea8c5105943f3ff1afd57f373b5`).

- [Research/demo workflow](https://github.com/bikramtheitguy/ai-coastal-security-defense/actions/runs/38694147010): all five jobs passed, including Windows/Linux demo tests, vessel benchmark and locked dependency audit. The Linux demo suite reported 64 passed.
- [ICCC workflow](https://github.com/bikramtheitguy/ai-coastal-security-defense/actions/runs/38694147067): backend suites passed on SQLite and PostgreSQL (53 tests each); frontend typecheck/build passed and all 8 browser tests passed.
- ICCC dependency auditing failed. See [affected packages and remediation gates](DEPENDENCY_REVIEW.md). Existing authorization/MFA/dispatch findings remain open despite behavioral test success.

These results apply to the implemented code revision above. The subsequent documentation-only update records the evidence without changing runtime code. The full local Streamlit suite stalled and was stopped; the completed clean CI jobs provide its validation evidence instead. Full ICCC execution remained unavailable locally because dependency downloads failed.
