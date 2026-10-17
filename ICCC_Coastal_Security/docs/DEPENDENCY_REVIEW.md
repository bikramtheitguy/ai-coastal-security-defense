# Dependency audit findings

The first ICCC CI audit failed on 15 October 2026 for code revision `92380289331f128fe168552f4094eac1a733fd25`. The [audit job](https://github.com/bikramtheitguy/ai-coastal-security-defense/actions/runs/38694147067/job/116286678602) contains the exact scanner output. The application build and behavioral tests passed separately; they do not invalidate dependency advisories.

| Ecosystem | Scanner-reported affected package | Review action |
| --- | --- | --- |
| Python | PyJWT 2.10.1 | Review signing/verification advisories, patched compatible release and token regression tests |
| Python | python-multipart 0.0.20 | Review upload/parser advisories and bounded-upload regressions |
| Python | Starlette 0.46.2 | Upgrade through a compatible FastAPI/Starlette pair; retest routes, auth, uploads and static serving |
| npm | maplibre-gl 5.24.0 | Critical sanitizer advisory reported; test a patched version without removing map capabilities |
| npm | next, postcss, sharp, source-map-js | Review direct/transitive advisories, static-export applicability and patched lockfile resolution |

`pip-audit` reported 47 advisory rows across three packages, including repeated identifiers; this is not a count of 47 independently verified exploits. The npm report summarized five affected packages: one critical, three high and one moderate. Exploitability in this application needs review, especially for build-time versus runtime paths and static export, but no exception or suppression has been approved.

## Remediation acceptance

1. Review advisory details and affected paths, selecting compatible patched versions rather than running a blind forced upgrade.
2. Regenerate dependency locks with integrity hashes in a network-enabled development environment. Preserve all map, auth, upload and workflow capabilities.
3. Re-run SQLite/PostgreSQL API tests, TypeScript/build, browser acceptance and the dependency audit at the updated revision. Add a focused regression test where an affected application path needs one.
4. Record residual advisories and any justified exception with evidence, owner and expiry. Never turn off the audit merely to make CI green.

The original Python requirements pinned only top-level versions. The remediation below adds transitive locks. A dependency/license inventory remains required; dependency patching alone does not establish operational security. DEP-01 is a P0 release blocker in [ROADMAP.md](ROADMAP.md).

## Selected remediation (15 October 2026)

| Package | Selected version | Reason and verification gate |
| --- | --- | --- |
| FastAPI / Starlette | 0.143.1 / 1.8.0 | Compatible declared requirements; API, multipart and static-serving tests on both databases |
| PyJWT | 2.15.1 | Patched release; explicit HS256/required claims, signature, expiry, malformed subject and session-binding regressions |
| python-multipart | 0.0.32 | Patched parser; byte/hash/download round-trip and oversize rejection tests |
| MapLibre GL JS | 6.14.0 | Sanitizer fix requires 6.4.1 or later; ESM import migration, offline layers/context panels and attribution regression |
| Next.js | 15.5.28 | Stay on the existing major; typecheck, static export and browser acceptance |
| PostCSS / sharp / source-map-js | 8.5.30 / 0.35.5 / 1.2.2 | Explicit npm overrides prevent resolution back to affected transitive versions |

The Next application uses static export and unoptimized images, so server-side Next image/cache paths are not served by its FastAPI deployment. The development server and build dependencies still need patching. Overrides are temporary maintenance constraints: remove them only when upstream resolution is patched and regression/audit checks pass. No feature is disabled and no advisory is suppressed. MapLibre 6 requires WebGL2; see the installation compatibility note.

Python runtime/development locks are generated with uv 0.13.1, universal markers, SHA-256 hashes, and a shared runtime constraint. Installers enforce those hashes. npm's generated lock retains package integrity hashes. The lock proposal is tied to its Actions run and source revision; it is reviewed before committing and has no write/deployment permission.

Upstream evidence: [FastAPI dependency metadata](https://github.com/fastapi/fastapi/blob/0.143.1/pyproject.toml), [PyJWT release](https://pypi.org/project/PyJWT/2.15.1/), [multipart release](https://pypi.org/project/python-multipart/0.0.32/), [MapLibre sanitizer advisory](https://github.com/maplibre/maplibre-gl-js/security/advisories/GHSA-jrc7-96c5-q579), [MapLibre migration changes](https://github.com/maplibre/maplibre-gl-js/blob/v6.4.1/CHANGELOG.md), [Next dependency metadata](https://github.com/vercel/next.js/blob/v15.5.28/packages/next/package.json), [PostCSS releases](https://github.com/postcss/postcss/releases), [source-map-js fix](https://github.com/7rulnik/source-map-js/releases/tag/v1.2.2).

The locked dependency audit passed on 15 October 2026 at code revisions `cc4eaa86c5274f2b3fb638598c0c58dc041c8739` and `9ce7eb14dd261fce1811a4491a256b0271a36bec`: Python reported no known vulnerabilities and npm reported zero. These are Linux CI runtime-dependency scans against the databases available at that time, not a security certification. Final application/browser evidence is recorded in [VALIDATION.md](VALIDATION.md). No advisory was suppressed.

MapLibre worker packaging is part of the migration: Next copies versioned ESM modules and the upstream license from the locked npm installation into the static export. Browser regressions verify the existing offline layers and context panels and check safe attribution text/link retention with event-attribute removal.
