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

The Python requirements currently pin top-level versions only. Transitive locking and a dependency/license inventory remain required. This document records findings; it does not claim patched dependencies or operational security. DEP-01 is a P0 release blocker in [ROADMAP.md](ROADMAP.md).
