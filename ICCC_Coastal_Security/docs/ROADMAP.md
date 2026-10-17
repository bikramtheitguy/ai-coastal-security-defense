# Prioritized implementation backlog

Use this backlog with the [development process](DEVELOPMENT.md). Priorities reflect dependencies and risk; delivery dates follow only after scope, owners and acceptance are agreed. Preserve existing capabilities throughout.

| ID | Priority | Work and acceptance | Status |
| --- | --- | --- | --- |
| POC-01 | P0 | Document current/proposed functionality, one demo flow, review findings and stage gates | Prepared in this PR |
| DATA-01 | P0 | Local synthetic incident contract: jurisdiction membership, original target preserved on forwarding, source/received times, coordinate quality and human-review state; negative tests | Implemented in this PR; see root contract docs and CI |
| QA-01 | P0 | Run backend SQLite/PostgreSQL, frontend build/typecheck and browser suite at the exact PR revision; preserve baseline research tests | CI verified at `242855b`: 81 tests per database, 9 browser tests, all root jobs pass; see validation record |
| SEC-01/02 | P0 | Consistent object policy for incidents/conversations/evidence/orders, including cancellation; authorized actions still work | Open |
| SEC-03 | P0 | Reauthenticate MFA replacement and retain the active factor until replacement confirmation | Open |
| SEC-05 | P0 | OTP lockout and timezone-correct token lifetime, required token claims and session binding | Implemented; API regressions pass on both databases |
| WF-01 | P0 | Dispatch promise requires attributable confirmed action; test direct/manual and field-order paths | Open |
| SEC-04 | P0 | Loopback demo deployment, explicit non-demo guard, generated secrets and test DB isolation | Loopback/test guards in this PR; runtime fail-closed mode open |
| DEP-01 | P0 | Compatible patched dependencies and Python transitive lock/hashes; preserve map behavior | Patched and clean runtime audit; see [dependency findings](DEPENDENCY_REVIEW.md) and [validation](VALIDATION.md) |
| DEP-02 | P1 | Full dependency/license and third-party attribution inventory | MapLibre worker license preserved; full inventory open |
| DEMO-01 | P1 | Reproducible local walkthrough, synthetic screenshots, keyboard/mobile checks, failure/empty states and capability parity | Open after P0 security fixes |
| DATA-02 | P1 | Import template and source register for station/asset/personnel data; validation report before persistence; no operational import by default | Planned |
| GEO-01 | P1 | Time/bbox/pagination contracts, coordinate-quality legends, stale tracks and replay fixtures; load profile | Planned |
| API-01 | P1 | Apply reviewed incident contract through authenticated API with idempotency and outbox; wrong-scope denial tests | Planned; DATA-01 alone does not provide this |
| ML-01 | P1 | Trajectory-aware vessel baseline and held-out per-detector false-alert/miss metrics | Planned |
| CV-01 | P2 | Approved image/video dataset, leakage-safe split, CV baseline and false-positive/negative review | Planned |
| CYBER-01 | P2 | Approved event schema, event-window evaluation and monitored security-event adapter | Planned |
| AI-01 | P2 | Evidence-linked advisory summaries, privacy evaluation, human review and deterministic fallback | Planned |
| OPS-01 | P1 | Restore drill, audit anchoring, bounded uploads, observability and incident/runbook ownership | Open |
| INT-01 | P2 | Approved maritime/weather/registry adapters with licensing, data-owner agreement and failure tests | Planned; no external integration assumed |
| SCALE-01 | P3 | Agreed capacity/SLO profile, jurisdiction isolation, shared storage, worker leadership, failover and measured recovery | Planned; no national-readiness claim |

P0 items gate a controlled demonstration release. A green build alone does not close security or data-provenance findings. Preserve the Lovable website as a read-only reference for this repository effort; its roadmap is not proof that matching GitHub capabilities exist.

## Next implementation PR

After this foundation, address SEC-01/02 as one focused authorization change: shared object-scope helpers, policy for unassigned/new incidents, list/detail/mutation/evidence/chat/order coverage, and both positive and negative API tests. Resolve policy ambiguities in the design before changing permissions. Keep all legitimate workflows available.

MFA reenrollment and dispatch confirmation follow as separate reviewable changes. OTP lockout and token/session regressions were delivered with dependency remediation. Do not merge an all-at-once redesign or remove features to make a test pass.
