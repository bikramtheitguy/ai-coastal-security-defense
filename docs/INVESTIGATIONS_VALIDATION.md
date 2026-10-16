# Investigation increment validation

Baseline: PR #15 merged at `e4a383bf79858a38ea4574699fa31a46f57b6532`. That hosted version was checked separately on 16 October 2026. This record concerns the new investigation proposal, not its deployment.

## Local evidence

- 24 casebook domain tests passed, including stale-to-recovered captures, frozen history, transitions, evidence acknowledgement, recording chronology, duplicate/backward captures, limits and export/restore equality.
- Recomputed outer checksums do not permit altered coordinates, rule explanations, future receipts, invalid transitions, sequences, extra fields, fixture mismatch, boolean schema versions or duplicate cases.
- Invalid/deep/oversized JSON, duplicate keys and binary input are rejected. Notes remain plain data; authorship is explicitly unverified.
- Local Streamlit AppTest could not run: this Windows sandbox blocks the loopback socket pair needed to construct its asyncio loop. A captured traceback stops in `socket.accept` / `_fallback_socketpair`, before app execution. No test is skipped or weakened in CI to conceal this limitation.

## CI acceptance (pending)

Run full Windows/Linux demo suites, research regressions, lock/export checks, Ruff and dependency audits. Run real Chromium against `streamlit run dashboard/app.py`, including open/review/recovery/export/reload/restore, invalid uploads, a separate empty session and narrow layout. Preserve all previous browser tests. Inspect screenshots before marking this proposal validated.

No production availability, identity, authorization, real maritime detection, or national-scale capacity is established by these checks. Deployment requires a separate merge decision and hosted verification.
