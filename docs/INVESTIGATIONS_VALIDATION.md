# Investigation increment validation

Baseline: PR #15 merged at `e4a383bf79858a38ea4574699fa31a46f57b6532`. That hosted version was checked separately on 16 October 2026. This record concerns the new investigation proposal, not its deployment.

## Local evidence

- 25 casebook domain tests passed, including stale-to-recovered captures, frozen history, transitions, evidence acknowledgement, recording chronology, duplicate/backward captures, limits, multi-case isolation and export/restore equality.
- Recomputed outer checksums do not permit altered coordinates, rule explanations, future receipts, invalid transitions, sequences, extra fields, fixture mismatch, boolean schema versions or duplicate cases.
- Invalid/deep/oversized JSON, duplicate keys and binary input are rejected. Notes remain plain data; authorship is explicitly unverified.
- Local Streamlit AppTest could not run: this Windows sandbox blocks the loopback socket pair needed to construct its asyncio loop. A captured traceback stops in `socket.accept` / `_fallback_socketpair`, before app execution. No test is skipped or weakened in CI to conceal this limitation.

## CI acceptance

Verified on 16 October 2026 against runtime/test revision `6ae9cafc608f6282bb8eb0216ad8d5b2c2ada283`, with main still at the baseline above. Later documentation/screenshot commits do not change that runtime. PR #16 remains an unmerged proposal.

| Check | Result | Evidence |
| --- | --- | --- |
| Linux demo suite, including Streamlit AppTest | 117 passed | [Demo baseline run](https://github.com/bikramtheitguy/ai-coastal-security-defense/actions/runs/38846400555) |
| Windows demo suite | 116 passed, 1 skipped | Same run; existing Bash-wrapper test runs on Linux only |
| Linux and Windows research regressions | 20 passed on each; documented default benchmark completed | Same run |
| Locked exports, Ruff and dependency audits | Passed; audits reported no known vulnerabilities in runtime, research and installed development tools | Same run; not a security certification |
| Chromium desktop and narrow layouts | 12 passed | [Browser QA run](https://github.com/bikramtheitguy/ai-coastal-security-defense/actions/runs/38846400557) |
| Browser tooling lint and dependency audit | Passed | Same browser run |

The browser suite starts the actual Streamlit application on loopback, blocks external requests, and fails on JavaScript page errors or console errors. All ten previous browser checks remain; two new checks cover case workflow and invalid uploads. Screenshots are from this CI preview, not the hosted application.

### Verified investigation scenario

1. Open the Odisha exercise vessel SIM-004 at minute 60; acknowledgement is required and a rejected submission retains draft notes.
2. Freeze its stale-report evidence and record **In review**.
3. Advance to minute 95 and append recovery evidence. The first capture remains 35 minutes stale; the second has a five-minute-old observation and no current flag. The case stays **In review**.
4. Record **Closed — benign**. Download four ordered actions and two captures; verify the export checksum and the handover's observation reference.
5. Confirm another browser session is empty. Reload, upload the downloaded casebook, validate its preview and restore the closure/history.
6. Render at 390 × 844 with no page-level horizontal overflow. Tables retain their own horizontal scrolling. Reject duplicate-key JSON without creating a case or offering restoration.

AppTest additionally verifies persistence across module/filter changes and backward clock seeks. Domain tests recompute source snapshots during import and reject resealed evidence tampering. Browser actions wait for the expected recorded-action count and the completed Streamlit run, avoiding stale success notices and partially replaced widgets. No failed assertion was removed to obtain the passing result.

### Visual and bounded performance review

Reviewed the desktop case queue, evidence comparison/receipt timeline, and restored mobile case screenshots. The existing spatial palette and controls are retained; forms, notices, labels and timeline render without overlap. The mobile case table scrolls independently.

A local synthetic timing check restored 12 cases with 60 events and **one capture per case**, a 775,911-byte export, in approximately 3.8 seconds after replay was isolated per case (previously approximately 28 seconds). This is one measured fixture, not a worst-case six-capture benchmark, a latency guarantee or national-scale capacity evidence. Limits remain 12 cases, six captures per case, 60 events per case and 8 MiB.

No production availability, identity, authorization, real maritime detection, or national-scale capacity is established by these checks. Deployment requires a separate merge decision and hosted verification.
