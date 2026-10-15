# Streamlit replay validation record

Baseline tree: `7853a4ffbdd3b481710a604c905a2548c3839b69`, identical to main `cb3e3078336b0ea8c5105943f3ff1afd57f373b5`. Development used a separate snapshot of 52 project files verified against Git blob identities; the original dirty working directory and PR #10 were preserved. The historical `venv/` was not used or copied into the verification snapshot.

## Checks for this increment

- Pure scenario tests: **25 passed locally** on Python 3.13.2. Checks cover fixture regeneration, validation, no future evidence, stable gap identity, recovery, distance edge cases, review gates/history and replay clock boundaries.
- Combined local checks excluding the blocked UI/server paths: **51 passed, 1 skipped** (the Bash wrapper runs on Linux CI). This includes the 25 new domain tests and existing asset/helper checks. Ruff lint and formatting pass.
- Streamlit AppTest: regression coverage for all three legacy views plus the new charts, tables, filters, empty states, play/pause/step, module switching, review validation, session history and reset.
- Actual server smoke test: launches `streamlit run dashboard/app.py` in a separate process, checks the HTTP health and HTML endpoints, and terminates the test process.
- Real Chromium: six browser cases exercise the historical views, rendered charts/tables, autoplay/pause/end, review transitions/reset and JSON/CSV/manifest downloads. Tests reject unexpected external HTTP requests and browser JavaScript errors.
- Existing CI runs on Windows and Linux, verifies locked dependency exports, runs the independent vessel benchmark and audits dependencies. No new runtime dependencies or service connections were added.

## Verified CI evidence

All six jobs passed for application/test commit `f0fdae96ae4fac67f1e199660b7cc04a7bad47fd` against unchanged main `cb3e3078336b0ea8c5105943f3ff1afd57f373b5`, on 15 October 2026: [demo/benchmark/audit run 38769794737](https://github.com/bikramtheitguy/ai-coastal-security-defense/actions/runs/38769794737) and [Chromium run 38769794911](https://github.com/bikramtheitguy/ai-coastal-security-defense/actions/runs/38769794911). CI used Python 3.13.16. The subsequent documentation/screenshot commit records this evidence without changing application code, tests, dependencies or workflows.

| Job | Verified result |
| --- | --- |
| Linux demo, job 116542300486 | 60 tests passed, including Streamlit chart/table/control tests and actual server health/HTML startup |
| Windows demo, job 116542300699 | 59 passed, 1 skipped; only the Linux Bash-wrapper test skipped |
| Linux vessel benchmark, job 116542300821 | 20 tests passed and documented default benchmark command succeeded |
| Windows vessel benchmark, job 116542300496 | 20 tests passed and documented default benchmark command succeeded |
| Locked dependency audit, job 116542300090 | Runtime, optional research and installed development audits each reported no known vulnerabilities |
| Linux Chromium, job 116542300712 | 6 cases passed in 26.55 seconds; browser-tool hash-lock verification, lint/format and dependency audit passed |

Both demo jobs installed the locked environment, verified regenerated dependency exports, and passed Ruff lint/format checks. The initial replay implementation also passed [run 38764453239](https://github.com/bikramtheitguy/ai-coastal-security-defense/actions/runs/38764453239). Earlier failures exposed an obsolete test element name and publication newline; both were corrected. No tests were removed or skipped to hide failures.

The proposal adds/modifies files with **no removed paths**. Historical assets, model files, notebooks, benchmark code and application/research dependency locks are unchanged. Browser testing adds a separately hash-locked test environment and read-only CI workflow. A supported-code/config scan found only the intended Lovable showcase hyperlink, and PR #10 remains open, draft and unmerged at `48c5c74dac3b6dd6bfd13fd29d5fb4a02c48b624`.

## Browser findings and visual review

The first Chromium run exposed a real Play/Pause defect: changing session state after rendering the button left its visible label stale while the timer fragment continued. A callback now updates state before rendering; AppTest and real-browser autoplay/pause assertions pass. Tests were also corrected to wait for replaced legacy images, click the visible acknowledgement label, and reopen the review-history expander after reset.

Screenshot inspection led to one replay heading/notice, fitted geographic axes with marker margins, explicit UTC speed labels, two-column legends and sufficient room for units. Pan/zoom and observation tooltips remain available. The five final screenshots were visually inspected for desktop controls/counts, geographic distribution, review evidence, narrow layout, full legend labels and UTC ticks. Screenshots were captured with an Asia/Kolkata browser timezone; the speed chart still displays 00:00–01:00 UTC at the default cursor.

See the [unedited before/after screenshots](STREAMLIT_POC_COMPARISON.md#visual-comparison-within-the-replay-increment) and [reproduction instructions](BROWSER_QA.md). The workflow retains traces, screenshots, server logs, browser version and JUnit results for 14 days. Selected JPEGs are committed for durable review. These are automated Chromium interactions plus visual inspection of their screenshots, not a claimed manual local browser session.

Coverage is Linux Chromium at 1365 × 900 and 390 × 844. Physical phones, other browser engines, accessibility, load/concurrency, persistent identities and a deployed hosted revision remain unverified. A dependency audit is a point-in-time check for known advisories, not a security certification.

## Local limitation and deployment boundary

The local Windows execution environment blocks sockets before Streamlit AppTest or the server can initialise. A diagnostic stack trace placed the hang in `socket._fallback_socketpair` during asyncio event-loop construction; a bounded socket probe also returned `PermissionError [WinError 10013]`. The stalled test/server processes were stopped. Host protections were not disabled. Pure local tests passed, and GitHub Actions supplied the working runtime for UI, server and real-browser checks.

The previously deployed main's three research views were checked earlier on 15 October 2026. That observation does **not** prove the new replay is deployed. No hosted Streamlit deployment, PR merge, Lovable change or connection is performed by this proposal. The browser verification gate for this increment is complete within the scope above; maintainer review and a separately verified hosted deployment remain release gates.

Passing software tests does not establish real-world model performance, operational incident handling, national-scale capacity, identity security or production readiness. See [limitations](LIMITATIONS.md) and the [POC acceptance walkthrough](STREAMLIT_POC.md).
