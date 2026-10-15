# Streamlit replay validation record

Baseline tree: `7853a4ffbdd3b481710a604c905a2548c3839b69`, identical to main `cb3e3078336b0ea8c5105943f3ff1afd57f373b5`. Development used a separate snapshot of 52 project files verified against Git blob identities; the original dirty working directory and PR #10 were preserved. The historical `venv/` was not used or copied into the verification snapshot.

## Checks for this increment

- Pure scenario tests: **25 passed locally** on Python 3.13.2. Checks cover fixture regeneration, validation, no future evidence, stable gap identity, recovery, distance edge cases, review gates/history and replay clock boundaries.
- Combined local checks excluding the blocked UI/server paths: **51 passed, 1 skipped** (the Bash wrapper runs on Linux CI). This includes the 25 new domain tests and existing asset/helper checks. Ruff lint and formatting pass. Full CI results will be recorded after verification.
- Streamlit AppTest: regression coverage for all three legacy views plus the new charts, tables, filters, empty states, play/pause/step, module switching, review validation, session history and reset.
- Actual server smoke test: launches `streamlit run dashboard/app.py` in a separate process, checks the HTTP health and HTML endpoints, and terminates the test process.
- Existing CI runs on Windows and Linux, verifies locked dependency exports, runs the independent vessel benchmark and audits dependencies. No new runtime dependencies or service connections were added.

## Local limitation and deployment boundary

The local Windows execution environment hangs in Python's socket-pair setup before Streamlit AppTest or the server can initialise. A diagnostic stack trace placed the block in `socket._fallback_socketpair` during asyncio event-loop construction, before application code execution. The stalled test/server processes were stopped. The pure calculation tests do not need networking and passed. The actual Streamlit behavior must therefore be verified on clean CI runners; there is no claim of a successful local browser session for this increment.

The previously deployed main's three research views were checked earlier on 15 October 2026. That observation does **not** validate this new replay or prove it is deployed. No hosted Streamlit deployment, PR merge, Lovable change or connection is performed by this proposal. Browser appearance, narrow-screen layout and real-browser autoplay require a subsequent visual verification on an available runtime before a release.

Passing software tests does not establish real-world model performance, operational incident handling, national-scale capacity, identity security or production readiness. See [limitations](LIMITATIONS.md) and the [POC acceptance walkthrough](STREAMLIT_POC.md).
