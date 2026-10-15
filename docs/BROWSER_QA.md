# Streamlit browser verification

`browser_tests/` complements domain tests and Streamlit AppTest with real Chromium. It starts `dashboard/app.py` on loopback in a separate process, uses a fresh browser context per test, and exercises invented data only. It does not deploy the app or visit the external showcase link.

## Isolated, reproducible tooling

The Streamlit runtime remains defined by `pyproject.toml` and `uv.lock`. Browser tooling uses a separate `.browser-venv`. `browser_tests/requirements.in` pins Playwright 1.64.0 and pytest 9.1.1; `requirements.txt` in that directory is their generated transitive hash lock. Python installs enforce hashes. Chromium is selected by the pinned Playwright distribution; its runtime version is recorded with the evidence. Browser binaries and Linux system packages are outside the application's runtime lock.

Linux, from the repository root:

```bash
uv sync --locked
uv venv .browser-venv --python 3.13
uv pip sync --python .browser-venv/bin/python --require-hashes browser_tests/requirements.txt
.browser-venv/bin/python -m playwright install --with-deps chromium
STREAMLIT_PYTHON="$PWD/.venv/bin/python" .browser-venv/bin/python -m pytest browser_tests -v
```

Windows PowerShell, where loopback sockets are permitted:

```powershell
uv sync --locked
uv venv .browser-venv --python 3.13
uv pip sync --python .browser-venv/Scripts/python.exe --require-hashes browser_tests/requirements.txt
.browser-venv/Scripts/python.exe -m playwright install chromium
$env:STREAMLIT_PYTHON = (Resolve-Path .venv/Scripts/python.exe).Path
.browser-venv/Scripts/python.exe -m pytest browser_tests -v
```

Do not disable host protections when loopback access is denied. GitHub Actions provides a separate Linux test environment with read-only repository permissions. It uploads evidence, never commits code or deploys the app.

## Coverage and evidence

- Three historical views: headings, decoded chart images and saved metrics.
- Replay: rendered position/speed charts and tables, seek/freshness boundaries, wall-clock autoplay, pause stability and end behavior.
- Review: acknowledgement/rationale validation, status transitions, downloaded JSON and reset.
- Narrow layout: controls, plots, page overflow and synthetic CSV/manifest downloads.
- Isolation: unexpected external HTTP requests and browser JavaScript errors fail tests.

Desktop size is 1365 × 900; narrow size is 390 × 844. Viewport emulation is not physical-phone validation. Firefox, Safari/WebKit, assistive technology, concurrent reviewers and operational workloads remain outside this check.

The workflow uploads `streamlit-browser-evidence`: screenshots, traces, visible text, server logs, browser version and JUnit results, retained for 14 days. Bounded JPEG previews of the synthetic UI are also emitted into its log for remote visual inspection. Download needed bundles before expiry; these are test evidence, not release archives.

Refresh the lock deliberately, review its diff, and audit before accepting:

```bash
uv pip compile --python-version 3.13 --universal --generate-hashes browser_tests/requirements.in --output-file browser_tests/requirements.txt
uv run --locked pip-audit --require-hashes --disable-pip -r browser_tests/requirements.txt
```

See [validation results](STREAMLIT_VALIDATION.md). Passing tests on synthetic examples does not establish model quality, operational readiness or a deployed revision.
