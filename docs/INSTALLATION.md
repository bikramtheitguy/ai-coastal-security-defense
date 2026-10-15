# Installation and verification

## Supported demo environment

- Python 3.13, using a maintained patch release.
- uv 0.13.1 when using the uv workflow.
- Windows and Linux are covered by CI. macOS is not yet a verified target.
- A fresh isolated environment; never the historical committed `venv/`.

`pyproject.toml` is the dependency source, `uv.lock` records exact resolutions and artifact hashes, and `requirements.txt` is its runtime-only export for pip and Community Cloud. Research and developer tools are excluded from that runtime.

## uv installation

Install the pinned uv release using the [official installation instructions](https://docs.astral.sh/uv/getting-started/installation/), then from the repository root:

```bash
uv --version
uv sync --locked --no-dev
uv run --locked --no-dev streamlit run dashboard/app.py
```

uv creates `.venv/`. If Python 3.13 is unavailable, install it before using the pip route, or allow uv's documented managed-Python installation. A change of interpreter patch version should be accompanied by the test suite.

## pip-only installation

Windows PowerShell, with `python` resolving to Python 3.13:

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --require-hashes -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

Linux:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install --require-hashes -r requirements.txt
.venv/bin/python -m streamlit run dashboard/app.py
```

Activation is optional. Use the environment's interpreter explicitly to avoid installing into another Python environment.

## Development and checks

```bash
uv sync --locked
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest
uv run --locked pip-audit --require-hashes --disable-pip -r requirements.txt
uv run --locked pip-audit --require-hashes --disable-pip -r requirements-research.txt
```

Only the supported demo/helpers/tests are linted. Legacy training scripts are preserved and have known defects; passing demo CI is not a training validation claim. The Bash wrapper is tested on Linux.

## Dependency updates

Use uv 0.13.1. Review direct-version changes in `pyproject.toml`, then:

```bash
uv lock
uv export --locked --no-dev --no-emit-project --output-file requirements.txt
uv export --locked --no-dev --group research --no-emit-project --output-file requirements-research.txt
uv sync --locked
uv run --locked pytest
```

Run lint, formatting and both audits before merging. CI regenerates exports and fails if the checked-in copies differ. Never hand-edit generated hashes or suppress a vulnerability solely to obtain a green check.

## Optional research tools

```bash
uv sync --locked --group research
uv run --locked --group research jupyter notebook notebooks/
```

This installs current exploratory tooling; it does **not** reproduce the original scikit-learn 1.4.2 models. Do not deserialize historical pickle files in the new environment. The vessel scorer, train/evaluation separation, paths and script generator still need repair before a supported training command can be offered. See [limitations](LIMITATIONS.md).

## Streamlit Community Cloud

Select this repository, the approved branch/commit and entry point `dashboard/app.py`. In advanced settings select Python 3.13. The root `requirements.txt` supplies the runtime; `.streamlit/config.toml` keeps CORS/XSRF protection enabled and disables usage statistics.

Record the deployed revision and runtime in release notes, check all three views, and inspect deployment logs. A devcontainer configuration does not configure Community Cloud. This PR does not redeploy or wake the existing hosted demo.

## Devcontainer

The development image uses Python 3.13 and is pinned to an immutable image digest. Creation installs hashed runtime dependencies into `.venv/`; attachment starts Streamlit with CORS and XSRF explicitly enabled. The container itself has not been built or security-scanned as part of local Windows verification; that remains a separate check. This configuration is not a production deployment.

## Troubleshooting

- **Asset integrity/missing-file error:** restore the committed CSV/PNG and `dashboard/artifacts.json` together. Do not bypass the checksum to display mismatched results. New artifacts require provenance, reviewed manifest changes and updated expectations.
- **Wrong interpreter:** verify `python --version` and run the exact environment executable above.
- **Different launch directory:** assets resolve relative to the app. Use an absolute path to `dashboard/app.py` when launching elsewhere. Run from the repository root to load its Streamlit configuration automatically.
- **Cloud demo sleeping:** use the local instructions; sleeping is not evidence that the new revision is deployed.
- **Research scripts fail:** the legacy research path is not part of the supported demo runtime.
- **Setup helpers:** `python setup_project.py` and `bash setup.sh` create missing directories only. They do not replace files, install packages, or launch Jupyter.
