# Security policy

## Scope and supported use

This repository is an experimental synthetic-data demonstrator, not an operational security product. Supported paths are the documented clean demo environment and offline synthetic vessel benchmark. There are no security-supported stable releases yet.

The historical `venv/`, serialized models, legacy training workflows and unmerged platform branch are outside the validated demo path. Do not expose Jupyter or reuse committed environments as a service.

## Reporting a vulnerability

Use GitHub's private vulnerability reporting feature if enabled for this repository. If it is unavailable, open an issue requesting a private reporting channel **without exploit details, credentials or sensitive data**; the maintainer must arrange a private channel before receiving the report. No response-time guarantee is currently established.

Include affected revision, component, expected/actual behavior and a minimal synthetic reproducer through the private channel. Do not test against live operational systems.

## Development controls

- Use the locked environment and review dependency audit results.
- Keep CORS and XSRF protection enabled; do not solve deployment issues by disabling them.
- Store deployment credentials outside Git. `.env` and Streamlit secrets files are ignored.
- Use only synthetic or explicitly approved public demo data.
- Never deserialize arbitrary uploaded/downloaded pickle or Joblib files. The demo and benchmark CLI do not load model artifacts. The benchmark writes a newly fitted Joblib pipeline; treat it as executable pickle content and load only trusted local output in its recorded environment. Tests load only models they just trained themselves.
- Check provider/data/model licenses and provenance before adding integrations or assets.
- Keep CI tokens read-only for untrusted PR checks and pin Actions references to immutable commits.
- Do not run legacy training scripts against important files: they can overwrite historical outputs.

Checksums protect against accidental artifact drift; they do not authenticate an untrusted repository or prove scientific validity. Passing dependency scans does not constitute a penetration test, historical secret scan or production certification.
