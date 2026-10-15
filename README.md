# Coastal Analytics Research Demo

A Python and Streamlit research demonstrator for exploring **saved synthetic-data experiments** in vessel anomalies, drone observations, and cybersecurity classification.

**Status: experimental. Not validated for operational decisions or production use.** The dashboard reads committed CSVs and PNGs; it does not run inference, receive live feeds, or control equipment.

[Hosted demo](https://ai-coastal-security-defense.streamlit.app/) · [Installation](docs/INSTALLATION.md) · [Architecture](docs/ARCHITECTURE.md) · [Limitations](docs/LIMITATIONS.md) · [Security](SECURITY.md)

The hosted demo may sleep. This repository does not establish which revision is currently deployed.

## What exists

| Module | Implemented | Limitations |
| --- | --- | --- |
| Vessel anomalies | Offline Isolation Forest research code and a saved geographic scatterplot | Invalid tuning-label mapping and in-sample evaluation; one observation per vessel, no timestamps or tracking |
| Drone observations | Offline Random Forest on supplied object labels and confidence scores | No image/video detector; weak saved positive-class results |
| Cybersecurity | Offline Random Forest on dummy packet records | No live monitoring; saved predictions miss all four positive cases |
| Dashboard | Three views, validated saved assets, recomputed metrics, clear errors | An artifact viewer, not an operational system |

LOF, Logistic Regression, SVM, live AIS, geospatial replay, autonomous surveillance, and AI-assisted incident analysis are **not implemented on this baseline**. The separate [ICCC platform proposal in PR #10](https://github.com/bikramtheitguy/ai-coastal-security-defense/pull/10) is unmerged work and is not part of this demo.

## Run the demo

Use Python **3.13** and [uv](https://docs.astral.sh/uv/getting-started/installation/) **0.13.1**.

```bash
git clone https://github.com/bikramtheitguy/ai-coastal-security-defense.git
cd ai-coastal-security-defense
uv sync --locked --no-dev
uv run --locked --no-dev streamlit run dashboard/app.py
```

Open the local URL printed by Streamlit. No credentials or external data services are required. See the [pip-only instructions](docs/INSTALLATION.md) for Windows and Linux, and the explicit Community Cloud settings.

Do not activate the committed `venv/` directory. It is a historical Linux/Python environment retained with the original repository; supported installations use a fresh `.venv/`.

## Saved results, recalculated

Class 1 is the positive class. These figures describe the committed prediction files, **not independently reproduced model runs**.

| Experiment | Rows | TP | FP | FN | TN | Precision | Recall | F1 | Accuracy | Always-negative accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Drone | 200 | 2 | 8 | 14 | 176 | 20.0% | 12.5% | 0.154 | 89.0% | 92.0% |
| Cyber | 200 | 0 | 1 | 4 | 195 | 0.0% | 0.0% | 0.000 | 97.5% | 98.0% |

Both saved classifiers fall below an always-negative accuracy baseline. Accuracy alone obscures missed positives. The cyber sample has only four positive cases; these results cannot estimate operational reliability.

The [artifact manifest](dashboard/artifacts.json) records SHA-256 digests of the original CSVs and charts. Tests verify integrity and metric consistency. Original training split IDs, scores, labeling rules, generator code and seeds were not saved. Checksums do not recover that provenance.

## Repository map

```text
dashboard/               Streamlit viewer, result helpers, saved artifacts and manifest
data/                    Original datasets described as synthetic
models/                  Original serialized research models; never loaded by the demo
notebooks/               Legacy experiments and notebook outputs; known limitations remain
tests/                   Demo, data-contract, integrity and setup tests
docs/                    Installation, architecture, limitations and provenance
.github/workflows/       Demo checks and dependency auditing
pyproject.toml            Runtime, development and optional research dependency groups
uv.lock                  Universal dependency lock
requirements.txt         Hashed runtime export for pip and Streamlit Community Cloud
requirements-research.txt Hashed optional research export
venv/                    Historical environment; retained, unsupported and unused
```

## Development

```bash
uv sync --locked
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest
```

CI checks Windows and Linux on Python 3.13, verifies dependency exports, and audits runtime and optional research dependencies. CI results describe the demo baseline, not the scientific validity of the legacy models. See [verification scope](docs/LIMITATIONS.md).

## Next milestones

1. Repair vessel scoring and introduce untouched evaluation data.
2. Document or regenerate deterministic synthetic data with explicit labels and provenance.
3. Extract reusable training pipelines; record splits, parameters, probabilities and model metadata.
4. Add geospatial replay and approved-source adapters.
5. Evaluate genuine computer vision and event-monitoring pipelines.
6. Introduce evidence-linked AI assistance with human review.

Each milestone needs separate validation. No automatic enforcement or operational decisions are provided.

## License and attribution

Project code is distributed under [MIT](LICENSE). Dataset provenance remains incomplete; the code license is not a substitute for per-asset rights verification. See [data provenance](docs/DATA_PROVENANCE.md) and [third-party notices](THIRD_PARTY_NOTICES.md).
