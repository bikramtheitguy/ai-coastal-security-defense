# Reproducible vessel-speed benchmark

This benchmark tests a reproducible machine-learning workflow on **invented synthetic data**. It does not establish the ability to identify maritime threats. It is separate from the historical vessel plot displayed by Streamlit.

See the [default-run validation record](VESSEL_VALIDATION.md) for observed results, independent calculation checks and the false-alert/recall tradeoff.

## Run

From the repository root, using Python 3.13 and uv 0.13.1:

```bash
uv sync --locked --group research
uv run --locked --group research python -m coastal_security.vessel --output artifacts/vessel-demo
uv run --locked --group research pytest training_tests
```

The output directory must not already exist. Choose a new directory for another experiment; existing results are never overwritten. An interrupted run may leave a partial directory. Only a bundle with `manifest.json` marked `complete: true` completed successfully. Original files under `data/`, `models/`, `notebooks/` and `dashboard/` are not rewritten.

Optional arguments are `--rows` (1,000–100,000, default 3,000), `--seed` (default 42), and `--max-fpr` (default 0.05). Declare experiment settings before examining test results. The false-positive budget is an illustrative research parameter, not a recommended operational threshold.

## Data contract and provenance

Generator version: `vessel-speed-synthetic-v1`. All generated data is created by repository code under the project MIT license; no external data or operational records are used.

Each row is one fictional observation of a distinct fictional vessel. Columns are `record_id`, `vessel_id`, `observed_at`, `speed_knots`, `scenario`, and `is_anomalous`. IDs and UTC timestamps must be unique. Nulls, nonfinite speeds, speeds outside the benchmark contract of 0–60 knots, and nonbinary labels are rejected. Repeated-vessel trajectories are deliberately unsupported and require a different grouped evaluation design.

The seeded generator uses these **invented distributions**, not maritime estimates:

- Each observation has an 8% probability of a synthetic anomalous scenario.
- Ordinary transit speed follows a normal distribution with mean 12 and standard deviation 3 knots, clipped to 0–24 knots. Twelve percent of ordinary observations instead use a uniform 0–4-knot distribution.
- Sixty percent of anomalous observations use a normal distribution with mean 28 and standard deviation 6 knots, clipped to 10–50 knots. The other forty percent use the same uniform 0–4-knot distribution as ordinary slow observations.
- Speeds are rounded to six decimal places. Dates start at 2025-01-01 00:00 UTC with one fictional observation per minute. This timestamp sequence is a split-ordering device, not an estimate of observation frequency or traffic exposure.

The `scenario` and label record which generating process was used. Neither is a model feature. The overlapping slow distributions intentionally demonstrate ambiguity: speed alone cannot tell whether a slow observation belongs to the ordinary or anomalous scenario.

## Evaluation protocol

1. Sort by timestamp and divide into 60% training, 20% validation and 20% test. Membership is independent of labels. Require both classes in each partition, disjoint vessel/record IDs and strictly increasing periods.
2. Fit a scikit-learn pipeline on **training speed only**. A column selector preserves the feature contract; no scaling or imputation is needed for this one-feature experiment. All other columns are excluded.
3. Search 100/200 trees and 128/256 maximum samples. Use a fixed model seed, one fitting worker and `contamination="auto"`.
4. Calculate anomaly scores as negative `IsolationForest.score_samples`, so larger scores mean more anomalous observations. These scores are **not probabilities**.
5. For each candidate, choose a validation threshold maximizing positive-class F1 while keeping validation `FP / (FP + TN)` at or below the declared budget. Include a no-alert fallback. Within a model, ties prefer fewer false alerts, then the higher threshold. Across models, ties prefer fewer false alerts, fewer trees, then fewer samples.
6. Freeze the selected pipeline and threshold. Evaluate test data without refitting, threshold adjustment or parameter selection. Report any test false-positive-rate budget exceedance; a validation constraint does not guarantee the same test rate.

Isolation Forest's default `-1` outlier label is mapped to class `1`; `+1` becomes `0`. The report also evaluates that default detector threshold and an always-normal baseline. There is no invalid comparison between `-1/+1` predictions and `0/1` targets.

The detector fits without labels, but **model and threshold selection use validation labels**. This is an unsupervised detector with supervised validation, not a fully label-free workflow. The selected estimator is not refitted on training plus validation, since doing so would change the score distribution used to choose the threshold.

## Metrics and interpretation

`report.json` records TP, FP, FN, TN, precision, recall, F1, accuracy, false-positive rate, miss rate, alert frequency, average precision and ROC-AUC. Class 1 is the positive class. Zero-denominator metrics are JSON `null`, rather than invented zeroes.

False-positive rate divides false alerts by **negative observations**. False alerts per 1,000 observations divides by **all observations**. These are different measures. Average precision is reported alongside positive-class prevalence as its constant-score baseline. Scenario slices expose failures that an aggregate metric can hide; their small sample sizes limit interpretation.

Alerts per hour/day, incident-level detection, geographical validity and real-world confidence intervals are unavailable. There is no actual traffic exposure, event consolidation or representative maritime population. All partitions use the same stationary generator; chronological separation alone is not a drift test.

## Experiment bundle

| File | Evidence |
| --- | --- |
| `dataset.csv` | Exact generated observations and synthetic labels |
| `splits.csv` | Record/vessel IDs, timestamps and partition membership |
| `predictions.csv` | Test observations, scores and both detector predictions |
| `validation_search.csv` | Every candidate's selected threshold and validation metrics |
| `report.json` | Configuration, split counts, selection, test metrics and limitations |
| `model.joblib` | Newly fitted feature pipeline, feature names and selected threshold |
| `MODEL_CARD.md` | Human-readable scope, results, limitations and reproduction notes |
| `manifest.json` | File hashes, generator/seed, normalized source hashes, lock hash and runtime versions |

Source hashes identify the executed code even in an uncommitted working tree; the recorded Git commit is context, not a claim that all files came from that commit. CSV output uses LF line endings. Repeated runs in the same locked environment are tested for identical generated artifacts. Binary models are not promised to be byte-identical across platforms or dependency changes.

`model.joblib` uses pickle-based serialization. Only deserialize output that you just created and trust, in its recorded environment. Hashes detect a mismatch but do not authenticate a maliciously replaced model and manifest. The CLI and Streamlit app never load model files. Historical pickles remain unsupported.

## What is verified and what remains

Tests verify deterministic generation, data contracts, split isolation, train-only fitting, feature exclusion, correct label mapping, threshold constraints, no refitting during evaluation, metric agreement with independent scikit-learn calculations, bundle hashes, safe output handling and round-trip behavior of newly trained pipelines.

This is a public software benchmark. Once its test results have informed development decisions, it cannot serve as a fresh independent final holdout. Further scientific claims need separately approved representative data, predefined acceptance criteria, vessel/time/scene separation, uncertainty estimates and prospective false-alert evaluation. Legacy drone/cyber training and the original vessel experiment remain separate follow-up work.

Implementation follows the official [Isolation Forest score/label contract](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html), [leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage), and [model persistence security guidance](https://scikit-learn.org/stable/model_persistence.html).
