# Vessel benchmark validation record

Assessment: **share with caveats as a synthetic software benchmark**. These results do not validate maritime threat detection or production readiness.

## Reproduction context

Run the [documented default command](VESSEL_BENCHMARK.md) with 3,000 rows, seed 42, and a validation false-positive-rate budget of 0.05. This record was calculated on 2026-10-15 with Python 3.13.2, NumPy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1 and Joblib 1.6.0. The bundle records full versions and source hashes.

Generator: `vessel-speed-synthetic-v1`. Dataset SHA-256: `4924b0e2d52ae28d248e38cb80ea0751a482d633b6a404d37bc11ceffef08ff1`.

| Partition | Observations | Positive labels | UTC interval |
| --- | ---: | ---: | --- |
| Training | 1,800 | 125 | 2025-01-01 00:00 to 2025-01-02 05:59 |
| Validation | 600 | 58 | 2025-01-02 06:00 to 2025-01-02 15:59 |
| Test | 600 | 44 | 2025-01-02 16:00 to 2025-01-03 01:59 |

Dates and prevalence are synthetic. There are zero null values, duplicate observation IDs or duplicate vessel IDs. The complete dataset has 2,438 ordinary-transit, 335 ordinary-slow, 129 synthetic-fast-deviation and 98 synthetic-slow-deviation observations. All partitions come from the same stationary generating process; the chronological ordering does not demonstrate temporal generalization.

## Held-out results

Validation selected 100 trees, 128 maximum samples and an anomaly-score threshold of approximately 0.627762. The forest fits only the training speed column. Class 1 is anomalous; thresholded scores are not probabilities.

| Test detector | TP | FP | FN | TN | Precision | Recall | F1 | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation-selected threshold | 20 | 7 | 24 | 549 | 74.07% | 45.45% | 0.5634 | 94.83% |
| Same forest, default threshold | 42 | 117 | 2 | 439 | 26.42% | 95.45% | 0.4138 | 80.17% |
| Always normal | 0 | 0 | 44 | 556 | Undefined | 0.00% | 0.0000 | 92.67% |

The selected threshold produces a 1.26% false-positive rate (7/556 negative observations) and 11.67 false alerts per 1,000 total observations (7/600 × 1,000). It detects 19/22 fast deviations but only 1/22 slow deviations. Five of its seven false alerts occur among the 72 ordinary-slow observations. These small synthetic slices describe this run only.

Ranking metrics for this forest are average precision 0.6666 and ROC-AUC 0.9167. The constant-score average-precision baseline is test prevalence, 0.0733. These ranking results coexist with substantial misses at the selected threshold. They do not imply calibrated probabilities or reliable incident detection.

## Verification and interpretation

- Confusion counts, precision, recall, F1, average precision and ROC-AUC were independently recomputed from the written prediction CSV using scikit-learn metrics and agreed with the report.
- Tests exercise generation repeatability, malformed data, split isolation, train-only fitting, exclusion of labels/metadata, test-label perturbation without prediction changes, no evaluation refit, threshold budgets, zero denominators, file hashes and trusted pipeline round trips.
- The selected threshold trades fewer false alerts for more missed anomalies than the default threshold. Selecting settings after inspecting this test set would invalidate its role as a fresh final evaluation.
- Ordinary and anomalous slow speeds share a generating distribution. This is an intentional feature-information limit, not a data-cleaning defect. Contextual features and approved representative data are needed to investigate that limitation.
- The reported 5% validation budget is an experiment setting. It is not a validated operational tolerance, and validation compliance does not guarantee compliance on future data.

## Remaining gaps

No representative maritime data, independent external holdout, realistic trajectories, event consolidation, alert-rate exposure denominator, drift study or real-world uncertainty assessment is available. Historical vessel, drone and cyber artifacts are not regenerated or validated by this benchmark. No private project-chat content or operational records are included in the implementation or this report.
