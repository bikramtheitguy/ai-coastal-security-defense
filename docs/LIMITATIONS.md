# Baseline limitations

## Capability and evidence boundaries

This is an educational artifact viewer. It has no live feeds, vessel trajectories, computer-vision detector, network-event collector, authentication, authorization, operational audit log, incident workflow or AI assistant. Do not upload or substitute restricted operational information.

The three source datasets are described as synthetic by the original README. Their generator, distribution assumptions and label rules are unknown. They cannot establish maritime or intrusion-detection effectiveness.

## Known research defects retained for separate review

- `vessel_phase1_upgrade.py` compares -1/+1 Isolation Forest outputs to 0/1 targets during F1 tuning, then evaluates the refitted estimator on the fitting dataset. No defensible generalization metric is available.
- The notebook uses a different vessel feature set and unseeded estimators. It includes a saved working-directory error and omits a complete cyber experiment.
- Drone preprocessing learns categories before the split; only the estimator is saved. There is no portable encoder/inference contract.
- Cyber IP octets 1 and 2 are constant. Four positives in the saved evaluation table are insufficient for reliable performance estimates.
- `phase1_upgrade.py` writes to a hardcoded `/mnt/data` location and its generated program has a string-literal syntax error.
- Legacy scripts have inconsistent working-directory assumptions and overwrite model/chart outputs.
- Saved prediction tables lack record IDs and probabilities. We can recompute classification counts, but cannot verify their original holdout membership or recompute ROC-AUC.
- Existing pickle artifacts embed scikit-learn 1.4.2 metadata. They are not authenticated by training provenance and are not loaded by this demo.

Historical training code and results remain unchanged. The new [vessel benchmark](VESSEL_BENCHMARK.md) provides a separate corrected evaluation path with deterministic synthetic data, train-only fitting, validation-only selection and held-out test reporting. It does not repair or retroactively validate the original experiment.

## What tests establish

Tests check all three Streamlit views, visible saved-result metrics, launch-directory independence, asset integrity, binary CSV validation, invalid/missing assets, explicit undefined metrics, source-data counts and basic contracts, non-destructive setup, and enabled development protections.

Separate vessel tests check data contracts, deterministic generation, split boundaries, exclusion of labels/metadata from fitting, validation threshold selection, no refitting during evaluation, metric calculations, saved bundle integrity and reproducibility. They include trusted round-trip loading only of models created by the test itself. The app and benchmark CLI never load model files.

They do not establish model quality, dataset realism, deployed-cloud behavior, load capacity, geospatial correctness, or security certification. Dependency audits cover supported locked demo/research dependencies; they do not clear the historical `venv/`, pickle contents, Git history, or the separate ICCC branch.

The new generator invents its anomaly prevalence and speed distributions. Slow normal and anomalous observations overlap; speed alone cannot distinguish them. Its stationary synthetic test set is public and must not be reused as an independent final holdout after results inform development. Operational alert-rate estimates and real-world confidence intervals remain unavailable.

## Repository and release limits

The committed historical `venv/` remains untouched to preserve the requested scope. It contains obsolete libraries and is never a supported runtime. Ignore rules prevent new untracked environment files but cannot remove already tracked ones. Release packaging must allowlist project files and exclude this environment.

There is no operational SLA, validated restore process, approved data integration, or release certification. No production readiness is claimed.
