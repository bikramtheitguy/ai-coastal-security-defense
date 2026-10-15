# POC comparison: PR14 baseline and India Coastal Watch

| Capability | PR14 baseline | Current proposed increment |
|---|---|---|
| Research views | Drone, vessel and cyber experiments | Preserved |
| Scenario replay and trajectory evaluation | Synthetic replay and benchmark views | Preserved |
| Geographic context | Schematic synthetic trajectory chart | Public coastline, 13 coastal states/UTs and dated district reference outlines |
| Area selection | Benchmark/replay filters | Coast and state/UT cohort; district map focus |
| Traffic | Synthetic benchmark scenarios | Adds 44 fictional watch vessels across Indian coastal groups |
| Observation handling | Scenario timestamps | Separate observation and receipt times; late/out-of-order evidence retained |
| Data quality | Experiment checks | Duplicate/conflicting/invalid/future payload quarantine and count reconciliation |
| Map semantics | Replay samples | Last-known positions, freshness, observation dots without interpolated outage paths |
| Analyst review | Benchmark evidence | Current rule explanations linked to evidence IDs; snapshot export with integrity hash |
| Provenance | Synthetic benchmark protocol | Pinned public-source hashes, third-party licences, transformations and known gaps |
| Aircraft / context feeds | Not implemented | Planned; not presented as connected or working |
| Production auth / durable cases | Not verified | Still future work; no readiness claim |

## Run and reproduce

Use the existing locked installation and `streamlit run dashboard/app.py`, then select **Coastal Watch**. Bundled data require no keys, map tiles or runtime network calls. The initial view uses a 60-minute exercise clock. Select Odisha to inspect the original eight-vessel exercise; select island territories to see the island cohorts. Move to minute 150 to demonstrate stale last-known positions.

The district control changes map focus only. Watch counts remain scoped to coast/state and vessel class; the UI says this explicitly. Source vintage and spatial parent-association gaps remain visible. Source-check dates are not survey dates or evidence of freshness.

To rebuild from separately downloaded pinned public source files:

```text
python scripts/build_india_reference.py --states <ADM1-file> --districts <ADM2-file> --output <new-reference.json>
python scripts/build_coastal_watch.py --source <Natural-Earth-file> --admin <new-reference.json> --output <new-bundle-directory>
```

Builders verify upstream hashes and refuse existing output targets. Compare generated outputs before replacement. `registry.json` supplies exact source revisions/URLs/hashes. Do not point the builders at a client dataset or a Drive source directory for output.

## Verification

Pure tests cover receipt causality, state/coast cohorts, delayed reports, duplicate semantics, invalid data, unknown/stale states, asset tampering, geometry holes and export hashes. AppTest and Chromium checks exercise the Streamlit controls and preserved modules in CI. Runtime validation status must be recorded against the PR commit before merge; this document alone does not assert that deployment passed.
