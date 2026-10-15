# Trajectory evaluation and spatial UI: validation record

Validated application revision: `1715687345dab08645a8df693b14edbf66cdcc35`, 15 October 2026. Documentation and unedited screenshots may be committed later without changing application behaviour. This is a proposal in PR #14; it has not been merged or deployed to the public Streamlit site.

## Executed checks

| Environment / check | Result | Evidence |
| --- | --- | --- |
| Windows local pure and regression suite | 60 passed, one Linux-only skip | Domain arithmetic, matching, causal replay, split isolation, deterministic report, overwrite refusal and preserved fixtures |
| Linux Streamlit application suite | 70 passed | [Baseline workflow](https://github.com/bikramtheitguy/ai-coastal-security-defense/actions/runs/38779133176), job 116574458122 |
| Windows Streamlit application suite | 69 passed, one Linux-only skip | Same workflow, job 116574457871 |
| Existing vessel ML benchmark | Tests and documented default CLI passed on both operating systems | Same workflow, jobs 116574457690 and 116574458041 |
| Dependency exports, Ruff, runtime/research/dev audits | Passed | Same workflow; dependency-audit job 116574457889 |
| Chromium desktop and narrow layout | Eight cases passed | [Browser workflow](https://github.com/bikramtheitguy/ai-coastal-security-defense/actions/runs/38779133243), job 116574457506 |
| Browser tooling lock, lint and dependency audit | Passed | Same browser workflow |
| Screenshot inspection | Reviewed desktop evaluation and preserved research views, mobile evaluation chart, replay and review views | Unedited JPEGs in `docs/screenshots/`; comparison below |

Browser tests use the real Streamlit server, Chromium, a desktop viewport and 390 × 844 narrow viewport, with unexpected external HTTP and browser JavaScript errors failing the test. They exercise existing experiments, replay controls, review transitions and downloads, plus evaluation filters, evidence/protocol tabs, complete JSON download and navigation. Streamlit AppTest independently covers evaluation state and metric changes. These checks do not constitute an accessibility, physical-phone, Safari/Firefox or load certification.

The first browser attempt passed seven cases and exposed a filter-test sequencing problem after chart scrolling. The test now exercises the filter before scrolling to the comparison. All eight pass on the final application revision. Visual inspection also caught a missing narrow-chart method label; horizontal grouped bars now retain every method label. Status text and bar contrast were strengthened. Caption text remains part of future full accessibility review.

## Default synthetic result

There are 48 test vessels, 96 monitored vessel-hours and 24 injected events. Calibrated and fixed-threshold rules both produce **23 matches, 20 false alerts, one miss and eight duplicate alerts**. Precision is 53.5%, recall 95.8%, F1 68.7% and false-alert workload 0.208 per vessel-hour. No-alert baseline misses all 24 injections. The calibration selected 8 kn speed delta, 40 kn implied speed and 15-minute reporting gap.

This result does **not** show a quality improvement over the fixed baseline. It demonstrates that the evaluation exposes missed events and substantial review workload rather than treating every designed replay flag as success. No confidence interval or real-world extrapolation is supplied for this small, designed same-family suite.

## Hosted PR #13 check

The maintainer explicitly approved merging PR #13. It merged at `4a1630a25d533a415948035563608db783fe00e2`. Post-merge [baseline](https://github.com/bikramtheitguy/ai-coastal-security-defense/actions/runs/38774353067) and [browser](https://github.com/bikramtheitguy/ai-coastal-security-defense/actions/runs/38774353132) workflows passed. The public Streamlit app was then opened: all three historical views and Scenario Replay rendered; replay playback advanced from 01:00 to its 01:55 end with four flags. The app does not expose a deployed commit ID, so this verifies visible behaviour, not an independently attested hosting revision. No Lovable link or editor was operated.

## Limits and data boundaries

Local Windows sockets remain restricted (`WinError 10013`); full runtime checks ran in CI without bypassing local protection. No client map files, extracted coordinates, external data adapters, model serialization or runtime dependencies were introduced. Only fictional generated data enter this increment. Public geographic layers remain a separate proposed increment. National-scale capacity, operational readiness, authentication, persistent review and field performance remain unverified.

See [POC comparison](STREAMLIT_POC_COMPARISON.md), [protocol](TRAJECTORY_EVALUATION.md) and [design/data roadmap](DESIGN_AND_DATA_ROADMAP.md).
