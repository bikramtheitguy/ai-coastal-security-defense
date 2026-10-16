# India Coastal Watch validation

Validated on 16 October 2026 (India time) against commit `aba8de14157d541565dd0433f2b4feb88d29aab0`, based on main `485357e97feecc9ed39a72ceb88009197af41910`. This is pre-merge CI evidence for draft PR #15, not a hosted deployment record.

| Check | Verified result |
|---|---|
| Linux Python 3.13 demo suite | 91 passed |
| Windows Python 3.13 demo suite | 90 passed, one Linux-only Bash-wrapper test skipped |
| Research benchmark suites | Passed on Linux and Windows; 20 tests per platform and default CLI exercise |
| Ruff and dependency export checks | Passed |
| Locked runtime/research/development dependency audits | Passed; this does not establish absence of all vulnerabilities |
| Chromium desktop and 390-pixel mobile suite | 10 passed, including the five preserved views and new watch workflows |
| Browser tooling audit | No known vulnerabilities reported by the checked advisory database |

Evidence: [baseline workflow](https://github.com/bikramtheitguy/ai-coastal-security-defense/actions/runs/38791639670) and [Chromium workflow](https://github.com/bikramtheitguy/ai-coastal-security-defense/actions/runs/38791639659). The browser workflow retains traces and screenshots as time-limited artifacts; selected screenshots are committed in `docs/screenshots/`.

## What was exercised

- Receipt availability at the clock, out-of-order observations, equivalently encoded UTC timestamps, rejection reasons, and received = accepted + rejected.
- State/coast cohort reconciliation, unknown and stale positions, polygon holes, district map focus and asset-tamper refusal.
- Streamlit state controls, all-coast/Odisha/island scopes, explicit Lakshadweep district gap, and return to the preserved drone view.
- Real Chromium map layers, finite rendered geometry, mobile resize, intake totals and downloaded snapshot SHA-256 verification. Unexpected browser page/console errors and external HTTP requests fail the tests.
- Visual inspection of national, Odisha and Lakshadweep maps, including mobile. Labels are reduced in national view; data notices and fictional exercise context remain visible.

The initial CI runs caught GeoJSON line-ending conversion on Windows, an invalid projection-fit container, polygon winding incompatibility, and responsive layout collapse from off-scope geometry. Repairs preserve strict byte checks, normalize only display copies and clip map marks. Regression coverage was added before the successful run.

## Limits

The local sandbox does not support the required Streamlit/browser sockets, so runtime checks ran in GitHub Actions. Local domain/schema tests and lint also passed. No live provider connection, production authentication, persistent incident store, large-load test or real-world detection evaluation was performed. Current district jurisdiction and geographic accuracy require independent verification before operational use. No merge or Streamlit deployment is claimed here.
