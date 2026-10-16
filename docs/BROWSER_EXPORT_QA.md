# Browser export regression diagnosis

The two export failures in browser run **38858818024**, attempt 2, were UI
synchronization races. They did not show missing export data or console errors.
The archived Playwright traces and the pinned Streamlit **1.66.0** frontend
explain why waiting for visible content alone was insufficient.

## Watch snapshot: navigation during a rerun

In `test_india_watch_scope_layers_and_evidence`, toggling Observation history
starts a Python rerun. The trace shows both subsequent tab clicks while
`stApp[data-test-script-state]` is `running`. The final page remains on Intake
quality rather than Sources & export, so the download button never becomes
available to the click. Seeing the intake text did not establish that Streamlit
had finished replacing stale elements.

`dashboard/watch_view.py` intentionally uses dynamic tabs with
`on_change="rerun"`; the globe-to-investigation handoff depends on their state.
The fix synchronizes test navigation with the existing run-state helper and
asserts both the selected tab and its visible panel. Dynamic tab behavior stays
intact.

## Replay review: clicking inside a clipped, animating expander

In `test_review_validation_disposition_download_and_reset`, the download click
starts about 57 ms after the expander click. The trace shows `details[open]`
still locked to `height: 40px; overflow: hidden`. Scrolling the download button
into view scrolls the clipped details region during its opening animation; the
click returns but no download request or event follows. The script is already
`notRunning`, so a server-idle check alone cannot solve this failure.

The pinned frontend's `useDetailsAnimation` and `animateHeight` temporarily
apply those inline styles and remove them when the transition ends. The test
now asserts that the details element is open **and** its temporary size/overflow
lock has cleared before interacting with its contents. This also protects the
replay provenance downloads. It does not disable animations, force clicks,
inject downloads, retry failed tests, or replace events with direct HTTP calls.

Source: [Streamlit 1.66.0 expander implementation](https://github.com/streamlit/streamlit/tree/1.66.0/frontend/lib/src/components/elements/Expander).

## Verification contract

Both formerly failing flows run at desktop (1365 px) and mobile (390 px) widths.
All original assertions remain, including snapshot SHA-256, Odisha scope,
synthetic provenance, review event order, row counts, validation, and reset.
Viewport overflow checks and selected-tab/panel assertions add coverage.
The complete browser suite still rejects JavaScript/console errors and
unexpected external requests. Globe rendering/picking, investigation handoff,
research panels, evaluation, playback, and the existing narrow-layout tests
remain part of the same suite.

Only browser test synchronization and coverage change. Application code,
datasets, model outputs, dependencies, and runtime capabilities are unchanged.
