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

## Related responsive test setup

Full-suite verification also exposed a sidebar race in the existing research
panel test (run **38865584144**). After resizing, the Collapse sidebar button
could be visible at the check but disappear before the click because Streamlit
automatically collapses the sidebar at that breakpoint. Narrow-layout tests now
collapse it at the stable desktop width before resizing. All existing sidebar,
chart, viewport, and content assertions remain. Desktop evidence filenames are
preserved and additional mobile export-flow screenshots are saved separately.

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

## Download source checks after a rerun

After PR #21 merged, `test_odisha_briefing_history_review_and_exports[desktop]`
intermittently failed at teardown: 5 of 10 local Chromium runs at `abc3e7c`.
Every assertion and download passed. The console recorded
`Client Error: Download Button source error - 404` for batches of `/media/`
JSON/TXT URLs.

In the pinned 1.66.0 frontend, every render of an eager download button sends
a background `checkSourceUrlResponse` request for its URL. Changing a control
reruns the script and registers new files; the server may release the previous
URLs before those checks return. Clicks use the current URL, so downloads still
worked, but any console error fails the browser suite and the race could block
unrelated PRs.

All views now render downloads through `dashboard/downloads.py`. It passes the
already-computed content as a deferred callable, so a button has no URL to check
until it is clicked, and sets `on_click="ignore"` so the click cannot rerun and
revoke the deferred registration before the file is fetched. Content is still
computed during the script run: exported bytes and displayed hashes are
unchanged. `tests/test_downloads.py` rejects direct `st.download_button` calls
in views and asserts rendered buttons are deferred and ignore reruns. Browser
tests still click the real buttons and validate saved files.

## Idle waits after a globe rebuild

Browser run **39006889984** (PR #23) failed in
`test_globe_narrow_and_research_views`: after the Odisha camera click,
`stApp[data-test-script-state]` stayed `running` past Playwright's 5 s default.
The same test also failed 1 of 8 local runs of unmodified `main`, so the race
predates the fictional-site layers.

A state/UT scope change and every camera preset deliberately re-key the pydeck
chart, so deck.gl rebuilds the globe. A traced local failure showed the scope
rerun finishing in 1.5 s, then the camera rerun running from 4.0 s to 11.4 s
while main-thread long tasks of 4.4 s and 1.8 s ran: under software WebGL the
frontend cannot record that the script finished until the rebuild yields. The
script itself takes about 0.2 s on `main` and on PR #23.

`wait_for_idle` now allows 20 s for the run-state to settle. It still fails if
a run never finishes; it does not add retries, skip tests or inject state. The
narrow globe test also confirms the Odisha scope (metrics, then idle) before
clicking the camera preset, so each rebuild has its own wait. Clean samples put
PR #23's site layers within about 10% of `main`'s rebuild cost; the rebuild on
camera changes is existing design and a candidate for the performance gate.
