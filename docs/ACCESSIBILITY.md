# Accessibility review

Roadmap item 5 lists an accessibility review among the release gates before
real-data use. This increment adds an automated WCAG gate to the browser
suite and fixes everything it found. It is an engineering check, not a
conformance claim or certification.

## Target standard

Verified as of 17 October 2026:

- **GIGW 3.0** (Guidelines for Indian Government Websites and Apps) states
  conformance with **WCAG 2.1 Level AA**. GIGW 2.0 was based on WCAG 2.0.
  Sources: [new features of GIGW 3.0](https://guidelines.india.gov.in/new-features-of-gigw-3-0/),
  [GIGW 3.0 document](https://cdnbbsr.s3waas.gov.in/s3c92a10324374fac681719d63979d00fe/uploads/2026/07/2026072438.pdf).
  GIGW is also the basis for STQC Website Quality Certification, a separate
  formal process this repository does not perform.
- **WCAG 2.2** is the current W3C Recommendation (5 October 2023, republished
  12 December 2024). It adds criteria and removes 4.1.1 Parsing; it does not
  deprecate 2.1. Sources: [WCAG 2.2](https://www.w3.org/TR/WCAG22/),
  [What's new in WCAG 2.2](https://www.w3.org/WAI/standards-guidelines/wcag/new-in-22/).

The gate targets **WCAG 2.2 Level AA** rules, which include the 2.1 AA set
GIGW 3.0 cites.

Verification caveat: these primary pages could not be opened directly from
the development environment (network policy). Their content was read through
search results restricted to the official domains. Two third-party sites state
that GIGW 3.0 requires WCAG 2.0 AA; they appear to describe GIGW 2.0, and the
official sources say 2.1. Re-check the primary documents before citing this in
a formal submission.

## What the gate checks

`browser_tests/test_accessibility_browser.py`, in real Chromium, on all six
modules and all five further Coastal Watch tabs (11 views), at 1365 px and
390 px:

| Check | WCAG | Method |
| --- | --- | --- |
| Machine-checkable rules | 2.0/2.1/2.2 A and AA | axe-core 4.12.1 via `axe-playwright-python` 0.1.8, tags `wcag2a`, `wcag2aa`, `wcag21a`, `wcag21aa`, `wcag22aa`; zero violations allowed |
| Text contrast | 1.4.3 | `browser_tests/contrast.js`; see below |
| Reflow | 1.4.10 | Each module at 320 CSS px wide has no horizontal scrolling |
| Keyboard | 2.1.1, 2.4.7 | Tab reaches "Skip to main content" and the module selector; focus visibly changes the selector; typing and Enter switch modules |

**Why a separate contrast check.** axe-core cannot compute contrast over the
theme's radial gradient, so it marks those elements "needs review" rather than
pass or fail (6–44 elements per view). `contrast.js` scrolls each visible text
element into view and asks the browser which boxes are painted beneath its
centre (`document.elementsFromPoint`). It composites their backgrounds, tests
every gradient colour stop, and keeps the worst ratio. It applies the opacity
of every layer between the text and its background, exempts disabled
controls (as 1.4.3 does) and skips visually hidden helper text. Thresholds
are 4.5:1, or 3:1 for large text.

Per-view results (violations, needs-review counts, passed rules, contrast
failures) are written to `artifacts/browser/accessibility-<width>.json` in the
CI evidence bundle.

## Findings and fixes

| Finding | Before | After | Fix in `dashboard/spatial_style.py` |
| --- | --- | --- | --- |
| Captions: Streamlit renders them at 60% opacity | 4.0–4.4:1 on panels (144 elements) | ≥ 6.8:1 | Captions at full opacity; colour unchanged |
| Primary buttons (e.g. "Open investigation…"): white on the light theme blue | 1.78:1 | 10.4:1 | Navy (#0b1424) text |
| Selected multiselect chips (Scenario Replay vessels) | 1.78:1 | 10.4:1 | Navy text |
| Numbers in JSON viewers (`st.json`) | 3.76:1 | ≥ 5.2:1 | #f0995f instead of #cc6633 |

The light-blue theme colour is unchanged. With the fixes removed, the gate
fails on all four findings; with them, all 11 views pass at both widths, with
26 or more axe rules passing per view.

Remaining "needs review" items:

- `color-contrast`: the gradient cases above, resolved by `contrast.js`.
- `aria-prohibited-attr`: Streamlit's multiselect chips put `aria-label` on a
  plain `<span>`. The label repeats the visible text, so no information is
  lost; it is framework markup outside this repository.

## Not covered

Automated tools find only part of the barriers that affect people. Not done
in this increment:

- Screen-reader testing (NVDA, JAWS, VoiceOver, TalkBack) and testing with
  people who use assistive technology.
- A full keyboard walkthrough of every control, including the globe (a WebGL
  canvas; its evidence is also available in tables, the vessel inspector and
  the 2D map), chart interactions and the investigation forms.
- Text alternatives and meaning of each chart, colour-only cues, zoom to 200%
  and 400%, motion and timing (the replay autoplay), and error identification
  in forms.
- Browsers other than Chromium, and physical phones.
- Third-party content: the external showcase link, and Streamlit's own
  development toolbar.

These need a manual audit by a qualified assessor before any conformance
statement.
