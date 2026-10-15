# Previous and current POC: side-by-side

Previous Streamlit baseline: GitHub main commit `cb3e3078336b0ea8c5105943f3ff1afd57f373b5` (merged PR #12). Current: the Streamlit scenario replay proposal containing this document. The current source is the branch/commit shown by GitHub; this comparison does not imply it has been deployed.

| Capability | Previous Streamlit baseline | Current Streamlit proposal |
| --- | --- | --- |
| Main application | Python/Streamlit, three research views | Same entry point and all three views, plus Scenario Replay |
| Historical vessel experiment | Saved 500-point scatter with methodological warnings | Preserved unchanged |
| Historical drone/cyber | Saved CSV metrics and PNGs; limitations visible | Preserved unchanged; no claim of improved model accuracy |
| Offline vessel benchmark | Fixed-seed snapshot generator, isolated train/validation/test, saved experiment bundle | Preserved unchanged |
| Vessel time series | No dashboard trajectory data | Four fictional tracks, 91 timestamped observations, fixed UTC clock |
| Playback | Static saved artifacts | Play/pause, seek, step, vessel filtering, bounded end, explicit empty states |
| Situational display | Historical PNG | Additional interactive position and speed charts, latest-report table and freshness counts |
| Flag explanation | No replay rules | Speed change, position jump and reporting gap, each linked to observation IDs and thresholds |
| Temporal behavior | No replay cursor | No future observations or evidence in rule calculations; historical flags remain after recovery |
| Human review | No Streamlit review exercise | Evidence acknowledgement, constrained transitions, rationale, session history and JSON export |
| Audit/identity | No operational identity/audit service | Still no operational service; self-asserted labels and volatile session history explicitly labelled |
| Synthetic provenance | Original generators unknown; separate benchmark traceable | Additional committed replay CSV, fixed seed, generator, MIT terms and SHA-256 manifest |
| Dependencies | Locked Python 3.13 runtime | Same dependency declarations and lockfiles; no new services |
| Verification | Existing artifact/helper/AppTest and benchmark checks | 60 Linux demo tests; 59 Windows tests plus 1 platform skip; 20 benchmark tests per OS; server startup and dependency audits passed |
| Lovable relationship | Showcase link plus hypothetical integration language | External hyperlink only; integration proposals withdrawn from current architecture |
| PR #10 | Unmerged separate ICCC application proposal | Preserved separately; not imported, expanded or deployed by this increment |
| Readiness | Experimental; not operationally validated | Reproducible workflow POC; still not operationally or nationally validated |

## PR #10 context, preserved separately

PR #10 is an unmerged FastAPI/Next.js proposal. Earlier work changed its baseline `6cccf04fe88712ae24c385388ed10792be66bd8d` to `48c5c74dac3b6dd6bfd13fd29d5fb4a02c48b624`. That work and its history remain separate from this Streamlit increment.

| Earlier PR #10 gap | Previously completed change at 48c5c74 |
| --- | --- |
| Object-level jurisdiction and cancellation gaps | Shared scope checks and explicit field-order delegation, with regression coverage |
| MFA replacement could weaken existing enrolment | Pending session-bound enrolment, expiry and reauthentication; retain active factor until confirmation |
| Manual dispatch attribution | Accepted-order confirmation records actor/channel/note through shared transition logic |
| Demo configuration boundary | Explicit demo/restricted mode and guarded synthetic seed/accounts/endpoints |
| Missing workflow fixtures | Fixed-clock invented vessel, drone-metadata and cybersecurity event packs |

Historical exact-head CI for that separate proposal: [ICCC run 38704124201](https://github.com/bikramtheitguy/ai-coastal-security-defense/actions/runs/38704124201): 139 backend tests for each database, 10 browser tests, dependency audits passed. These results do not establish Streamlit capabilities or production readiness. No PR #10 merge or Lovable operation is part of the current change.

The [current Streamlit POC](STREAMLIT_POC.md) is the development plan going forward. Its [validation record](STREAMLIT_VALIDATION.md) distinguishes local checks, CI and hosted deployment.
