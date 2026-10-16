# India Coastal Watch: investigation increment

| Capability | Previous: PR #15 | Current proposal |
| --- | --- | --- |
| Review queue | Current rule flags and source rows | Same queue plus opening a vessel case |
| Evidence over time | Snapshot download at one clock | Up to six frozen captures per case, with observation IDs, freshness, rejection counts and rule changes |
| Reporting-gap recovery | Inspect each clock separately | Compare stale and recovered captures while retaining both |
| Analyst decisions | Available in the separate Scenario Replay view | Also available for Coastal Watch: New, In review, Closed — benign, Escalated — demo; rationale and acknowledgement required |
| Receipt timing | Table of observation and receipt times | Per-observation delivery-delay timeline within the case |
| Handover | Raw watch/vessel JSON | Evidence-linked plain-text case brief and portable JSON casebook |
| Recovery | Watch has no saved cases | Validate and restore a downloaded casebook into an empty session |
| Import integrity | No casebook import | Bounded JSON, duplicate-key rejection, strict event replay, fixture identity and independently reproduced snapshots |
| Geography and source data | 13 coastal states/UTs, dated district context, 44 fictional vessels | Preserved unchanged |
| Existing modules | Six views | All six preserved; Investigations is a tab within Coastal Watch |
| Persistent authenticated service | Not implemented | Still proposed; downloadable recovery is not shared durable storage |

## Demonstration

1. Choose **Coastal Watch → Investigations**, with minute 60 and All coasts selected.
2. Select **SIM-004**, read its stale-report evidence, enter a fictional reviewer label and rationale, acknowledge the evidence, and open the investigation.
3. In **Review & handover**, record **In review** with a rationale.
4. Advance the watch clock to minute **95**. Under **Captured evidence**, expand **Append evidence at current watch clock** and record a new capture.
5. Compare minutes 60 and 95. The earlier last observation was 35 minutes old. The later capture contains received recovery evidence without changing the case status automatically.
6. Record **Closed — benign** or **Escalated — demo**. These are local exercise decisions and send nothing.
7. Download the casebook and handover. Open a fresh session, expand **Restore a downloaded casebook**, upload the JSON, inspect the validated preview and restore it.
8. Verify history, hashes and scope. An unrelated session has no saved cases. Changing map filters or seeking backward retains the casebook and clearly distinguishes it from the current watch.

## Scope and interpretation

Cases are per vessel for the whole bundled exercise, not detected incident episodes. Capturing new evidence retains a prior disposition and displays a review reminder. The casebook is independent of map cohort filters. It supports at most 12 cases, six captures per case, 60 events per case and an 8 MiB export/import. There is no case deletion or implicit history replacement.

Reviewer labels, notes and recording timestamps are self-asserted. A valid imported casebook is not authenticated evidence. Source snapshots are reconstructed from the pinned fictional exercise; geographic rights and source notices remain in the embedded registry. Hashes are reproducibility fingerprints, not signatures.

See [design and acceptance plan](INVESTIGATIONS_PLAN.md) and [validation](INVESTIGATIONS_VALIDATION.md). Screenshots will be added after real-browser verification of the proposed code.
