# Coastal dashboard concepts for repository development

The separate [Lovable coastal dashboard](https://icccodishacshq.lovable.app/) is a read-only design and source reference for this repository effort. It must not be edited, deployed or queried for operational records as part of this work. Its features and data structures inform repository design; no database or API integration is implemented by this document.

Source reference: project `fbbb81f7-47f0-4ad9-9d77-68978f5c1f37`, revision `3024e4d24e4637078eb363d289e91be3d979d2c8`, inspected 15 October 2026. The source revision is not proof of the currently published deployment. Roadmap checkmarks are author-reported progress, not independent acceptance results.

## Concepts and their repository destination

| Lovable source concept | Evidence inspected | Repository implementation or next step |
| --- | --- | --- |
| Operations, incidents, tasking and officer dashboard | `src/lib/menu.ts`, role capability matrix, roadmap | Existing ICCC incident/command views; reconcile terminology and test one complete human-reviewed flow |
| National/state/district/station hierarchy | `staff_scope`, `districts`, `police_stations` types and approval policy | New synthetic jurisdiction contract; persistent multi-state ICCC model/policy remains proposed |
| Citizen chooses report destination; forwarding preserves it | `incidents.target_state`, `report_state_forwards`, project guidance | New immutable intake record and separate offline forward proposal; future API must enforce permissions |
| Original message, interpretation, source and receipt times | `incidents` type definition | Preserve original synthetic text and timezone-aware timestamps in the contract; interpretations never replace evidence |
| Coordinate quality and reference-place provenance | `reference_places` type and project guidance | Keep simulated/unknown quality explicit; approved public layers need source/license records and accuracy labels |
| Role-specific verification, escalation and authorization | `src/lib/roles.tsx` | Map concepts deliberately to ICCC permissions; its different role names are not interchangeable |
| Assets, personnel/training, patrol routes and progress | Menu, schema types and roadmap | Retain ICCC readiness/crew/patrol features; design contract tests before any cross-system mapping |
| Evidence, after-action reviews and audit | Menu and schema types | Existing ICCC custody/AAR/audit foundation; external anchoring and recovery remain open |
| AI briefing and public assistance | Component inventory, role guidance and roadmap | Preserve advisory/human-review boundary; separately evaluate model privacy, grounding and failure modes |
| Isolated public demonstration | `/demo` source additions and local fixtures | Additive synthetic showcase concept; not a replacement for full application capabilities |

## Planned work visible in the dashboard roadmap

The inspected roadmap leaves data-entry templates, approved station coordinates and notification delivery unfinished or dependent on owner decisions/services. Carry these forward as requirements with explicit acceptance evidence, not as already functioning repository features. Do not publish private account details, test report identifiers or internal notes from the dashboard roadmap.

## Data reuse boundary

Use schema concepts, workflows and newly authored synthetic fixtures. Do not copy operational rows, staff identities, contacts, credentials, real incidents or precise restricted locations. The new example uses invented `SYN-` identifiers and artificial coordinates; it contains no data export from Lovable. A declaration of synthetic data is not an automated privacy detector, so fixture review remains necessary.

Approved public sources can be considered later with owner, URL/version, license, retrieval date, coordinate quality and redistribution terms. Data sharing, synchronization and approved live feeds require separate adapter design and review.

## Preserve capabilities

Feature parity is a requirement for repository improvements. Inventory routes/actions, successful authorized workflows, failure states and data persistence before and after a change. Fix access controls through verified roles and scope; do not replace a legitimate capability with unconditional denial. This requirement does not authorize any change to the separate Lovable project.
