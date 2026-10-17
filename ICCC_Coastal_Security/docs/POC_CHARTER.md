# Coastal security POC charter

This POC demonstrates how a synthetic report becomes a reviewed incident, an authorized task, a field acknowledgement and an auditable outcome. Its purpose is to test workflow, software behavior and data contracts before considering approved integrations. National coverage is a long-term design ambition, not a demonstrated deployment capability.

## Three related deliverables

| Deliverable | Purpose | Integration status |
| --- | --- | --- |
| Repository research demo and vessel benchmark | Reproduce saved experimental results and evaluate synthetic vessel-speed anomalies | Existing baseline on `main`; no operational inference or feeds |
| ICCC platform in PR #10 | Exercise incident, readiness, command, evidence and administration workflows with synthetic data | Separate FastAPI/Next.js application; under review |
| [Lovable coastal dashboard](https://icccodishacshq.lovable.app/) | Explore the command-centre interface and national/state/district/station concept | Separate source, hosting and database; no verified connection to this Python code |

Changes must preserve existing capabilities. Improving authorization means retaining a usable path for authorized users, with negative tests for unauthorized access. A read-only showcase is an additional presentation mode; it does not replace account provisioning, citizen reporting, staff dashboards or other existing workflows.

## Demonstration boundary

Use a disposable local environment and synthetic identities, locations, reports, evidence and telemetry. Public screenshots must show synthetic labels and contain no credentials, personal data or restricted information. Use approved public sources only after recording provenance and permission to redistribute. Do not submit demonstration reports into the hosted Lovable reporting workflow; its database is separate from this local POC.

No operational dispatch, emergency service, government endorsement, authoritative navigation or validated detection performance is claimed. Citizen messages, multilingual interpretations and recommendations require human review. Any future ML/LLM suggestion must carry its evidence and uncertainty and must not independently authorize tasking.

## Acceptance flow

Run the scenario in [VALIDATION.md](VALIDATION.md). It must demonstrate intake, human verification, resource selection, authorization, field acknowledgement, resolution, evidence verification and closure. Negative cases must show that a wrong-jurisdiction user cannot read or act on the incident and that no dispatch promise appears before confirmed action.

The current code includes a manual C5 path that checks an assigned asset without requiring field confirmation. This conflicts with the intended dispatch guarantee and is tracked in [REVIEW_FINDINGS.md](REVIEW_FINDINGS.md). Passing the existing happy-path scenario alone does not satisfy acceptance.

## Evidence required to advance

| Stage | Exit evidence |
| --- | --- |
| Source reviewed | Capability inventory, dependency/provenance review, findings and exact source revision |
| Reproducible local POC | Fresh-install build, backend tests on SQLite/PostgreSQL, browser acceptance, dependency review, repeatable synthetic reset |
| Controlled demonstration | Authorization and MFA blockers closed, feature parity checked, screenshots and demo script tied to the tested revision |
| Approved integration pilot | Data-owner permission, signed-off contracts, retention/access rules, failure handling, field evaluation and rollback exercise |
| Broader deployment | Measured capacity, multi-jurisdiction isolation, independent security review, disaster recovery evidence, operational ownership and acceptance |

The present review is at the first stage. CI results and fixes must be recorded before changing that status. Stage gates are evidence requirements, not a delivery-date promise.
