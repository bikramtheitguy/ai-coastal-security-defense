# Synthetic incident exchange contract

This small, dependency-free Python module validates fictional intake records before they could be mapped into an application. It applies concepts from the [coastal dashboard](LOVABLE_CONCEPT_MAPPING.md): explicit jurisdiction, original destination preserved on forwarding, source/received timestamps and coordinate quality. It does not connect to either dashboard, write records, authenticate users or authorize response actions.

## Run

From the repository root:

```bash
uv run --locked python -m coastal_security.incidents --input data/synthetic_incident_exchange.json
uv run --locked pytest tests/test_incident_contract.py
```

The CLI prints aggregate validation results only. The example yields two accepted synthetic records, one per fictional state, with `human_review_required: true` and `persisted: false`. Invalid input exits with status 2. The CLI limits input files to 1 MB; it is a local fixture tool, not an exposed upload service.

## Version 1 fields

| Field | Required rule |
| --- | --- |
| schema_version / data_class | Exactly `coastal.incident.v1` / `SYNTHETIC` |
| incident_id / source_id | Bounded ASCII `SYN-` identifiers; incident IDs unique within a bundle |
| source_time / received_time | ISO 8601 with UTC offsets; source instant no later than receipt |
| target | Registered state; optional district/station must belong to its parent; no inferred jurisdiction |
| position | Paired finite coordinates within latitude/longitude bounds and `SIMULATED`, or both null and `UNKNOWN` |
| original_language / original_message | Bounded nonempty text preserved as provided; no translation claim |
| verification | Exactly `UNVERIFIED`; imported verified/dispatch flags are not trusted |

All fields are required; extra keys are rejected. The registry explicitly maps stations to districts and districts to states. Unknown location and a state-only destination are supported, without inventing GPS coordinates or a station assignment.

The [fixture](../data/synthetic_incident_exchange.json) was authored for this repository using fictional identifiers, fixed timestamps and artificial coordinates at zero latitude/longitude. It is not an approved navigation point or a copy of operational data. This new repository-authored example is covered by the root license; no third-party dataset is embedded.

## Forwarding invariant

`validate_incident` returns a frozen record with a frozen jurisdiction. `propose_forward` returns a separate proposal containing the incident ID, origin, destination and reason. It leaves the original destination unchanged. No notification, authorization decision or state transition occurs.

Future API integration must add authenticated actor scope, approval/acknowledgement, idempotency, durable event storage and transactional failure handling. Do not treat structural membership validation as access control. Public/operational datasets are deliberately unsupported by this first contract until their provenance and access requirements are designed.

## Verification

Tests cover accepted intake, unknown positions, mismatched jurisdiction, timestamps with offsets, non-finite/out-of-range/boolean coordinates, forged verification, unexpected fields, duplicate IDs, invalid registries, immutable targets, invalid forwards and CLI error behavior. They run in the existing research/demo CI on Windows and Linux. Synthetic pass rates do not validate operational incident quality or national-scale performance.
