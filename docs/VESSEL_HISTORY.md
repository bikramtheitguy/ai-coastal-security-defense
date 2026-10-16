# Vessel history and reporting continuity

## Plan and acceptance contract

Build on merged PR19 (`4c4de199d53450e7c454ef720a0699cef3e7967f`). The audience
is a reviewer asking what was known about a fictional vessel at a chosen time.
Add a Vessel history tab and an inspector handoff inside India Coastal Watch.
Keep the globe first and retain every existing module, case workflow and export.
Lovable remains a hyperlink only. Project relocation remains on hold.

The primary gap is temporal understanding: a last position alone does not show
whether reports were missing, delayed, rejected or subsequently recovered.
This increment uses the existing versioned Indian synthetic exercise; no new
data provider, credential, model or network request is required.

1. Scope a searchable dossier to the existing coast/state/class cohort, including
   catalogue vessels awaiting their first accepted report. Synchronize selection
   with the globe and inspector when an accepted position exists.
2. Show a graphic of **freshness as known at receipt time**, followed by the
   observation-to-receipt timeline and segmented reported-speed history. Offer
   the last 30 minutes, last 60 minutes and full exercise so far; show IST by
   default with explicit UTC selection. The exercise clock remains authoritative.
3. Distinguish current last-known observation age, delayed arrivals, historical
   stale duration and unobserved track intervals. A late old report must never
   move the latest position backward or erase a past stale interval.
4. Sum spherical distances only between accepted adjacent observations no more
   than 10 minutes apart, both in the selected window. Describe the result as
   sampled-link distance, never total travel or navigation distance. Rejected
   points and gaps do not become tracks. Speed is reported, not inferred heading.
5. Export deterministic bounded JSON with selection, definitions, registry,
   source receipt/observation IDs, derived intervals and a canonical SHA-256.
   Hashes establish content identity, not authentication. No upload or persistence.

## India acceptance scenarios

- Odisha SIM-004 at minute 60: last observation minute 25, age 35; no future
  recovery available. At minute 95: recovery received at minute 91, a historical
  stale interval remains visible, and the 25-to-90-minute track gap stays open.
- Odisha SIM-006 at minute 45: the delayed minute-25 report arrives after the
  minute-40 report. Mark it delayed/out of order; keep minute 40 as last known.
- Odisha SIM-008 at minute 0: awaiting first report, no fabricated location,
  distance or speed. A shorter window with no samples is also explicit.
- A rejected coordinate is present in receipt evidence but never in distance or
  speed calculations. Changing the selected cohort must not retain another
  cohort's vessel. Exercise-clock rewind must exclude future receipts.
- Both layouts render without overflow. New export verifies and existing watch,
  replay and casebook exports and all existing assertions remain intact.

## Why these semantics

AIS-related information includes distinct position and static/voyage reports;
reported data quality still requires review. The implementation uses these
concepts without claiming that its invented receipts are AIS packets or an
Indian live feed. References reviewed 16 October 2026:
[USCG NAVCEN AIS FAQ](https://navcen.uscg.gov/ais-frequently-asked-questions) and
[IMO AIS operational guidance](https://wwwcdn.imo.org/localresources/en/KnowledgeCentre/IndexofIMOResolutions/AssemblyDocuments/A.1106%2829%29.pdf).
The 15-minute freshness and 10-minute track-link limits are existing demo policy,
not regulatory thresholds, sensor coverage or evidence of unlawful activity.

## Before and after this increment

| Capability | PR19 baseline | This branch |
| --- | --- | --- |
| Vessel inspection | Last accepted position, age and source rows | Preserved, plus a history handoff and cohort-scoped dossier |
| Historical freshness | Current age and frozen investigation captures | Receipt-causal intervals retain earlier outages after recovery |
| Delayed observations | Intake counts and raw rows | Observation-to-receipt chart with delayed/out-of-order evidence |
| Track interpretation | Globe links separated by the 10-minute policy | Segmented speed history and explicitly partial sampled-link distance |
| Time scope | Global exercise minute | Preserved; adds 30/60-minute or full-history windows and IST/UTC display |
| Downloads | Watch, vessel evidence, replay and casebook | All preserved; adds a separately versioned history JSON with hash |
| No-report state | Unknown-vessel list | Dossier for registered unknown vessels without fabricated locations |

## Subsequent increments

Approved Indian observation adapters and source-health contracts; licensed
environmental context; authenticated persistent case service and recovery;
independent evaluation with benign counterexamples; measured national-scale
load and resilience. Each needs its own access, provenance and acceptance gate.
