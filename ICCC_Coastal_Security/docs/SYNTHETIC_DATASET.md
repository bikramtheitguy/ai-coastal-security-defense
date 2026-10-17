# Synthetic Dataset Description — SIMULATED / POC DATA

Generated deterministically by `seed/generate.py` (seed `SEED_RANDOM`, default 20260924) the first time the application starts
on an empty database. **Nothing in it represents real people, deployments, credentials, boundaries or intelligence.**

## What is generated (default seed — counts verified against a fresh seed on 2026-09-25)

| Entity | Volume | Notes |
|---|---|---|
| Districts | 6 | Balasore, Bhadrak, Kendrapara, Jagatsinghpur, Puri, Ganjam (public names; approximate centroids) |
| Marine Police Stations | 18 | **Illustrative** coastal locality names and hand-placed approximate positions — not the official list |
| Personnel | 242 (230 station-posted) | Per station: IIC, 2 boat masters, 7–10 marine crew, driver, UAV pilots at UAV stations; plus ICCC/HQ staff. Synthetic Odia-style names from name pools; masked synthetic mobiles `90000xxxxx` |
| Qualifications & training | 1,297 qualifications + 1,297 training records | 11 qualification types, 11 courses, validity windows; ~6 % of swimming qualifications deliberately expired |
| Boats | 26 FIBs (16 × 12 T, 10 × 5 T), 5 hired trawlers, 8 RWCs | Crews assigned from station personnel; maintenance history |
| UAVs | 6 | Pilots assigned; UAV-02 at Dhamra has a battery defect (demo) |
| Vehicles, comms, sensors | 18 vehicles, 18 VHF base sets, 8 EO/IR sensors | |
| Map reference places | 142 | FLCs, CCTV, jetties, cyclone shelters, comm towers per station; public ports/harbours/islands/river mouths (approximate); generic placeholder "vulnerable landing points", "sensitive installations", radar sites and surveillance towers |
| Zones | 11 | 3 restricted geofences around placeholder installations, 6 SAR sectors, 2 watch zones — all illustrative |
| Vessels | 149 | 140 fishing craft (trawler, gillnetter, motorised, non-motorised), 8 merchant ships named "(SIM)", 1 dark radar contact; registrations `OD-SIM-<DIST>-nnnn`, MMSIs prefixed `SIM` so they cannot collide with real vessels |
| Track history | 1,788 points at seed time | 2 h per vessel, 10-min spacing |
| Patrols | 90 completed (30 days) + 6 active | Active Dhamra patrol is `PATROL-<year>-0098` on FIB-12T-04 (spec example) |
| Incidents | 46 closed/false (60 days) + 1 open | Realistic interval distributions for response-time analytics |
| Alerts | 60 historical + live ones raised by the analytics engine | |
| Weather | 1 per district | Labelled "SIMULATED, not IMD" |
| Data sources | 14 | Honest integration status: SIMULATED / NOT_INTEGRATED / LIVE (public map tiles only) |
| Users | 15 demo accounts | One per role; shared `DEMO_PASSWORD`; flagged `is_demo` |

## Deliberate demonstration states
* **Dhamra MPS ≈ 75 % AMBER** with explainable reasons: FIB-12T-03 under maintenance (gearbox), UAV-02 battery below threshold
  with a battery-replacement defect, backup VHF test overdue, sea-ready personnel on leave, trawler short of qualified crew.
* **FIB-12T-04** patrolling from Dhamra, 6/6 sea-ready crew, qualified master, fuel 68 %, 18 kn (the specification's example card).
* One boat with a CRITICAL defect, one at 22 % fuel, one with deficient safety equipment, one in reserve, one with degraded VHF.
* Sonapur VHF base degraded; Kharinashi backup link offline; one FLC camera degraded.
* Intelligence picture: one identity mismatch, one AIS gap, two designated TOIs with exercise reasons, two watch-list entries, one dark contact.

## Live behaviour
The simulator moves patrols along routes, tasked assets towards their destination and vessels according to their behaviour, burns fuel,
records track points every ~30 s, refreshes telemetry timestamps and runs analytics/fusion every 5 ticks. Scenario injection
(Analytics › Exercise Scenarios) adds the 15 test situations on top — see TEST_SCENARIOS.md.

## Resetting
Docker: `docker compose down -v && docker compose up -d`. Local: delete `data/` (or `dev_server.sh restart --fresh`).
Change `SEED_RANDOM` for a different but still deterministic dataset.

## Replacing with authoritative data
Load real master data through Administration (stations, personnel, assets, qualifications, zones) or a one-off import script against the
same tables; set `SEED_ON_START=false`. Replace the illustrative station list first, and remove every "(SIMULATED)" label only when the
corresponding record comes from an authoritative source.
