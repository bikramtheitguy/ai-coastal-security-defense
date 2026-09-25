# Known Limitations

Read before any demonstration. This is a **Proof of Concept / Operational Demonstrator**; it must not be used for real operations.

## Data & integrations
1. **All operational data is synthetic** (personnel, assets, vessels, tracks, weather, incidents, intelligence). See SYNTHETIC_DATASET.md.
2. **Marine Police Station names and positions are illustrative**. The official list could not be retrieved during the build; replace it with
   authoritative master data.
3. **No live integrations**: AIS, coastal radar, UAV video, CCTV, satellite, NABHMITRA/VCSS, the vessel registry, IMD weather, MRCC/MRSC,
   ERSS-112 and WhatsApp are simulated or not integrated. The Data Source Health view states each one's status honestly.
4. **Map**: public OSM/OpenSeaMap tiles are not authorised for navigation and have usage policies unsuitable for production load; the offline
   coastline is a coarse hand-digitised approximation, not an official boundary. **No ENC** is integrated (layer slot only).
5. **Static nautical chart**: none was supplied — a labelled placeholder is shown; no georeferencing is attempted.
6. Zones (restricted, watch, SAR sectors), "sensitive installations" and "vulnerable landing points" are generic placeholders.

## Analytics & AI
7. Vessel analytics are **rule-based** with simple thresholds (tunable in Administration); they are cues for verification, not detection
   guarantees, and have not been validated against real traffic. Risk scores are heuristic.
8. The chatbot is a **deterministic keyword/rule engine**, not a neural language model. It handles the listed intents and common phrasings;
   unusual phrasing, dialects, sarcasm or long narratives may be unrecognised (the operator always sees the original message).
   The "canonical English" for non-English input is a structured interpretation plus key-term gloss, **not a full translation**.
9. The Odia/Hindi/Bengali/Telugu lexicon and replies are **unvalidated drafts** — native-speaker review is required.
10. Voice messages are stored but **not transcribed**.
11. Resource recommendation uses great-circle distance and cruise speed; it ignores coastline obstruction, currents, sea state and river bars.

## Platform & security
12. Authentication is local (JWT + server-side sessions, optional TOTP); no Government IdP/SSO integration.
13. Demo accounts share one password; TLS must be provided by a reverse proxy; no VAPT has been performed (see CYBERSECURITY.md §9).
14. Evidence integrity is a SHA-256 placeholder on local disk — not a certified evidence-management system; legal admissibility has not been assessed.
15. Backup/restore is a JSON export suitable for the POC dataset size; production needs PostgreSQL PITR.
16. Single-node deployment; the in-app simulator and rate limiter assume one replica. SQLite mode is for single-user demos only.
17. Time is stored in UTC and displayed in IST; there is no multi-timezone handling.
18. Readiness is computed on every request — fine at POC scale (18 stations, ~240 personnel, ~90 assets); large deployments should materialise it.
19. Accessibility has been considered (contrast, keyboard focus, labels, text status words beside colours) but not formally audited (e.g. against GIGW / WCAG).
20. The UI is designed for desktop/video-wall first; tablet works; the citizen channel is mobile-first. Very small screens in the ICCC views are not optimised.

## Facts shown in the application
21. Emergency numbers and programme names shown to citizens (112, 1554, VHF Ch 16, NABHMITRA/VCSS) were checked against public sources
    on 2026-09-24/25 (see README) but must be re-confirmed by the Coastal Security Wing before any public release.
