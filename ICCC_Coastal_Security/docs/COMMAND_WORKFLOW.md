# Operational Command Workflow

## 1. Command desk (Command & Tasking › Command Desk)
Authorised supervisors issue three kinds of order; every order carries **priority** (FLASH / IMMEDIATE / PRIORITY / ROUTINE), **recipients**,
the **order text**, the **issuer and rank**, optional **validity/expiry**, and timestamped transitions. All are audited.

| Order type | Permission | Recipients | Flow |
|---|---|---|---|
| Operational alert | `ORDERS_ISSUE` | stations and/or assets | SENT → ACKNOWLEDGED → (ACCEPTED) → COMPLETED / UNABLE |
| Personnel tasking | `ORDERS_ISSUE` | named personnel | same as above |
| Asset movement / incident response | `TASK_ASSETS` | one asset (+ destination or incident) | SENT → ACKNOWLEDGED → ACCEPTED / UNABLE → EN ROUTE → ON SCENE → COMPLETED; CANCELLED by the issuing authority |

Jurisdiction applies: an IIC may only task and address its own station's assets/personnel; district officers their district; state roles all.

## 2. Asset movement — step by step
1. **Select incident or destination.**
2. **System lists eligible READY assets** — only mission-ready assets are eligible.
3. **Nearest suitable assets ranked** by distance/ETA, readiness score, qualified crew, fuel/battery margin for the round trip and
   communications (weights configurable). Patrolling assets are eligible but flagged "would divert an active patrol". UAVs are listed separately
   as aerial search/confirmation (they cannot rescue persons). Non-eligible assets are shown with their reasons for transparency.
   Nearest MPS, current weather (simulated) and data provenance/freshness are shown. Label: **AI RECOMMENDATION — HUMAN AUTHORISATION REQUIRED**.
4. **Supervisor selects the resource** — any eligible asset; tasking a non-ready asset is refused unless an override reason is given (audited as
   `READINESS_OVERRIDE`). The recommendation shown at decision time is stored with the order for the After-Action Review.
5. **Movement order created** — asset becomes TASKED; an active patrol is marked diverted; incident timeline + audit updated.
6. **Field unit receives the order** — Field Unit Console for the boat master / UAV operator (assigned asset or crew membership) or the station IIC.
7. **Field unit acknowledges** (ACKNOWLEDGED; acknowledgement stored) and **accepts** or reports **UNABLE** with a mandatory reason (asset freed).
8. **EN ROUTE** — asset moves on the map at cruise speed, crew marked DEPLOYED, incident moves to **C5 Dispatch Confirmed**, citizen notified.
9. **ON SCENE** — confirmed by the field unit (the map separately flags "arrived — awaiting ON SCENE confirmation"; if the tracked position disagrees
   with the destination by > 1 NM, the transition note records the discrepancy).
10. **COMPLETED** — completion report stored; asset RETURNS to base and becomes IDLE/AVAILABLE on arrival; crew stand down.

Every transition stores status, UTC timestamp, user and note, and is shown as a timeline on the order.

## 3. Acknowledgements & oversight
* *Command Acknowledgements* lists orders with acknowledgements (who, when, note).
* *Active / Completed Orders* show status and full transition history; issuing authorities can cancel.
* The Live Nautical Map shows active movement orders as asset→destination lines and lists them in the context panel.

## 4. Separation of duties (why the chain is trustworthy)
| Step | Actor | Cannot |
|---|---|---|
| Detect / structure | AI (chatbot, analytics) | verify, task, dispatch, close |
| Verify | ICCC operator | task assets or close |
| Review & decide | Supervisor / authorised officer | acknowledge on the field unit's behalf |
| Execute | Field unit | issue or re-assign orders |
| Close & learn | Supervisor / IIC / senior officer | — (closure requires a recorded outcome) |

## 5. UAV mission console
UAV operators land on *UAV Mission Console*: their UAV's readiness (battery, link, GNSS, pilot), current mission and assigned orders with the same
acknowledgement flow. UAV missions can be planned in *Assets › Patrol Planning*.
