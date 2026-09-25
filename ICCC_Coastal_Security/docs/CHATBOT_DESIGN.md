# AI Maritime Public Assistant — Design

One digital gateway for any marine problem: distress, missing boats/fishermen, suspicious activity, pollution and hazards, and
information requests (weather, nearest Marine Police Station, sea-safety, VHF help, live-location help).

## 1. Channels
| Channel | POC status |
|---|---|
| Web / mobile web (`/citizen/`) | Implemented (mobile-first) |
| QR code | Implemented — QR should point to `/citizen/?channel=QR` (channel is recorded) |
| WhatsApp | **Simulated** endpoint `POST /api/public/chat/webhook/whatsapp-sim` (one conversation per sender number); no WhatsApp Business integration |
| Voice messages | Implemented as audio upload (recorded in browser), preserved and hashed; **no automatic transcription** — operator listens |
| SMS fallback, native mobile app, IVR voice | Planned (same engine; see ROADMAP) |

## 2. Language pipeline (per message)
```
citizen message
 → language & script detection  (Unicode block counts: Odia / Devanagari / Bengali / Telugu; romanised markers for transliteration)
 → ORIGINAL STATEMENT PRESERVED (stored verbatim, shown to operators)
 → normalisation (native digits → ASCII, ZWJ/ZWNJ removed)
 → concept detection            (multilingual lexicon: engine, drifting, sinking, taking water, injured, suspicious, …)
 → intent = incident family     (42 families covering every family in the specification; ordered rules — see pipeline.FAMILIES)
 → entity extraction            (coordinates / DMS, place names in 5 scripts + "N km east of <place>", persons on board,
                                 registration numbers, injuries / water ingress / life-jackets, time expressions)
 → emergency classification     (L1–L4 + escalation rules)
 → missing safety-critical info (per family class) → next single question
 → canonical English interpretation for the operator (labelled "machine interpretation — verify against original", with key-term gloss)
 → structured incident (provisional C1) when the creation threshold is met
```
Supported languages: **Odia, English, Hindi, Bengali, Telugu**, including native script, romanised (transliterated) and mixed-language input.
Replies use simple language in the citizen's language. **Adding a language** = its Unicode range + lexicon entries + reply templates;
no code changes.

## 3. Priorities
| Level | Meaning | Examples | Behaviour |
|---|---|---|---|
| **L1 CRITICAL** | Immediate threat to life / serious security event | sinking, capsized, man overboard, fire, explosion, drowning, hijacking, robbery, beach missing person, cyclone distress | Incident created **immediately** (even before location); safety advice + 112/1554 message; location then POB asked |
| **L2 URGENT** | Serious, likely to worsen | engine failure / drifting, collision, medical, missing boat/fisherman, grounding, fuel shortage, being followed | Incident created once location + persons are known |
| **L3 PRIORITY / SECURITY** | Suspicious activity, crime, pollution, hazard | suspicious vessel/landing, illegal fishing, oil spill, floating obstruction | Incident once location known; observable facts requested |
| **L4 INFORMATION** | Weather, safety, nearest station, reporting help | | Answered directly; no incident |

Escalation: reported water ingress, severe medical signs (unconscious, chest pain…), persons in water, or storm conditions raise an L2
distress to L1; a "yes" to "is anyone injured / is water coming in?" escalates conservatively.

## 4. Only the safety-critical questions
| Class | Asked (one at a time, only if missing) |
|---|---|
| Distress | location → persons on board → injuries / water ingress |
| Missing | location → persons → boat name/registration → last contact time & place |
| Security | location → what exactly was observed (boat description, persons, direction) → when; then an offer to send media *only if safe* |
| Hazard | location → description |

Information the citizen already gave (in any message, any language) is not asked again.

## 5. Intelligence principle — not believing interpretations
Words that express a conclusion ("smuggling", "terrorist", "contraband", "तस्करी", "ଚୋରା ଚାଲାଣ", "পাচার", "స్మగ్లింగ్" …) are detected as
**citizen allegations**. The incident is classified "POSSIBLE SUSPICIOUS VESSEL — citizen report, requires human verification", the title
never repeats the allegation, the operator's canonical English carries an explicit note, and the citizen is asked for observable facts and told
not to approach, follow or confront anyone.

## 6. Human-in-the-loop guarantees (enforced in code)
* The assistant can only create **provisional (C1)** incidents. It cannot verify, task, dispatch, seize, intercept, accuse or close.
* Citizen status messages are generated **only by lifecycle events performed by authorised humans or field-confirmed orders**:
  C2 verified, C3 MRCC notified, C5 dispatch confirmed (only on field EN ROUTE of an authorised order), ON SCENE, C6 safe (recorded by operator),
  C7 closed, C8 not verified. Templates for these cannot be sent manually by operators.
* No bot template ever says help is on the way; the automated tests assert this before C5.
* **Human takeover:** an operator takes over a conversation; the bot then stays silent; the operator replies with translated templates
  (sent in the citizen's language) or free text; the operator can return the conversation to the assistant.
* **MRCC/MRSC handoff:** records a handoff with a structured summary and moves the incident to C3. There is **no live MRCC link** (simulated).

## 7. Operator view
Every citizen message shows the original text, detected language/script, mixed/transliterated flags, intent and priority, allegation flag,
extracted facts, key-term gloss and the canonical English interpretation. Distress, suspicious, takeover and handoff queues are separate
views; analytics show language/channel/priority distributions and the count of unrecognised messages for lexicon improvement.

## 8. "My Boat"
Registration + owner mobile (POC stand-in for OTP) returns the boat's registry details (name, owner, home FLC, crew, expected return,
NABHMITRA/VCSS transponder, MMSI, safety equipment, emergency contact, Marine Police Station, previous incidents). Starting a chat with a
registration links the boat to any incident automatically.

## 9. Provider interface & data protection
The POC engine is deterministic and offline — **no citizen data leaves the server**. A neural translation or LLM provider (for example an
approved national language platform, or an on-premise model) can be introduced behind `pipeline.interpret()` provided that: the original is
always preserved and shown; the provider output is labelled as machine interpretation; the human-in-the-loop rules above stay in the dialogue
layer, not the model; and data-residency approval is obtained.

## 10. Validation required before public use
The Odia, Hindi, Bengali and Telugu keyword lexicon and reply templates are **drafts written for the POC**. They must be reviewed by native
speakers and coastal field staff (including dialect terms used by fishing communities in Balasore, Kendrapara, Puri and Ganjam) and tested
with real message samples. The emergency numbers used (112, 1554) were checked against public sources on 2026-09-24/25; see the README.
