# Threat model and misuse cases

A safety system that watches people can hurt them. This document says how, and what the
build does about it. It is written the way an engineer would write it for a review, not the
way a brochure would: the mitigations that exist are named with the file that implements
them, and the risks that remain are stated plainly rather than softened.

Scope: the system as built - a laptop running the detector, the FastAPI server and the
dashboard, with a webcam standing in for a site camera.

---

## 1. What is worth protecting

| Asset | Why it matters | Where it lives |
|---|---|---|
| Camera frames | show identifiable people, continuously | RAM only; never written unless snapshots are on |
| Event snapshots | a stored picture of a worker is personal data | `data/evidence/`, face-masked, off by default |
| Violation log | says who was non-compliant and when | `data/violations.jsonl`, metadata only |
| Audit chain | the record of who decided what | `data/server.db`, hash-chained |
| Model weights | the project's trained artefact | `models/`, 5 MB |
| The verdicts themselves | can affect someone's job | produced by deterministic rules |

The highest-value asset is not the model. It is the **link between a picture of a person
and a judgement about their behaviour**, which is what makes this system worth attacking
and worth constraining.

---

## 2. Trust boundaries

```
  camera ──▶ [ detector + rules ]  ──▶  [ server + dashboard ]  ──▶ human decision
             all in one process            127.0.0.1 only            named actor
                                                │
                                                └──▶ data/outbox.jsonl  (the seam where
                                                      a real site system would attach)
```

Three boundaries matter:

1. **Camera → process.** Frames enter memory and are dropped. Nothing crosses to disk
   unless `PPE_STORE_SNAPSHOTS=true`, and then only face-masked (`shared/privacy.py`).
2. **Process → network.** `run.py` binds `127.0.0.1`, not `0.0.0.0`, so the dashboard is
   not reachable from the LAN by default. No frame is ever sent to an external service;
   there is no external service.
3. **Machine → action.** Only an approved copilot action writes to the outbox, and only
   with a named actor (`server/routes/copilot.py`).

---

## 3. Threats, in STRIDE order

| # | Threat | Realistic form | What the build does | Residual risk |
|---|---|---|---|---|
| T1 | **Spoofing** an identity | approving an escalation as somebody else | sign-in with hashed passwords; the actor comes from the signed-in session, never from the request body (D-027) | Reduced. No rate limiting or lockout; no TLS (127.0.0.1 only) |
| T2 | **Tampering** with the record | editing `violations.jsonl` or the events table | audit chain: editing a row breaks every hash after it; `/api/v1/audit/verify` finds the first bad row | Detection, not prevention. Whoever owns the file can rewrite the whole chain. |
| T3 | **Repudiation** - "I never approved that" | disputed escalation | every proposal and decision is an audit row with actor and time | Bounded by T1 |
| T4 | **Information disclosure** | stored snapshot leaks | snapshots off by default; when on, face band pixelated, fail-closed | Masking is not anonymisation (§5) |
| T5 | **Denial of service** | camera unplugged, model missing | camera released on disconnect; missing weights return 503 rather than crashing | Accepted |
| T6 | **Elevation of privilege** | a request asking the copilot to do something not on the list, or a viewer approving an action | `check_action()` refuses any id outside `ACTIONS`; repeat counts read from the DB, not trusted from the caller; `require_role` gates deciding, and a forged session cookie fails its signature | Low |
| T7 | **Model evasion** | dark hi-vis jacket, occlusion, turning away | measured and documented (D-014); UNCERTAIN state; human review | **Real and unsolved.** See §5. |
| T8 | **Automation bias** | supervisor approves every suggestion without looking | one suggestion at a time, reason and rule shown, approval requires a name | Behavioural, not technical |

---

## 4. Misuse and abuse cases

These are the uses the system must *not* be quietly good at. Each one is a design
constraint, not a disclaimer.

**M1 - Turning it into surveillance of individuals.**
The obvious abuse: log who is careless, use it at appraisal time. The build makes this
hard rather than merely discouraged. There is no face recognition and no identity store;
a worker is `Worker-017` inside one session and the numbering restarts every run, so the
log cannot be joined across days. *Residual:* a small site with four workers is
identifiable by context alone. Technology does not fix that - site policy does.

**M2 - Productivity or behaviour monitoring.**
The same pipeline could count how long someone stands still. It does not: nothing measures
dwell time, and the log stores a verdict, not a timeline. Adding that would be a change in
purpose, and should require the same review as a new system.

**M3 - Automated discipline.**
The copilot cannot discipline anyone: its whole vocabulary is five actions, none of which
touch a person's record, and none of which execute without a named human. Anyone extending
the allow-list is making a policy decision, and the allow-list is deliberately in one small
file so that decision is visible in a diff.

**M4 - Evidence laundering.**
A masked snapshot could be presented as proof of wrongdoing. The honest limits are written
into the evaluation: `preserved` is a *detector* measure, the model's recall is 0.708, and
one frame is not proof. The UNCERTAIN state exists so the system can say "I could not tell"
rather than guessing.

**M5 - Silent scope creep in the camera.**
A webcam pointed at a site is also pointed at whoever walks past. The system cannot tell a
worker from a visitor, and treats both identically - which means a visitor without a helmet
generates a violation. Documented as a limitation; a real deployment needs signage and a
defined camera field of view.

**M6 - Using it where it was not measured.**
Trained on construction-site photography. Quoting the test mAP for an indoor webcam demo
would be dishonest, so the README and the evaluation both say so in the sections where
someone would otherwise copy the number.

---

## 5. What the system genuinely cannot do

- **Masking is not anonymisation.** Build, clothing, gait and context still identify people.
  The band is placed by geometry, not by finding the face, so an unusual pose can leave part
  of a face visible.
- **Recall is 0.708.** The detector misses PPE more often than it invents it. That is the
  safer direction, but it means absence of a violation is not evidence of compliance.
- **One camera, one viewpoint.** Occlusion is not solved by temporal filtering, only smoothed.
- **Sign-in is local and minimal.** Passwords are hashed and sessions signed, but there is
  no rate limiting, no lockout and no TLS; read access to the console is not gated at all.
  Deciding is.

---

## 6. The three things a real deployment would need first

1. **Hardening the sign-in that now exists** (D-027) - rate limiting, account lockout, and
   TLS with a real identity provider. The current design is honest about being a prototype:
   local-only, no transport encryption, no lockout.
2. **Retention enforcement** - `evidence_retention_days` exists as a setting; nothing
   deletes old snapshots yet.
3. **Consent and signage** - a legal and organisational control, not a technical one, and
   the first thing a workplace-privacy review would ask for.

---

## 7. How this document is kept honest

Every mitigation above names the file that implements it, and every one has a test in
`tests/`. Claims with neither are written here as residual risk instead.
