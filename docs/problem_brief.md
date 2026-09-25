# Industry problem brief

Why this system exists, who it is for, what it deliberately does not do, and what would
come next. Written before the build would have been ideal; written from the build, with the
scope decisions that were actually made, is what this is.

---

## 1. The problem

On construction and industrial sites, personal protective equipment is mandatory and
routinely not worn - most often for short periods, in exactly the places where the risk is
highest. Compliance is checked by a supervisor walking the site. That has three failure
modes: a supervisor cannot be everywhere, a walk-round is a sample rather than a measure,
and by the time a near-miss is investigated there is no record of what people were actually
wearing.

The gap is therefore not "detect helmets". It is **continuous, low-cost observation that a
supervisor can act on, without turning the site into a surveillance operation.** The second
half of that sentence is what makes it a real engineering problem rather than a detection
exercise: a system that solves the first half and ignores the second will be switched off by
the workforce, or should be.

---

## 2. Stakeholders

| Stakeholder | What they want | What they fear | How the build answers |
|---|---|---|---|
| Site safety supervisor | to know where the risk is now, without watching a monitor | alert fatigue; being blamed for a machine's call | one proposal at a time, with the reason; the decision is theirs and is recorded |
| Worker | to do the job; to be treated fairly | being tracked, ranked, or disciplined by a camera | no face recognition, no identity, anonymous session-scoped labels |
| Site manager | evidence for the shift report and for audits | data-protection exposure | metadata-only log; images off by default and masked when on |
| Safety officer / auditor | to reconstruct what happened and who decided what | an unauditable black box | hash-chained audit log of every proposal and decision |
| IT | something that does not become their problem | cloud bills, data egress, another server to run | one local process, no external service, no account |

The two stakeholders in tension are the supervisor (who wants more information) and the
worker (who wants less). Almost every design decision in `docs/decisions.md` is a point on
that line, which is why they are written down individually.

---

## 3. User stories

**Supervisor**
- As a supervisor, I want to see which workers are missing PPE *right now*, so I can
  intervene before something happens.
- As a supervisor, I want the system to tell me *why* it thinks so, so I can judge whether
  it is right.
- As a supervisor, I want to approve or dismiss what it proposes, so I stay accountable for
  what happens on my site.
- As a supervisor, I do **not** want sixty alerts about the same person.

**Worker**
- As a worker, I want to know the system does not recognise my face, so I can trust it is
  about equipment and not about me.
- As a worker, I want an unclear case to be checked rather than logged as a violation.

**Safety officer**
- As a safety officer, I want to reconstruct a decision months later, including who took it.
- As a safety officer, I want to know how often the machine is wrong.

**Each of these maps to something that exists:** the live view, the reason and rule shown on
every suggestion, the approve/reject gate, the logging cooldown, the absence of face
recognition, the UNCERTAIN state, the audit chain, and the evaluation numbers.

---

## 4. Scope - what this is not

Excluded deliberately, each for a reason:

| Excluded | Why |
|---|---|
| Face recognition / worker identity | it is the abuse this system is designed to avoid |
| Cloud inference or storage | frames would leave the machine; the privacy claim would be gone |
| Login and user accounts | out of the brief's scope; a half-built auth system is worse than a stated gap |
| Multi-camera tracking, re-identification across cameras | that *is* person tracking, and it is the line this project does not cross |
| Automatic alerting to a siren or SMS | nothing should act without a human; the outbox is the seam where it would attach |
| A language model anywhere | policy is a rule table: auditable, deterministic, instant |
| Kubernetes, microservices, a model registry, a message queue | one process on one laptop, by design |
| Dedicated edge hardware (Jetson, Pi) | the brief allows emulation; a laptop webcam stands in |

---

## 5. Success criteria, and whether they were met

| Criterion | Target | Actual |
|---|---|---|
| Detects the four classes usefully | mAP50 > 0.7 | **0.804** (test split) |
| Runs at a usable rate on a laptop | > 10 FPS | measured, see `docs/evaluation/` |
| False alarms controlled | fewer than the single-frame baseline | **30 → 0** with the temporal filter |
| Stores no personal data by default | no images written | enforced and tested |
| Every action human-approved | no autonomous action | enforced by `check_action` + decide gate |
| Claims backed by measurement | no unmeasured claim in the README | the privacy claim was measured, **failed**, and was fixed (D-025) |

---

## 6. Backlog - what comes next, in order

1. **Authentication and roles** - the actor field is typed, not proven (threat T1).
2. **Retention enforcement** - delete snapshots past `evidence_retention_days`.
3. **Better vest recall on dark hi-vis** - the known detection failure (D-014); needs more
   training data of that specific case, not a lower threshold.
4. **Copilot effectiveness measurement** - the table already stores approvals and rejections;
   the rejection rate is the number that says whether the copilot is worth having.
5. **Multi-camera support** - as separate independent instances, not as cross-camera
   tracking, to keep the scope exclusion in §4 intact.
6. **Real hardware port** - ONNX export and a Jetson/Pi benchmark, if hardware is available.

---

## 7. The honest summary

This is a small system that does one thing and refuses to do several adjacent things that
would be easy to add. The refusals are the design.
