# Presentation and demo guide

An 8-slide structure, the live demo flow with exact commands, and what to do when
something misbehaves in front of an audience.

---

## Part 1 - Slide structure (8 slides, ~10 minutes)

### Slide 1 - Title
**Edge Vision Safety Copilot for PPE Compliance with Privacy-Preserving Inference**
Your name, roll number, course, guide's name.

*Say:* one sentence only - "A camera-based system that checks whether construction workers
are wearing their helmet and vest, running entirely on a local machine, and that proposes
what to do rather than deciding for the supervisor."

### Slide 2 - The problem
- PPE checks are done by a supervisor walking around: intermittent by nature
- The moment a violation happens is usually not the moment someone is watching
- Cameras already exist on most sites; the footage is just not used
- But sending worker footage to a cloud service raises a privacy problem of its own

*Say:* the system does not replace the supervisor - it tells them where to look. The
stakeholder table in `docs/problem_brief.md` is the one-slide backup if asked who it is for.

### Slide 3 - How it works (the architecture diagram)
`Camera → YOLO detection → person-PPE association → zone → temporal filter → compliance
rules → status + event → copilot proposal → human decision → log + audit chain`

Mark the **privacy boundary**: frames stop at the edge; only structured events cross it.

*Say:* "The neural network says what is visible. A deterministic rule engine decides whether
that is a problem. A human decides what to do about it. Those three are separate on purpose."

### Slide 4 - The model, and what it costs to run
- YOLO26n (nano), 2.51 M parameters - chosen to run in real time on a laptop
- Transfer learning from COCO: 612 of 708 tensors transferred, not trained from scratch
- 4 classes: person, helmet, vest, mask; 1 143-image seeded subset; 30 epochs, 9 min on a T4

| split | precision | recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| validation (84) | 0.885 | 0.735 | 0.828 | 0.482 |
| **test (59)** | 0.896 | 0.708 | **0.804** | 0.447 |

63.9 FPS sustained on the laptop GPU; latency p50 15.3 ms, **p95 19.1 ms**, p99 22.7 ms -
measured as a distribution, not an average. Peak memory 1.23 GB. 20.9 FPS on the CPU alone -
**no GPU required**. Safety rules: 0.03 ms per frame, 0.1 % of the time.

*Say:* the test score sitting just below validation is what an honest split looks like. And
"precision is higher than recall, which for a safety system is the wrong way round - that is
why the next slide exists."

### Slide 5 - Making a decision, not just a detection
The measured comparison - one of the two strongest slides in the deck:

| scenario | single frame | 15-frame filter |
|---|---|---|
| Detector blinks (worker IS compliant) | **30 false alarms / 200 frames** | **0** |
| Real violation (no vest) | flagged at frame 1 | flagged at frame 15 (0.6 s later) |

*Say:* "A supervisor shown thirty false alarms stops reading the alerts. Half a second of
delay on a construction site changes nothing. I measured that trade rather than assuming it."

### Slide 6 - Privacy: claimed, measured, corrected
This is the slide that shows engineering rather than assembly.

- No face recognition anywhere; anonymous tracker IDs that reset each session
- Inference local; server binds 127.0.0.1; violation log holds metadata only
- Snapshots **off by default**; when on, the face band is pixelated before writing, and a
  masking failure writes nothing at all (EVIDENCE_WITHHELD)
- **I measured whether masking destroys the evidence, and it did:**

| | helmet preserved | helmet confidence |
|---|---|---|
| first design (fixed band) | **33 %** | 0.818 → 0.567 |
| after measuring the geometry + restoring PPE boxes | **93 %** | 0.818 → 0.809 |

*Say:* "I wrote a script to test my own privacy claim and the claim failed. I measured where
the head actually sits using the dataset labels, moved the band, and made it structurally
impossible to erase helmet evidence. The metric was wrong too - it reported 129 % retention -
so detections are now matched by overlap." Then the honest wording: "inference is local and
raw camera data is not transmitted", never "100 % private".

### Slide 7 - The copilot: the machine proposes, a person decides
- Five allow-listed actions - and **nothing else can ever be proposed** (property-tested)
- One suggestion at a time, with the rule that fired and the evidence it saw
- Nothing executes without a **named** person approving it; decided once, 409 on a re-click
- Proposal and decision both go into the **hash-chained audit log** - `/api/v1/audit/verify`
- No LLM: policy is a rule table - auditable, deterministic, microseconds

*Say:* "This is the part that makes it a copilot rather than an automated enforcer. The
allow-list is published in the dashboard, so a user can read everything the system is capable
of proposing instead of trusting me." Misuse cases are in `docs/threat_model.md` if pushed.

### Slide 8 - Limitations and next steps
- Dark/navy hi-vis jackets are missed: confidence 0.070, below the 0.35 threshold - kept as a
  documented failure rather than fixed by lowering the threshold
- Recall 0.708: about three PPE items in ten are missed
- Masking is not anonymisation - build, clothing and context still identify people
- **Sensor noise defeats it**: at sigma 15 (night footage) person detection falls to 39 %,
  while darkness at gamma 2.2 is fine at 91.5 %. Brightness it tolerates, noise it does not
- Next: dark-vest training data, 50-60 epochs, real site footage, low-light evaluation, and
  measuring how often supervisors reject the copilot's advice

Finish on the gate: "`check_acceptance` reads eleven criteria - accuracy, false alarms,
privacy, latency, memory, lighting - against thresholds written in advance. It says GO. A
measurement I have not taken shows as SKIPPED, never as a pass." 

*Say:* "The threshold fix would have made my demo image pass and made the system less safe."
Examiners remember candidates who volunteer their failures.

---

## Part 2 - Live demo flow (about 5 minutes)

### Before you walk in

```powershell
cd C:\Users\Anushka\OneDrive\Documents\Desktop\ppe-safety-copilot
.venv\Scripts\activate
python -m pytest                    # 233 passed - have this on screen already
```

Close anything heavy (games, extra Chrome windows) so the GPU has free VRAM. Have
`samples\` open in File Explorer. Close any other uvicorn or `edge.detect` window - the
camera and port 8000 can only be held by one program.

### Step 1 - Start the system (20 seconds)

```powershell
python run.py
```

Point at the banner: four URLs, "Everything runs on this machine." The browser opens on the
dashboard by itself.

### Step 2 - A compliant-looking site photo (40 seconds)
Drop `samples\helmet_only.jpg` on the dashboard.

Show: annotated image with boxes, People 1, Violations 1, and the row
`Person-1 · helmet 0.93 · missing vest · MISSING_VEST`.

*Say:* "The helmet is detected and linked to this person's head region; no vest is associated,
so the rule reports MISSING_VEST."

### Step 3 - No PPE at all (30 seconds)
Drop `samples\person_only.jpg`.

Show: two people, two violations, both `MISSING_HELMET_AND_VEST`.

*Say:* "Two workers on a residential build, neither wearing PPE - the system finds both."

### Step 4 - The honest failure (45 seconds) - do NOT skip this
Drop `samples\compliant_candidate.jpg`. It reports MISSING_VEST although the worker wears a
navy hi-vis jacket.

*Say:* "This is a real limitation and I know exactly why: the vest scores 0.070, far below the
threshold, because the training data is almost all bright mesh vests. I could have lowered the
threshold to make this pass - and the system would then invent vests elsewhere and mark
unprotected workers compliant. I chose not to."

This is the moment that separates a project that works from a student who understands it.

### Step 5 - Video (45 seconds)
Scroll to the video panel, drop `samples\demo_clip.mp4`, every 5th frame.

Show: frames analysed, violation rate, the worst frame annotated, the per-frame timeline.

*Say:* "Sampling every fifth frame - 900 frames in a 30-second clip is too slow to demo, and
the verdicts do not change."

### Step 6 - The violation log (20 seconds)
Press **Refresh log** in the Recent violations panel.

*Say:* "Time, source, anonymous person label, status, missing items, confidence. No images -
that is what makes the privacy claim testable rather than a slogan."

### Step 6b - The copilot decides nothing by itself (60 seconds) - the other slide to not skip
Scroll to the **Copilot review queue**. The violations you just created are waiting there.

Show: one card per worker with the proposed action, the reason, and the rule that fired.
Open **What the copilot is allowed to propose** - the whole allow-list, five items.

Type your name in *Deciding as*, press **Reject** on one and **Approve** on another. Then
open **Approved alerts (local outbox)** and show the single line that appeared.

*Say:* "The system proposed; nothing happened until I approved it, under my name. It can
propose these five things and nothing else - that is enforced in code, not policy. And both
the proposal and my decision are now in the hash-chained audit log."

If the examiner pushes, open `/api/v1/audit` and then `/api/v1/audit/verify` in the browser:
`{"ok": true}`. *Say:* "Editing any old row breaks every hash after it, and this endpoint
finds the first bad one."

### Step 7 - Live camera (60 seconds)
Click **live camera** in the header, press **Start camera**.

Show: yourself detected as a person, `MISSING_HELMET_AND_VEST`, the live FPS tile.

*Say:* "Every frame is read, analysed and discarded on this machine. If I unplug the network
right now, this keeps working." (If you have a helmet or a hi-vis jacket, put it on - the
status changes live and it is the best 10 seconds of the demo.)

### Step 8 - Close
Return to the terminal and show the request log scrolling.

*Say:* "233 tests, none needing a GPU or a camera. The measurements behind every number are in
`docs/evaluation/`."

---

## Part 3 - If something goes wrong

| Problem | Fix, calmly | What to say |
|---|---|---|
| Port 8000 busy | `python run.py --port 8001` | "Another copy is still running - the start script catches that." |
| Camera will not open | Close any other `edge.detect` window; press Start again | "Only one program can hold a webcam." |
| First request is slow | Wait - it is the model loading | "The model loads on first use so the server starts instantly." |
| Live page blank | Ctrl+F5, press Start again | - |
| No GPU / low VRAM | Nothing - it falls back to the CPU and says so | "It checks free VRAM and uses the CPU rather than crashing - still 21 FPS." |
| Detection looks wrong on your webcam | Say so plainly | "Trained on construction photography; an indoor room is out of distribution - I would not quote my test numbers for this." |

**The golden rule:** never explain a failure by guessing. "I measured X, I did not measure Y"
is a stronger answer than a confident wrong one.

---

## Part 4 - The one-minute version

If you get cut short, say this:

> "It detects workers and their PPE from a camera, decides compliance with a deterministic
> rule engine rather than the neural network, and confirms violations over 15 frames so one
> bad frame is not an alarm - which removed all 30 false alarms in my test. Everything runs on
> the laptop: 45 FPS on the GPU, 21 on the CPU, no cloud. The test-set mAP50 is 0.80, and its
> main weakness is dark hi-vis jackets, which I documented rather than hid."
