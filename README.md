# Edge Vision Safety Copilot for PPE Compliance with Privacy-Preserving Inference

A camera-based system that detects construction workers and their PPE, decides whether the
required equipment (helmet and vest) is being worn, and records violations — running
**entirely on the local machine**, so camera footage never leaves it.

BSc Artificial Intelligence final-year project (BAI-01) · Windows + VS Code · Python 3.11

```
Camera / image / video  →  YOLO detection  →  person-PPE association  →  zone
                        →  temporal filter  →  compliance rules  →  status + event
                        →  FastAPI backend  →  dashboard  →  violation log
```

| | |
|---|---|
| Model | YOLO26n (nano), fine-tuned, 2.51 M parameters, 5.4 MB |
| Test accuracy | mAP50 **0.804**, precision 0.896, recall 0.708 |
| Speed | **45.2 FPS** on a laptop GPU, **20.9 FPS** on the CPU — no GPU required |
| Tests | **184**, none needing a GPU, a camera or the model file |

---

## Quick start

```powershell
cd C:\Users\Anushka\OneDrive\Documents\Desktop\ppe-safety-copilot
.venv\Scripts\activate
python -m pytest          # 233 passed
python run.py             # starts everything, opens the dashboard
```

First-time setup, if the virtual environment does not exist yet:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

PyTorch is installed first and separately so pip does not fall back to the CPU-only build.
No GPU? Skip that line — everything still runs, about half as fast.

---

## Project structure

```
ppe-safety-copilot/
├── run.py            ONE command: starts the backend + dashboard
├── frontend/         the web pages (plain HTML + JS, no build step)
├── edge/             detection, association, zones, temporal filter, compliance, events
├── server/           FastAPI app, routes, settings, database, audit log
├── shared/           schemas, violation log, device selection, snapshot masking, copilot rules
├── training/         dataset preparation, training, validation, the Colab notebook
├── scripts/          dataset checks, benchmarks, experiments, environment check
├── datasets/ppe4/    the 1 143-image dataset (not committed — rebuilt by script)
├── models/           trained weights (not committed)
├── samples/          demo images and a demo clip
├── configs/          zones.yaml, policies.yaml — site rules, no code changes needed
├── tests/            233 pytest tests
└── docs/             model card, architecture, decisions, evaluation, threat model,
                      problem brief, viva Q&A, slides
```

---

## How the system works

1. **Detection** — YOLO26n finds `person`, `helmet`, `vest`, `mask` boxes in one forward pass.
2. **Association** (`edge/association.py`) — each person box gets a *head* region and a *torso*
   region. A helmet or mask must overlap the head region, a vest the torso region, with at
   least 50 % of the PPE box inside it. Each PPE box goes to **one** person, so one helmet
   cannot make two people compliant.
3. **Zone** (`edge/zones.py`) — the worker's foot point decides which polygon from
   `configs/zones.yaml` they stand in, and each zone type carries its own required PPE.
4. **Temporal filter** (`edge/temporal.py`) — a 15-frame sliding window per worker per item.
   10+ missing = violation, 5–9 = UNCERTAIN, under 5 = the detector blinked.
5. **Compliance** (`edge/compliance.py`) — a deterministic rule engine produces the verdict.
6. **Event + log** — a violation is written to `data/violations.jsonl` (metadata only).
7. **Copilot** (`shared/copilot.py`) — proposes *one* action from a fixed list of five, with
   the rule and the evidence attached, and waits. A named person approves or rejects it in
   the dashboard; only then does anything happen, and both the proposal and the decision go
   into the hash-chained audit log.

**The design rule:** the neural network says *what is visible*; a separate, deterministic
rule engine decides *whether that is a problem*; and a human decides *what to do about it*. The rules are unit-tested with hand-written
boxes — no GPU, no camera — and cost 0.03 ms per frame, about 0.1 % of the time.

---

## Using it

### The dashboard — <http://127.0.0.1:8000/dashboard>

Upload an image or a video, see the annotated result, per-person statuses, and the violation
log. Statuses are `COMPLIANT`, `MISSING_HELMET`, `MISSING_VEST`, `MISSING_HELMET_AND_VEST`.

Try the samples:

| Image | Expected |
|---|---|
| `samples\helmet_only.jpg` | MISSING_VEST |
| `samples\vest_only.jpg` | MISSING_HELMET |
| `samples\person_only.jpg` | 2 people, both MISSING_HELMET_AND_VEST |
| `samples\compliant_candidate.jpg` | MISSING_VEST — a documented model limitation, see below |

### Live camera — <http://127.0.0.1:8000/live>

Press **Start camera**. The page shows the annotated feed with live people / violations / FPS
tiles. The stream is `multipart/x-mixed-replace` (one JPEG after another), which a plain
`<img>` renders natively — no video library. The camera is released when you press Stop or
close the tab. Only one program can hold the webcam: close any `python -m edge.detect
--source 0` window first.

### Command line

```powershell
python -m edge.detect --source samples\helmet_only.jpg     # one image
python -m edge.detect --source samples\demo_clip.mp4       # a video
python -m edge.detect --source 0                           # webcam (Q or Esc to quit)
python -m edge.run_edge                                    # full pipeline: tracking, zones, temporal, events
python -m edge.run_edge --mode baseline                    # single-frame baseline, for comparison
```

### API

| Endpoint | Purpose |
|---|---|
| `GET /` | service info and the list of endpoints |
| `GET /health` | health check (also checks the database) |
| `POST /api/v1/predict` (aliases `/detect/image`, `/predict`) | one image → per-person verdicts + annotated image |
| `POST /detect/video` | a clip → sampled-frame summary, timeline, worst frame |
| `GET /live/stream`, `GET /live/status` | MJPEG camera stream, latest verdicts |
| `GET /api/v1/violations?limit=50` | the local violation log |
| `GET /docs` | interactive API documentation |

```powershell
curl.exe -F "image=@samples\person_only.jpg" -F "annotate=false" http://127.0.0.1:8000/detect/image
curl.exe -F "video=@samples\demo_clip.mp4" -F "every_nth=5" http://127.0.0.1:8000/detect/video
```

Video analysis **samples frames** (every 5th by default): a 30-second clip at 30 fps is 900
frames, too slow for a demo, and the verdicts do not change. The response always reports
`frames_analysed`. The temporal filter is not applied there — it needs consecutive frames.

---

## Where the work runs

| Job | Where | Why |
|---|---|---|
| Backend, dashboard, live camera, tests, demo | **Laptop**, device `auto` | GPU when it has free VRAM (45 FPS), CPU when it does not (21 FPS) |
| Training and heavy evaluation | **Google Colab T4** | a free GPU that is not this laptop |

`shared/device.py` checks that the GPU has at least 1500 MB of VRAM free before using it, and
falls back to the CPU otherwise — printing which device it chose and why. A slowdown, never an
out-of-memory crash mid-demo. Override with `--device cpu`, `--device 0`, or
`PPE_DETECT_DEVICE` in `.env`.

---

## Dataset

1 143 images — **1 000 train / 84 validation / 59 test** — a seeded subset of the Roboflow
Universe *Construction Site Safety* dataset (CC BY 4.0), remapped to four classes.

| split | person | helmet | vest | mask |
|---|---|---|---|---|
| train | 3 641 | 1 267 | 1 175 | 631 |
| val | 188 | 94 | 55 | 27 |
| test | 175 | 111 | 60 | 29 |

```powershell
python -m training.prepare_ppe4 --src "<downloaded YOLOv8 export .zip>"
python -m scripts.check_dataset            # DATASET OK (1143 images)
```

`prepare_ppe4.py` groups images by the **original photo** (Roboflow names augmented copies
`<photo>_jpg.rf.<hash>`), so the same photo can never land in two splits, then samples
class-aware with seed 42 so `mask` and `vest` survive. Same source + same seed = identical
dataset.

`check_dataset.py` verifies counts, missing labels and images, corrupt images, malformed label
lines, boxes outside 0–1, invalid class ids, and the class distribution.

---

## Model and results

YOLO26n fine-tuned from COCO weights (**transfer learning**, not from scratch: 612 of 708
tensors transferred). 30 epochs, batch 16, 640 px, patience 8, seed 42, `mosaic 0.0` — 560
seconds on a Colab T4.

`mosaic` is off because the training images are *already* Roboflow mosaics; mosaicking them
again would shrink every object to a few pixels.

| split | precision | recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| validation (84) | 0.885 | 0.735 | 0.828 | 0.482 |
| **test (59)** | 0.896 | 0.708 | **0.804** | 0.447 |

Per class on the test split — mAP50: person 0.771 · helmet 0.874 · vest 0.782 · mask 0.788.

### Training on Colab

```powershell
python -m scripts.make_colab_bundle      # packs code + dataset into colab_bundle.zip
```

Upload it to `MyDrive/ppe_project/`, open `training/colab_train.ipynb` in Colab with a T4
runtime, run the cells. Cell 10 downloads the weights and reports; put `best.pt` into
`models\` and the JSON/CSV/PNG files into `docs\evaluation\`.

---

## Edge performance

`python -m scripts.benchmark_edge` — 120 frames, warm-up discarded, 640 px:

| device | detect ms | rules ms | draw ms | total ms | FPS |
|---|---|---|---|---|---|
| RTX 3050 laptop GPU | 21.25 | 0.03 | 0.83 | 22.12 | **45.2** |
| CPU (i7-12650H) | 47.07 | 0.03 | 0.78 | 47.88 | **20.9** |

All the time is in the neural network. The safety logic costs 0.03 ms — so making the rules
more careful is effectively free, and only the detector is worth optimising.

### Does the temporal filter earn its place?

`python -m scripts.compare_modes` measures it:

| scenario | single frame | 15-frame filter |
|---|---|---|
| Detector blinks (worker IS compliant) | **30 false alarms / 200 frames** | **0** |
| Real violation (no vest) | flagged at frame 1 | flagged at frame 15 (0.6 s later) |

A supervisor shown thirty false alarms stops reading the alerts; half a second of delay on a
site changes nothing.

---

## Measured on this machine

| | value | required |
|---|---|---|
| Frame latency p50 / p95 / p99 | 15.3 / 19.1 / 22.7 ms | p95 ≤ 200 ms |
| Sustained frame rate | 63.9 FPS | ≥ 8 FPS |
| Peak memory | 1.23 GB (580 MB of it the model) | ≤ 2.5 GB |
| People found in dim light (gamma 1.6) | 94.9 % | ≥ 60 % |
| Helmets found across all lighting tested | 96.8 % | — |

`python -m scripts.check_acceptance` reads these back and prints **GO** or **NO-GO** against
eleven criteria in `configs/acceptance.yaml`. It currently reads **GO**. A missing report is
reported as SKIPPED, never as a pass.

One measured failure worth knowing: at sensor noise sigma 15 — what night footage looks like —
person detection falls to **39 %**. Brightness the model tolerates; noise it does not. Night
operation is out of scope, and that number is why.

---

## The copilot: what it may propose, what only you may decide

The system never acts on its own. It proposes, from a list you can read in full at
`/api/v1/copilot/catalog` and in the dashboard's **Copilot review queue** panel:

| Action | When it is proposed | What approving it does |
|---|---|---|
| Record and take no action | fallback | nothing beyond the existing log line |
| Flag for supervisor review | first violation seen (R5) | adds a review flag |
| Notify the area supervisor | same worker seen 3+ times (R4) | appends to `data/outbox.jsonl` |
| Escalate: restricted zone | required PPE missing in a RESTRICTED zone (R3) | high-priority outbox line |
| Ask for a second look | the verdict was UNCERTAIN (R2) | marks it for another observation |

**Who may decide.** Approving or rejecting needs a signed-in account with the `supervisor`
or `admin` role. Create one:

```powershell
python -m scripts.make_user --username ana --role supervisor --display-name "Ana Silva"
```

Passwords are hashed (PBKDF2, per-user salt) into `configs/users.yaml`, which is gitignored.
Roles: `viewer` sees the site, `supervisor` decides, `admin` also configures. The decision is
recorded against the **signed-in account**, never a typed name — see `docs/decisions.md`
D-027.

Three properties, each enforced in code and covered by tests:

- **Allow-listed** — `check_action()` gates every path that stores or executes anything, so
  an action id invented by a caller is refused. A property test sweeps every status × zone ×
  repeat-count combination and asserts nothing outside the list can ever be produced.
- **Human-governed** — a suggestion is inert until `decide` arrives from an authenticated
  supervisor, and it can be decided only once. There is no auto-approve and no confidence
  threshold that skips review.
- **Traceable** — proposal and decision both append to the tamper-evident audit chain
  (`/api/v1/audit`, `/api/v1/audit/verify`), one as `copilot`, one as the person.

There is **no language model** in this layer. "Missing helmet in a restricted zone →
escalate" is a policy, not a judgement, and a rule table is auditable, deterministic and
instant. Details and the reasoning: `docs/decisions.md` D-026.

Approved alerts are written to a **local outbox file** — nothing is sent anywhere. That file
is the seam where a real site would attach a siren or an SMS gateway.

---

## Privacy

The defensible claim: **inference runs locally and this system does not transmit raw camera
data.** Not "100 % private" — if snapshot evidence were enabled, images would exist on that
machine.

| Claim | How it is enforced | Test |
|---|---|---|
| Uploaded images are not stored | decoded in memory; response says `image_stored: false` | `test_the_image_is_never_stored` |
| Uploaded videos are not kept | temp file deleted in a `finally` block | `test_the_uploaded_clip_is_deleted_from_the_temp_folder` |
| The violation log holds no images | metadata-only writer; stores a path + privacy status | `test_no_image_or_frame_is_ever_written` |
| Stored evidence is masked | face band pixelated before writing; fail-closed if masking fails | `test_the_saved_image_really_is_the_masked_one`, `test_nothing_is_written_when_masking_cannot_be_done` |
| The camera is released | `capture.release()` on disconnect | `test_status_reflects_the_last_frame_and_camera_is_released` |
| The server stays on this machine | `run.py` binds 127.0.0.1, not 0.0.0.0 | `test_defaults_are_local_only` |

Snapshots are **off by default** — the system stores metadata only. Turn them on with
`PPE_STORE_SNAPSHOTS=true` and every stored frame has the face band pixelated first
(`shared/privacy.py`); if masking fails, nothing is written and the event is marked
`EVIDENCE_WITHHELD`.

The band sits *below* the helmet line so the evidence survives. Where exactly that is was
**measured, not guessed** — and the first version was wrong: `scripts/privacy_utility.py`
showed a fixed band destroying two thirds of the helmet detections, and
`scripts/face_band_geometry.py` showed why (the head sits at a different fraction of the
person box depending on the box's shape). The band is now placed from that measurement, and
detected helmet/vest boxes are restored after pixelation so masking cannot erase the
evidence at all. Full story: `docs/decisions.md` D-025.

Masking is *not* anonymisation, and the project does not claim it is: build, clothing and
context can still identify someone.

There is **no face recognition** anywhere: the system detects *a person*, never *which*
person. Tracker IDs are anonymous and reset every session.

---

## Known limitations

1. **Dark / navy hi-vis jackets are missed.** On `samples/compliant_candidate.jpg` the vest
   scores **0.070** — below the 0.35 threshold — in five fragmented boxes over the correct
   torso. Not an association bug (the best box lies 70 % inside the torso region) and not a
   class-mapping bug: the training data is almost all bright mesh vests. **Not fixed by
   lowering the threshold**, because a false "vest present" marks an unprotected worker
   compliant — the failure that actually hurts someone.
2. **Recall 0.708 < precision 0.896** — about three PPE items in ten are missed, the dangerous
   direction for safety. Mitigated by the temporal filter, the UNCERTAIN state and human
   review; not solved.
3. **Small evaluation sets** — 84 validation and 59 test images; `mask` has 29 instances, so
   its 1.000 precision means "no false masks in 29", not "solved".
4. **Training images are pre-augmented mosaics**, evaluation images are clean photos.
5. **Domain gap** — trained on construction photography; an indoor webcam demo performs
   materially worse, and these numbers should not be quoted for it.
6. **The model is under-trained, not overfitted** — losses were still falling at epoch 30 and
   early stopping never fired. 50–60 epochs is the obvious cheap improvement.

---

## Testing

```powershell
python -m pytest        # 233 passed, ~14 seconds
```

No test needs a GPU, a camera or the model file — the detector is replaced by a fake wherever
it would be required. Coverage spans detection plumbing, association geometry, compliance
rules, the temporal filter, zones, all three APIs, both web pages, the violation log, dataset
preparation, the Colab bundle, the start command, snapshot masking and the copilot's
allow-list, approval gate and audit trail.

---

## Command reference

| Command | What it does |
|---|---|
| `python -m scripts.check_acceptance` | Go / no-go: every claim against its threshold |
| `python -m scripts.measure_resources` | Memory, latency percentiles, sustained frame rate |
| `python -m scripts.robustness_lighting` | How much detection survives dim, bright and noisy light |
| `python -m scripts.retention --dry-run` | What stored evidence has expired |
| `python -m scripts.make_user --list` | Sign-in accounts (never hashes) |
| `python run.py` | start backend + dashboard (`--port`, `--no-browser`, `--reload`) |
| `python -m pytest` | run all tests |
| `python -m scripts.check_env` | verify Python, GPU, webcam, packages |
| `python -m scripts.check_dataset` | validate the dataset |
| `python -m training.prepare_ppe4 --src <zip>` | rebuild the dataset subset |
| `python -m training.train` | train (laptop; Colab is the default) |
| `python -m training.validate --weights models\ppe4_yolo26n_best.pt --split test` | test-set metrics |
| `python -m edge.detect --source <image\|video\|0>` | single-frame detection demo |
| `python -m edge.run_edge` | full pipeline with tracking, zones, temporal filter, events |
| `python -m scripts.benchmark_edge` | measure FPS and latency |
| `python -m scripts.compare_modes` | temporal filter vs single-frame baseline |
| `python -m scripts.make_colab_bundle` | pack code + dataset for Colab |
| `python -m scripts.make_demo_clip` | rebuild `samples/demo_clip.mp4` |
| `python -m scripts.draw_zones` | draw safety zones on a camera snapshot |

---

## Documentation

| File | What it holds |
|---|---|
| `docs/model_card.md` | Model + system card: intended use, metrics, failure modes, ethics |
| `docs/architecture/architecture.md` | Architecture and data-flow diagrams, contracts, failure behaviour |
| `docs/decisions.md` | Every engineering decision with its date and reason (D-001 onwards) |
| `docs/evaluation/system_evaluation.md` | Every measured number, with the command that produced it |
| `docs/evaluation/detector_results.md` | Detector results, interpretation and limitations |
| `docs/admin_guide.md` | Install, settings, accounts, routine jobs, backups, troubleshooting |
| `docs/test_strategy.md` | What is tested, what is not, and why the suite needs no hardware |
| `docs/data_dictionary.md` | Every stored field, what it means, and why it is safe to keep |
| `CHANGELOG.md` | Release notes per milestone, including what each one left open |
| `docs/threat_model.md` | Assets, trust boundaries, STRIDE threats, misuse cases, residual risk |
| `docs/problem_brief.md` | The industry problem, stakeholders, user stories, scope exclusions, backlog |
| `docs/viva_qa.md` | Viva questions with answers |
| `docs/presentation.md` | 8-slide structure, demo flow, what to do if something breaks |
| `docs/tech_notes.md` | WHAT / WHY / HOW for each technology |
| `frontend/README.md` | The web pages and why there is no build step |
| `samples/README.txt` | Provenance and licence of the demo media |

---

## Licences

* **Code** — Ultralytics is AGPL-3.0, so this repository should be released under a
  compatible licence.
* **Dataset** — Roboflow Universe *Construction Site Safety*, CC BY 4.0 (attribution
  required). Cite it in the report.
