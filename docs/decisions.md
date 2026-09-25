# Engineering decisions log

Record every important choice here with the date and the reason. This becomes a
section of the report and helps in the viva ("why did you choose X?").

## D-001 Base model: Ultralytics YOLO26n (pretrained), fine-tuned
- Date: Day 1
- Why: nano model (about 2.4 M parameters per Ultralytics docs) fits a 4 GB GPU for
  training at batch 8 and runs in real time for inference; designed for edge deployment
  (NMS-free end-to-end inference option). Transfer learning from COCO weights.
- Fallback: yolo11n.pt if any YOLO26 issue appears (same API).
- Not chosen: s/m/l/x variants (slower, more VRAM, little benefit for a webcam demo).

## D-002 Dataset: SH17 (Kaggle / Zenodo), remapped to 4 classes
- Source: https://github.com/ahmadmughees/SH17dataset (paper: arXiv 2407.04590)
- Licence: CC BY-NC-SA 4.0
- Classes used: Person, Helmet, Safety-vest, Face-mask-medical -> person, helmet, vest, mask
- Split: official train/test lists; validation = 10 % of official train (seed 42); official test untouched.
- Real counts: see datasets/ppe4/stats.json (copy the table here after running prepare_sh17).
- Known limitations: images are stock photos (Pexels), mostly high-resolution and not
  webcam-like; helmet / vest / mask instances are far fewer than person instances (class
  imbalance); "mask" means medical face mask only (not respirators).
- Download date: ____    Downloaded by: ____

## D-003 Zones are image polygons, not GPS
- A webcam cannot know GPS location; zones are configured polygons in pixel coordinates per camera.

## D-004 Worker IDs are anonymous tracker IDs
- ByteTrack (motion + box overlap), no face features. IDs restart per edge session, so the DB key is a UUID.

## D-005 Person-PPE association by body regions
- Helmet/mask must lie >= 50 % (IoA) inside the person's head region, vest inside the torso region.
- Each PPE box goes to exactly one person (best overlap, then nearest). Held-in-hand PPE is not "worn".
- Limitation: heavy overlap between people can mis-assign; mitigated by temporal filter + human review.

## D-006 Temporal filter and decision thresholds
- Window 15 frames per worker per item; missing >= 10 -> POTENTIAL_VIOLATION, 5-9 -> UNCERTAIN, < 5 -> worn.
- Fewer than 15 frames observed -> UNCERTAIN (INSUFFICIENT_EVIDENCE). Zone change resets the window.
- Values live in configs/policies.yaml; to be tuned on VALIDATION clips, never on the final test clips.

## D-007 Events only on state change + 30 s cooldown
- Prevents flooding the backend with one event per frame; one sustained violation = one event.

## D-008 Restricted zones without identity
- The system cannot know who is "authorized" without identifying people, so any presence is flagged
  RESTRICTED_ZONE_ENTRY for human review, unless a supervisor-approved time window is open.

## D-009 Edge device (measured Day 1, see docs/evaluation/env_report.json)
- GPU: NVIDIA GeForce RTX 3050 A Laptop GPU, 4 GB VRAM, driver 592.82; PyTorch 2.11.0+cu128; Ultralytics 8.4.158.
- Webcam delivers 640x480 (the 1280x720 request is not supported by this camera); zones are rescaled automatically.
- Day 1 smoke test (yolo26n.pt COCO, imgsz 640, GPU FP16, 347 frames, warm-up excluded):
  inference 21.26 ms mean / 26.41 ms p95, full loop 63.95 ms mean = 15.64 FPS.
  Source: docs/evaluation/day1_smoke_benchmark.csv
- Training runs on Google Colab (T4), not on this laptop, to keep the edge device's load light.

## D-010 Dataset switched to a SMALL seeded subset (Roboflow Construction Site Safety)
- Date: Day 3
- What changed: the primary dataset is now a ~1000 / 250 / 250 image subset built by
  `training/prepare_ppe4.py`, not the full SH17 download.
- Source: Roboflow Universe, "Construction Site Safety" by Roboflow Universe Projects
  (https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety),
  v27, 2 801 images, YOLOv8 export, licence CC BY 4.0 (attribution required).
- Classes used: Person -> person, Hardhat -> helmet, Safety Vest -> vest, Mask -> mask.
  The NO-Hardhat / NO-Mask / NO-Safety-Vest classes and all site-object classes
  (machinery, vehicles, cones, ladders) are dropped: absence of PPE is DERIVED by the
  rule engine from person-PPE association, it is not a detector class. This keeps the
  model small and the decision auditable.
- Subset method: source test split is kept as our test split (never trained or tuned on);
  train and val are sampled from the source train/valid splits with seed 42 by a
  class-aware picker that guarantees a minimum number of images per class before filling
  the rest at random, so the rare classes (mask, vest) are not lost. Rebuild is exactly
  reproducible: same zip + same seed = same subset (recorded in subset_manifest.json).
- Images are downscaled to 640 px on the longest side; YOLO labels are normalised, so
  they remain correct after resizing. Exact duplicate images across train/val and test
  are detected by MD5 and reported.
- Why: the project is optimised for a working, demonstrable system on a 4 GB laptop GPU -
  small dataset, short training, fast iteration - not for benchmark accuracy.
- SH17 remains available via `training/prepare_sh17.py` if a larger dataset is wanted
  later; D-002 still records its licence and limitations.
- Real counts: see datasets/ppe4/stats.json and docs/evaluation/dataset_summary.json.
- Download date: ____    Downloaded by: ____

## D-010a Dataset obtained from a public mirror; augmentation and split notes
- Date: Day 3
- Roboflow Universe requires an account to download, which was not usable, so the same
  CC BY 4.0 export was taken from a public GitHub mirror that redistributes it
  (VoxDroid/Construction-Site-Safety-PPE-Detection, Model-Training/Dataset). The dataset
  itself, its licence and its citation stay the Roboflow Universe one (see D-010).
- The mirror's TRAIN split is pre-augmented by Roboflow: every training image is a 4-photo
  mosaic with cutout squares (2605 files from 514 photos). VALID and TEST are clean single
  photos. Consequences we accepted and must state in the report:
  * training images are mosaics -> Ultralytics' own mosaic augmentation is turned OFF
    (`mosaic=0.0`) so the model does not train on mosaics of mosaics;
  * evaluation uses the clean splits only, so the reported metrics are on realistic
    pictures, not on augmented collages;
  * val = 84 and test = 59 images: smaller than the 200-300 originally planned, because
    that is all the clean held-out material this source has. Padding them with augmented
    copies of training photos was rejected - it would inflate the numbers and leak.
    Metrics on 59 test images are noisy; the report must give per-class counts alongside.
- 6 photos appeared in more than one source split; they were assigned to the evaluation
  split (test > val > train), so no photo is both trained on and tested on.
- Real counts: datasets/ppe4/stats.json, docs/evaluation/dataset_summary.json.

## D-011 Training settings for the small dataset
- Date: Day 3
- 30 epochs, batch 8, imgsz 640, patience 8 (early stopping), seed 42, AMP on, workers 2.
  Rationale: 1000 training images + a nano model; 100+ epochs would only overfit and cost
  an hour of laptop GPU time. Early stopping keeps the best epoch by validation mAP.
- `mosaic=0.0` and `close_mosaic=0`: the source training images are already mosaics
  (D-010a). Horizontal flip stays at 0.5 - PPE looks the same mirrored.
- Training runs on the laptop (RTX 3050, 4 GB) by default; Colab is the fallback if
  training gets heavier later. This reverses the Day-1 plan, which assumed the full SH17
  dataset.
- Checkpoints: Ultralytics writes best.pt and last.pt into runs/train/<name>/weights/;
  train.py copies both into models/ and writes models/<name>_info.json (library versions,
  arguments, GPU, data.yaml hash) so a run can be reproduced and cited in the report.
- Interruption safety: `python -m training.train --resume` continues from last.pt.

## D-011a Training moved to Google Colab; laptop GPU reserved for the live demo
- Date: Day 3
- The laptop GPU is not used for training. Training runs on Colab's free T4; the laptop is
  for development, the backend, tests and the webcam edge demo.
- Transfer method: `python -m scripts.make_colab_bundle` packs training/, scripts/ and
  datasets/ppe4/ into one ~69 MB `colab_bundle.zip` that is uploaded to Google Drive. This
  removes the earlier dependency on a GitHub clone and a Kaggle API token inside the
  notebook - fewer accounts, fewer failure points, and the exact dataset that was validated
  on the laptop is the one that gets trained on.
- `training/colab_train.ipynb` rewritten around the bundle: GPU check, Drive mount, unpack,
  dataset check, 3-epoch sanity run, full run (30 epochs, batch 16 on the T4, mosaic 0.0),
  RESUME cell, val metrics, test metrics, export + download.
- Checkpoints are written to MyDrive/ppe_project/runs/ every epoch, so a Colab disconnect
  costs at most one epoch.
- Laptop smoke run (before the switch) proved the pipeline: 3 epochs on 100 images, 1.57 GB
  VRAM at batch 8, 7.5 ms inference per image, mAP50 0.008 -> 0.082. Kept as evidence that
  the code runs on the edge device too.

## D-012 Detector trained - measured results (Day 3)
- Run: Colab Tesla T4, YOLO26n fine-tuned, 30 epochs, batch 16, imgsz 640, patience 8, seed 42,
  mosaic 0.0. 560 s wall clock. Record: models/ppe4_yolo26n_info.json.
- VAL  (84 images): P 0.885, R 0.735, mAP50 0.828, mAP50-95 0.482
- TEST (59 images): P 0.896, R 0.708, mAP50 0.804, mAP50-95 0.447
- Per class on TEST: person 0.771, helmet 0.874, vest 0.782, mask 0.788 (mAP50).
- Early stopping never fired and both losses were still falling at epoch 30 - the model is
  under-trained rather than overfitted. Kept at 30 epochs deliberately (fast, reproducible);
  a 50-60 epoch run is the obvious cheap improvement if accuracy is ever the priority.
- Recall (0.71) is well below precision (0.90): the detector misses PPE more often than it
  invents it. For a safety system this is the dangerous direction, which is why the decision
  path keeps the temporal filter (D-006), the UNCERTAIN band and the human review gate.
- Full write-up with limitations: docs/evaluation/detector_results.md.
- Deployed weights: models/ppe4_yolo26n_best.pt (this is what edge/run_edge.py loads).

## D-013 Two inference entry points, on purpose (Day 3)
- `edge/detect.py`: ONE frame at a time - image, video file or webcam. No tracking, no zones,
  no temporal filter. Status per person is COMPLIANT / MISSING_HELMET / MISSING_VEST /
  MISSING_HELMET_AND_VEST, computed by `status_for()` against a configurable --required list.
  This is the demo-and-explain path, and it is what the project brief asks for.
- `edge/run_edge.py`: the full system - ByteTrack tracking, zone polygons, 15-frame temporal
  filter, UNCERTAIN state, events, audit trail. This is the path the report evaluates.
- Both share the SAME association module (head/torso regions, >= 50 % IoA, one person per PPE
  box), so the two paths cannot disagree about who is wearing what.
- Keeping them separate means the simple rule can be unit-tested with hand-written boxes
  (tests/test_detect.py, 15 tests) with no camera, no GPU and no model file.
- run_edge's default weights are now models/ppe4_yolo26n_best.pt (was the COCO yolo26n.pt).
- Demo images in samples/ are copies from the VALIDATION split; the test split is left
  untouched so it stays a clean final measurement.

## D-014 Prediction API + dashboard (Day 3)
- POST /api/v1/predict (alias POST /predict) takes ONE uploaded image and returns the same
  verdict as `python -m edge.detect`: per person the boxes, what they wear, what is missing
  and COMPLIANT / MISSING_*. It reuses edge.association and edge.detect.status_for, so the
  API and the CLI can never disagree.
- The detector sits behind `server/predict_service.PredictService`: weights load lazily on the
  first request (instant startup, and the server still starts with no model - that request
  gets a 503 with a plain-English message), and tests replace the object with a fake, so the
  whole API is testable with no GPU, no weights and no Ultralytics (tests/test_predict_api.py,
  17 tests).
- Dashboard: one static HTML file (server/static/dashboard.html) served at /dashboard - drag a
  photo in, see the annotated image, a per-person table and the counts. No React, no build
  step, no CDN: everything is local, which keeps the privacy claim simple to defend.
- Privacy: the uploaded frame is decoded in memory and discarded; nothing is written to disk
  and the response carries `image_stored: false`. The existing evidence/snapshot path
  (server/models.py, D-012 area) stays the only place images can ever be persisted.
- `/` stays a JSON index (existing test) and now links to /dashboard and /api/v1/predict.
- requirements.txt gains python-multipart (FastAPI needs it for file uploads).

## D-014 Known limitation: dark hi-vis jackets are missed by the vest class (Day 3)
- Evidence, raw model output on samples/compliant_candidate.jpg (navy hi-vis jacket + red
  hard hat), weights models/ppe4_yolo26n_best.pt, conf lowered to 0.01 for the test:
      helmet 0.899  (154, 26, 282, 166)
      person 0.821  ( 33, 28, 477, 640)
      vest   0.070  (206, 277, 376, 578)   <- correct area, almost no confidence
      vest   0.046 / 0.035 / 0.031 / 0.021 (four more fragments over the same torso)
- Diagnosis: the detector localises the garment but cannot commit to it. Not a threshold
  bug (no sane threshold sits below 0.07), not an association bug (the top vest box lies
  70 % inside the torso region built from the detected person box, well over the 0.50
  requirement), and not a class-mapping bug (checkpoint class order is person, helmet,
  vest, mask, matching data.yaml and edge/detector.py's NAME_ALIASES).
- Cause: training had 1 175 vest instances, overwhelmingly bright yellow/orange mesh vests.
  A navy jacket with reflective stripes is out of distribution. Consistent with vest being
  the weakest class on validation (mAP50 0.765, recall 0.688).
- Decision: do NOT lower the confidence threshold to make this image pass. In a safety
  system a false "vest present" marks an unprotected worker compliant - the dangerous
  direction. The image stays in samples/ as a documented hard case.
- Planned improvement, not yet done: a longer run (50 epochs, patience 10) was prepared but
  Colab's free GPU quota was exhausted; the 30-epoch model remains in use. Retraining alone
  is unlikely to lift 0.07 past 0.35 on this image - more dark-vest training data is the
  real fix.

## D-015 Backend endpoint naming: one handler, three paths (Day 4, Phase 6)
- `POST /api/v1/predict` is the canonical path; `POST /detect/image` and `POST /predict` are
  aliases on the SAME function, not copies. Versioned path for the API contract, plain path
  because the project brief names it, short path for quick curl tests.
- The handler reuses edge/detect.py (build_workers, status_for, draw) and edge/association.py,
  so the HTTP path and the command line cannot drift apart. Nothing about detection or
  compliance lives in the route.
- Rejected: a separate /detect/image implementation (duplicate logic = two things to keep in
  step), and renaming /api/v1/predict (would break the dashboard and existing tests).

## D-016 Video analysis by frame sampling, not full decode (Day 4, Phase 8)
- `POST /detect/video` analyses every Nth frame (default 5, hard ceiling 300 analysed
  frames). A 30 s clip at 30 fps is 900 frames; at ~40 ms each that is 36 s of GPU time for
  a demo that must feel live. Sampling gives the same verdicts in a fifth of the time, and
  the response reports `frames_analysed` and `every_nth` so nothing is hidden.
- Implementation detail: `capture.grab()` skips frames without decoding them and
  `capture.retrieve()` decodes only the sampled ones - much cheaper than decoding all.
- The temporal filter is deliberately NOT used here: it needs a tracker following one person
  across CONSECUTIVE frames, and sampling breaks that chain. Per-frame verdicts plus a
  summary is the honest thing to report; temporal confirmation stays in edge/run_edge.py.
- Privacy: OpenCV cannot read a video from memory, so the upload goes to the OS temp folder
  and is deleted in a `finally` block before the response is sent. A test asserts no file is
  left behind. `video_stored` is always false in the response.
- The route reuses build_workers / status_for / draw from edge/detect.py - no second copy of
  the detection or compliance logic.

## D-017 Live camera in the browser over MJPEG (Day 4, Phase 9)
- `GET /live/stream` sends `multipart/x-mixed-replace` (one JPEG after another on a single
  connection). A plain `<img src="/live/stream">` renders it, so the page needs no video
  library, no WebRTC and no new dependency - the simplest thing that works, and easy to
  explain in a viva.
- Rejected: WebRTC (a whole signalling stack for a localhost demo), and sending frames to the
  browser for client-side inference (that would put the model and the frames in the browser,
  undermining the "inference happens on this machine" claim).
- `GET /live/status` returns the newest verdict summary as JSON; the page polls it once a
  second. Images and numbers travel separately, so the JSON stays small.
- SINGLE-FRAME verdicts only. The 15-frame temporal filter needs a tracker following one
  person across consecutive frames (edge/run_edge.py). It is not reimplemented here.
- FPS is capped (default 10, max 30) so the laptop stays cool and the stream stays smooth.
- Testability: the camera is opened through `app.state.camera_factory`, which tests replace
  with a fake capture - so the stream, the status state and the release-on-disconnect
  behaviour are all tested without a webcam (tests/test_live_api.py).
- Privacy: frames are annotated in memory and sent only to 127.0.0.1; none are written to
  disk, and `capture.release()` runs in a `finally` block when the browser disconnects.

## D-018 Violation log: one JSONL file, metadata only (Day 4, Phase 12)
- `shared/violation_log.py` appends one JSON object per line to `data/violations.jsonl`.
  Used by the image API, the video API and the live camera, so there is ONE record of what
  the system flagged, whatever path found it.
- Why a file and not the SQLite database: a violation record is written once and read in
  order - exactly what an append-only text file is good at. No schema, no migration, and it
  can be opened in Notepad during a demo. The database in server/ stays for the tracked edge
  pipeline's reviewed events (D-007).
- Stored: time, source, anonymous person label, status, missing items, confidence, required
  list. NOT stored: images or frames. A test asserts no image-ish content reaches the file.
- Flood protection: the same (source, person, status) is written at most once every 30 s,
  matching the event cooldown in configs/policies.yaml. This is a LOGGING cooldown, not a
  safety decision - edge/temporal.py still decides whether a violation is real.
- A video writes ONE summary line (frames analysed, frames with a violation, rate, status
  counts), not one line per frame.

## D-019 One start command: run.py (Day 4, Phase 15)
- `python run.py` starts the backend, which already serves the dashboard, the live page, the
  APIs and the docs - so a demo needs one terminal and one command, not a memorised uvicorn
  invocation.
- It binds 127.0.0.1 by default, NOT 0.0.0.0: the demo stays on this laptop, which matches
  the privacy claim in the project title. Exposing it on the network would be a deliberate
  choice, not an accident.
- It checks the port before starting, so a second copy prints "port 8000 is already in use"
  with the fix, instead of a raw [Errno 10048].
- It warns (but still starts) when models/ppe4_yolo26n_best.pt is missing, so the pages can
  be shown even before a model exists; detection then returns a clean 503.
- Training, dataset preparation and the OpenCV window stay separate commands: they are
  development tools, not part of the running system.

## D-020 The temporal filter is justified by measurement, not assertion (Day 4, Phases 10-11)
- `scripts/compare_modes.py` runs the real temporal filter and compliance engine over two
  simulated scenarios and writes docs/evaluation/temporal_comparison.json.
- Measured: a detector that misses the helmet in 15% of frames produces
  30 false alarms in 200 frames under the single-frame baseline and
  0 under the 15-frame filter. A genuine missing vest is confirmed at frame
  15 instead of frame 1 - a delay of
  14 frames, about 0.6 s at 25 FPS.
- The detector output is SIMULATED on purpose: the experiment then isolates the decision
  logic, runs in under a second, needs no GPU, and gives the same answer every time (seeded).
  It is evidence about the rules, not about the model - the model's own numbers are in
  docs/model_card.md.
- Phases 10 and 11 were already implemented (edge/temporal.py, edge/zones.py +
  configs/zones.yaml with GENERAL / CAUTION / RESTRICTED polygons). Nothing was rebuilt;
  this phase added the evidence that the design choice was right.

## D-021 Edge benchmark: measure stages separately, name the device honestly (Day 4, Phase 14)
- `scripts/benchmark_edge.py` times detect / rules / draw / total per frame and reports mean
  and p95, so the report can say WHERE the time goes rather than quoting one FPS number.
- Warm-up frames (default 10) are discarded: the first frames pay for CUDA initialisation and
  cost tens of times steady state. The report states that they were discarded.
- Measured on this laptop, 640 px, ~1.25 people per frame:
  GPU (RTX 3050) 22.12 ms/frame mean (31.16 p95) = 45.2 FPS; CPU 47.88 ms (51.29 p95) = 20.9 FPS.
  Association + compliance take 0.03 ms - about 0.1 % of a frame - so the safety logic is
  effectively free and only the detector is worth optimising.
- Two bugs found and fixed after the first run: the CPU run printed the GPU's name (it read
  torch.cuda.get_device_name regardless of --device), and both runs wrote to the same JSON, so
  the CPU run silently overwrote the GPU numbers. Now the device label follows what was
  actually requested and each device writes edge_benchmark_<gpu|cpu>.json. Tests cover both.
- Consolidated results: docs/evaluation/system_evaluation.md.

## D-022 The laptop GPU is not used; CPU for inference, Colab T4 for GPU work (Day 4)
- Project rule: anything that needs a GPU runs on Google Colab (T4). The laptop runs the
  backend, the dashboard, the live camera, the tests and the demo - on the CPU.
- This is affordable because it was measured, not assumed: 47.88 ms per frame on the CPU
  = 20.9 FPS, against 22.12 ms = 45.2 FPS on the RTX 3050
  (docs/evaluation/system_evaluation.md section 2). A camera delivers 25-30 FPS, so the CPU
  keeps up with a live demo; the GPU was never the thing making the project possible.
- Defaults changed accordingly: server settings `detect_device = "cpu"`, and `--device cpu`
  in edge/detect.py, edge/run_edge.py and scripts/benchmark_edge.py. The GPU is still one
  flag away (`--device 0`, or PPE_DETECT_DEVICE=0 in .env) for a benchmark comparison.
- Training stays on Colab (D-011a). The GPU benchmark numbers already recorded were measured
  before this rule and are kept as the comparison row - they are a measurement of the
  hardware, not a dependency of the system.
- Two tests pin the policy: the backend settings default to cpu, and edge/detect.py's
  --device default is cpu.

## D-022a Adaptive device choice with a VRAM guardrail (Day 4, supersedes part of D-022)
- Forcing the CPU everywhere (D-022) was too blunt: short bursts - one uploaded image, one
  clip - are twice as fast on the GPU and barely touch it. Forcing the GPU is worse: a 4 GB
  laptop card shared with Windows' display work can run out of memory mid-demo.
- `shared/device.py: resolve_device()` now decides. "auto" (the default everywhere) uses the
  GPU only when CUDA is present AND at least 1500 MB of VRAM is free RIGHT NOW; otherwise it
  falls back to the CPU. A nano model at batch 1 needs about 1.5 GB, hence the threshold.
- The fallback is a slowdown, never a failure: 20.9 FPS on the CPU vs 45.2 on the GPU
  (docs/evaluation/system_evaluation.md), and a camera only delivers 25-30 FPS.
- An explicit `--device cpu` or `--device 0` (or PPE_DETECT_DEVICE) always wins - the
  guardrail is a default, not a policy that cannot be overridden.
- Every caller prints WHICH device was chosen and WHY, e.g.
  "device: cpu (NVIDIA GeForce RTX 3050 A Laptop GPU has only 400 MB free of 4096 MB
  (need 1500 MB) - using the CPU instead)". Silent fallbacks are how people end up
  mis-reporting their own benchmark numbers.
- A driver error during the check is caught and treated as "no GPU": a broken driver must
  slow the demo down, not crash it.
- Training stays on Colab (D-011a). Unchanged.
- tests/test_device.py (8 tests) covers every branch with a fake torch module, so the
  guardrail is tested on machines with and without a GPU.

## D-023 The web pages live in frontend/, and stay plain HTML (Day 4)
- `server/static/*.html` moved to `frontend/dashboard.html` and `frontend/live.html`. The
  interface is a deliverable of its own, not an asset buried inside the server package, and
  the project structure now says so at a glance.
- One place knows the path: `FRONTEND_DIR` in server/config.py, imported by server/main.py
  and server/routes/live.py. (A first attempt imported it from server.main inside the route,
  which is a circular import waiting to happen - config.py is the neutral home.)
- Still plain HTML + CSS + JS, deliberately: fetch() calls the APIs, a plain <img> renders the
  MJPEG stream natively, and the annotated picture arrives as a data: URI. React would add a
  toolchain, node_modules and a build step and show a judge nothing extra.
- No static mount: two files are served with FileResponse. Editing a page means saving it and
  hard-refreshing the browser - no server restart, because the file is read per request.
- frontend/README.md records all of this. A test asserts both pages exist there and are served.

## D-024 Event snapshot masking: blur a region, never detect faces (Day 4)
- Snapshots stay OFF by default (`store_snapshots=False`): the system's normal state is
  metadata only. When a supervisor needs the evidence, PPE_STORE_SNAPSHOTS=true turns them
  on and masking becomes mandatory - there is no path that writes an unmasked frame.
- We do NOT add a face detector. A second model trained to find faces is itself a privacy
  risk, and it fails exactly when the worker is turned away - which is when the risk is
  highest. Instead `shared/privacy.py` blurs a geometric FACE BAND derived from the person
  box the detector already produced, so it cannot "miss" a face.
- The band sits BELOW the helmet line on purpose: the helmet stays visible, so the stored
  evidence still shows what the violation was about. The first version used a fixed
  8 %-34 % of the person box height; that turned out to be wrong and was corrected in
  D-025. `scripts/privacy_utility.py` reports how many detections of each class survive
  masking, which is how the error was found.
- Pixelation, not a light Gaussian blur: a mild blur can be partly inverted, a heavy
  pixelation cannot.
- FAIL CLOSED: if masking cannot be done, no file is written and the status is
  EVIDENCE_WITHHELD. A missing snapshot is an inconvenience; an unmasked one is a breach.
  Tests assert that a broken frame and an unmaskable box both leave the folder empty.
- The violation log stores a PATH and a privacy_status, never image data.
- tests/test_privacy.py includes one test that reads the saved JPEG back and checks the face
  area really changed, and one that checks the helmet strip did NOT.
- Hardware note: this project is software-only - the laptop webcam stands in for a site
  camera, which the brief allows ("hardware can be emulated").

## D-025 The face band was measured, found wrong, and placed from the data (Day 4)
This is the decision I would put in front of an examiner first, because the project
caught its own mistake.

**What went wrong.** D-024 asserted that a band at 8 %-34 % of the person-box height sits
below the helmet. `scripts/privacy_utility.py` measured it on 40 validation images:

| class | detections before | after | kept | conf before | conf after |
|---|---|---|---|---|---|
| person | 84 | 71 | 84.5 % | 0.672 | 0.650 |
| helmet | 45 | 15 | **33.3 %** | 0.818 | 0.567 |
| vest | 17 | 10 | 58.8 % | 0.691 | 0.656 |
| mask | 7 | 7 | 100.0 % | 0.796 | 0.777 |

Two thirds of the helmet evidence was destroyed by the privacy step - the opposite of
what I had written down, and exactly the claim the script exists to test. `mask`, which I
predicted would suffer, was untouched.

**Why.** `scripts/face_band_geometry.py` measures where PPE boxes really sit inside person
boxes, using the dataset's own labels - no model, no GPU, pure arithmetic. The answer is
that there is no single fraction to use, because person boxes come in two shapes:

| person box shape (height/width) | helmet lower edge, median fraction of box height |
|---|---|
| full body, >= 1.8 | 0.13 (val) / 0.17 (train) |
| upper body, 1.2-1.8 | 0.38 / 0.33 |
| head crop, < 1.2 | 0.72 / 0.51 |

In a full-body box the helmet ends around 0.15h, so a band starting at 0.08h lands *on the
helmet*. In a head-and-shoulders crop the same band lands on the forehead and misses the
face entirely - it was both destroying evidence and under-protecting privacy.

**The fix, in two parts.**
1. The band is placed by interpolating on the box's aspect ratio between the measured
   points (1.0, 0.60), (1.5, 0.35), (2.5, 0.15), and extends to 1.9x that depth for the
   rest of the face. A distant full-body worker gets 0.15h-0.29h; a close-up head crop
   gets 0.60h-1.00h.
2. `blur_faces(..., keep_boxes=...)` restores the detected helmet and vest boxes after
   pixelation, so a misplaced band can no longer erase the evidence *at all*. This makes
   the failure structurally impossible instead of merely unlikely. A face mask is
   deliberately NOT restored: it sits on the face, and identity wins over that one class.
   The compliance verdict is in the log either way.

**The result** (40 val images, detections paired by IoU >= 0.5):

| class | preserved, band + guard | preserved, band alone |
|---|---|---|
| helmet | 93.3 % | 64.4 % (was 33.3 %) |
| vest | 94.1 % | 35.3 % |
| mask | 85.7 % | 85.7 % |
| person | 52.4 % | 40.5 % |

**A second measurement bug, found the same way.** The first version of the evaluation
counted detections before and after. That produced `vest 17 -> 22 = 129.4 % retained`,
which is impossible and exposed the metric: pixelation, and the sharp edges where the
guard restores a patch, make the detector invent boxes. The script now pairs each original
with the best overlapping detection of the same class and reports `preserved`, `lost` and
`spurious` separately. Spurious boxes never change a verdict - compliance is decided on
the original frame, before anything is stored.

**Honesty notes.**
- Helmet and vest preservation with the guard on is high *partly by construction*, since
  the guard restores those pixels. That is why `--no-keep` exists and why both columns are
  published.
- `mask` was predicted to be the casualty and was not (85.7 % either way). Recorded as
  measured rather than adjusted to fit the earlier claim.
- Person boxes preserve worst (52.4 %). IoU >= 0.5 is strict for tall thin boxes, and in
  crowded photos the detector's "one worker or two" decision flips once part of the image
  is pixelated. It does not affect the system: the person count in the log comes from the
  original frame, not the stored picture.
- `preserved` measures the *detector*. A supervisor can usually read a pixelated image the
  detector cannot, so it is a lower bound on human readability.

**What this cost.** One evaluation script, one geometry script, and a decision entry
admitting the first design was wrong. What it bought is a privacy step that has been
tested against its own purpose rather than asserted.

## D-026 The copilot proposes, a human decides, the chain records it (Day 5)
The brief asks for a *human-governed* copilot, not an autonomous one. That is a design
constraint, so it is enforced in code rather than promised in a paragraph.

**Three properties, each with a test.**
1. *Allow-listed.* `shared/copilot.ACTIONS` is five actions - record, flag for review,
   notify the supervisor, escalate a restricted-zone breach, ask for a second look. Every
   path that stores or executes anything goes through `check_action()`, which raises on
   anything else, including an action id arriving in an API request. A property test
   sweeps every status x zone x repeat-count combination and asserts no input can produce
   an action outside the list. The list is also published at `/api/v1/copilot/catalog`
   and rendered in the dashboard: a user can read the copilot's entire vocabulary.
2. *Human-governed.* A suggestion is inert. It carries out nothing until
   `POST /suggestions/{id}/decide` arrives with a named actor, and it can be decided only
   once (a second click gets 409, naming who decided first). There is no auto-approve
   path and no confidence threshold that bypasses review.
3. *Traceable.* Both the proposal and the decision append to the existing hash-chained
   audit log (D-016), with `actor="copilot"` for the proposal and the person's name for
   the decision. A test approves a suggestion and then asserts the chain still verifies.

**No language model, deliberately.** Mapping "missing helmet in a restricted zone" to
"escalate" is policy, not judgement. A rule table is auditable, deterministic, runs in
microseconds on a laptop, and cannot be argued into proposing something off the list. An
LLM would cost all four properties and buy nicer sentences. The rules are five lines in
`suggest()`, labelled R1-R5, and every suggestion stores which rule fired.

**One suggestion at a time.** Each branch returns at most one proposal. A queue that
offers three options per worker is a queue nobody reads.

**What "executing" actually does.** The two alerting actions append a line to
`data/outbox.jsonl`; the others set a flag. Nothing is sent anywhere. That is the honest
implementation for a project whose whole claim is that data stays on the machine - and in
a real deployment the outbox is the seam where a site siren or SMS gateway would attach.

**Repeat counts are read, not trusted.** Rule R4 ("seen repeatedly") counts prior
suggestions in the database rather than believing a `repeat_count` sent by the caller, so
a request cannot talk its way up to a supervisor alert.

**Cost:** one module, one table, one route file, one dashboard panel, 24 tests. No new
dependency, no new service, no model.

## D-027 Sign-in and three roles: the decision record now names a person (Day 5)
D-026 made every action wait for a human. It recorded the name that human typed into a box,
which is a label, not an identity - anyone could approve an escalation as "supervisor_ana".
For a system whose whole value is a traceable human decision, that was the weakest link
(threat T1). This closes it.

**Three ordered roles**, because the work has three shapes: `viewer` (see the site),
`supervisor` (decide on proposals), `admin` (also configuration and maintenance).
`require_role("supervisor")` admits an admin too. A general permission system would be more
flexible and, at three roles, more code than it earns.

**The actor is taken from the session, never from the request body.** The `actor` field was
deleted from the decision payload: sending one now changes nothing, and a test asserts that.
The audit row records the authenticated name *and* the role it acted with.

**Mechanics, all standard library.**
- PBKDF2-HMAC-SHA256, per-user salt, 200 000 iterations. Never plaintext, never a bare
  SHA-256 that a GPU chews through. Accounts live in `configs/users.yaml` (gitignored),
  created by `python -m scripts.make_user`, which hashes the password and discards it.
- The session is a signed token in an HttpOnly, SameSite=strict cookie:
  `base64(payload).HMAC-SHA256`. No server-side session store; a tampered payload fails the
  signature. Tests forge one and assert the refusal.
- The signing secret is generated fresh at start-up unless configured, so there is no secret
  to commit. The cost is signing in again after a restart - the right trade here.
- Wrong username and wrong password return the identical message, and a missing user still
  pays the hashing cost, so neither the wording nor the timing reveals which accounts exist.

**No JWT library, no OAuth, no password reset.** Each is more moving parts than a
single-site console needs, and the honest scope of this project is a prototype.

**What is still open, and stays written down rather than hidden:** no rate limiting or
account lockout, and no TLS - the server binds 127.0.0.1, so the cookie never crosses a
network. A real deployment puts this behind HTTPS and an identity provider; `server/auth.py`
is the seam where that attaches. Read access is also left open on the local console: the
gate is on deciding, which is where accountability actually matters.

