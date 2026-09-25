# Model card - ppe4_yolo26n

A short, honest description of the PPE detector used by this project: what it is, what it
may be used for, how well it works, and where it fails. Every number here was measured on
this project's own data; nothing is copied from a paper or a leaderboard.

---

## 1. Model details

| | |
|---|---|
| Name | `ppe4_yolo26n` (file: `models/ppe4_yolo26n_best.pt`, 5.4 MB) |
| Architecture | Ultralytics **YOLO26n** (nano), 260 layers, 2.51 M parameters, 5.9 GFLOPs; 2.38 M / 5.3 GFLOPs after fusing for inference |
| Type | Single-stage object detector (boxes + class + confidence, one forward pass per image) |
| Starting point | COCO-pretrained `yolo26n.pt` - **transfer learning**, not trained from scratch. 612 of 708 weight tensors transferred; the COCO `person` row of the classification head was reused by name, the other three classes are new |
| Classes | `0 person`, `1 helmet`, `2 vest`, `3 mask` (order fixed in `datasets/ppe4/data.yaml`, in the checkpoint, and locked by a unit test) |
| Input | RGB image resized to 640 x 640 |
| Output | Bounding boxes in pixels + class + confidence. **It does not output compliance** - that is decided afterwards by a deterministic rule engine |
| Training | 30 epochs, batch 16, imgsz 640, patience 8 (never triggered), seed 42, AMP on, `mosaic=0.0`. 560 s on a Google Colab Tesla T4 |
| Libraries | Ultralytics 8.4.160, PyTorch 2.11.0+cu130 |
| Reproducibility | `models/ppe4_yolo26n_info.json` (arguments, versions, GPU, data.yaml hash), `docs/evaluation/train_args.yaml`, `datasets/ppe4/subset_manifest.json` |
| Licence note | Ultralytics is AGPL-3.0, so code built on it must be released under a compatible licence |

## 2. Intended use

**Intended:** an academic demonstration of edge PPE-compliance monitoring on a construction
site - detecting people and their helmet / vest / mask so a deterministic rule engine can
flag *possible* non-compliance for a **human** to review.

**Explicitly not intended for:**

* disciplinary action, pay decisions or any automated consequence for a worker;
* identifying *which* person is in frame (the system has no face recognition and assigns
  anonymous tracker IDs that reset every session);
* safety-critical automation where a missed violation could go unnoticed - recall is 0.71,
  so roughly three in ten PPE items are missed;
* environments unlike the training data (indoor offices, warehouses at night, hospitals,
  road works at distance);
* detecting PPE the model was never trained on: gloves, goggles, boots, harnesses,
  respirators.

## 3. Training data

| | |
|---|---|
| Source | Roboflow Universe, *Construction Site Safety* by Roboflow Universe Projects, licence **CC BY 4.0** |
| Size used | 1 000 train / 84 validation / 59 test images (1 143 total), built by `training/prepare_ppe4.py` with seed 42 |
| Original classes | 10 (Person, Hardhat, Mask, Safety Vest, NO-Hardhat, NO-Mask, NO-Safety Vest, Safety Cone, machinery, vehicle) - remapped to our 4, the rest dropped |
| Box counts (train) | person 3 641, helmet 1 267, vest 1 175, mask 631 |
| Content | Construction-site photography: building sites, scaffolding, timber framing, roadworks, machinery, plus some stock and CCTV-style frames |

**Two properties of this data that matter:**

1. The source's **training images are pre-augmented mosaics** - four photos tiled into one
   with cutout squares - while validation and test are clean single photos. Ultralytics'
   own mosaic augmentation is therefore disabled (`mosaic=0.0`). Train and evaluation data
   are not drawn from identical distributions.
2. **`vest` instances are overwhelmingly bright yellow/orange mesh vests.** This directly
   causes the failure in section 6.

## 4. Evaluation

Measured with `training/validate.py`. The test split was used **once**, after all training
decisions were final; it was never trained on or tuned against.

| split | images | precision | recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| validation | 84 | 0.885 | 0.735 | 0.828 | 0.482 |
| **test** | **59** | **0.896** | **0.708** | **0.804** | **0.447** |

Per class, test split:

| class | instances | precision | recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| person | 175 | 0.812 | 0.703 | 0.771 | 0.418 |
| helmet | 111 | 0.955 | 0.765 | 0.874 | 0.523 |
| vest | 60 | 0.817 | 0.683 | 0.782 | 0.416 |
| mask | 29 | 1.000 | 0.679 | 0.788 | 0.430 |

Raw files: `docs/evaluation/detector_val_metrics.json`,
`docs/evaluation/detector_test_metrics.json`, `results.csv`, `results.png`,
`confusion_matrix.png`.

**Speed:** 7.5 ms inference per image on an RTX 3050 laptop GPU (640 px, FP16); the full
edge loop including capture, tracking, rules and drawing ran at 15.6 FPS on the same laptop.

## 5. How to read these numbers honestly

* **Precision (0.90) is much higher than recall (0.71).** The detector rarely invents PPE
  but regularly misses it. For a safety system that is the *wrong* direction - a missed
  helmet means a real violation is never raised - which is why the pipeline adds a 15-frame
  temporal filter, an `UNCERTAIN` state and a human review gate instead of acting on one
  frame.
* **`mask` precision 1.000 means "no false masks in 29 instances"**, not that mask detection
  is solved. One missed mask moves its recall by 3 points. Never quote it without the count.
* **mAP50-95 (0.447) sits far below mAP50 (0.804):** boxes are found but not tightly placed.
  That is normal for a nano model and mostly harmless here, because compliance depends on
  *which* PPE box overlaps *which* person, not on pixel-perfect edges.
* **59 test images is small.** All per-class figures carry wide error bars.

## 6. Known failure modes

| Failure | Evidence | Consequence |
|---|---|---|
| **Dark / navy hi-vis jackets are not recognised as `vest`** | On `samples/compliant_candidate.jpg` the jacket peaks at confidence **0.070** (threshold 0.35), in five fragmented boxes over the correct torso area | A properly dressed worker is reported `MISSING_VEST` - a false alarm, which wastes reviewer time |
| Missed PPE in general (recall 0.71) | Test split | A real violation is not raised - the dangerous direction |
| Heavy overlap between people | By design of the association rule | A helmet may be credited to the wrong person |
| Small / distant people | Nano model at 640 px | Workers far from the camera are missed entirely |
| Domain shift | Trained on construction photography | Accuracy on an indoor webcam demo is materially worse than the table above; do not quote these numbers for that setting |
| No absence classes | `NO-Hardhat` etc. were dropped deliberately | A *missed* vest and an *absent* vest look identical to the rules; the confidence band and temporal filter exist to soften this |

The dark-jacket case is kept in `samples/` on purpose. It was **not** fixed by lowering the
confidence threshold: at 0.07 the system would accept almost any activation as a vest, and a
false "vest present" marks an unprotected worker compliant - the failure that actually hurts
someone. More dark-vest training data is the real fix. See `docs/decisions.md` D-014.

### Resource and robustness profile (measured, RTX 3050 Laptop)

| | value |
|---|---|
| latency p50 / p95 / p99 | 15.3 / 19.1 / 22.7 ms |
| sustained throughput | 63.9 FPS |
| peak memory | 1.23 GB (580 MB is the model) |
| helmet detection retained, all lighting tested | 96.8 % |
| person detection retained, dark (gamma 2.2) | 91.5 % |
| **person detection retained, sensor noise sigma 15** | **39.0 %** |

The last row is the honest one: the model tolerates brightness and darkness, and is defeated
by noise. It should not be pointed at night footage from a high-gain camera without being
re-evaluated on that footage first.

## 7. Privacy and ethics

* **All inference is local.** Frames are read, analysed and discarded on the machine that
  captured them. There is no cloud vision API anywhere in the code path, and the system keeps
  working with the network unplugged.
* **No identity.** No face recognition, no matching, no name database. People are tracked
  with ByteTrack, which uses motion and box overlap only, and IDs reset each session.
* **Metadata, not pictures.** By default only structured results are stored (timestamp,
  camera, anonymous worker ID, status, missing PPE, confidence). If a snapshot must be kept
  as evidence, faces are blurred first.
* **Human oversight is enforced, not requested.** The model says what is visible; the rule
  engine decides whether it is a problem; the copilot proposes one action from a five-item
  allow-list; and nothing happens until a named person approves it in the dashboard. There is
  no auto-approve path and no confidence threshold that bypasses review
  (`shared/copilot.py`, `docs/decisions.md` D-026).
* **Every proposal and decision is recorded** in a hash-chained audit log, one as `copilot`,
  one as the person, so a decision can be reconstructed and silent edits are detectable.
* **Stored evidence is masked and measured.** When snapshots are switched on, the face band
  is pixelated before writing and a masking failure writes nothing. The cost of that masking
  to the evidence was measured, found unacceptable in the first design, and fixed
  (D-025) - 93 % of helmet detections now survive masking, against 33 % before.
* **Foreseeable misuse:** re-purposing the tracker as a productivity or attendance monitor,
  or pairing it with face recognition. Both are out of scope and would require different
  consent, retention and governance. Six named misuse cases, the trust boundaries and the
  residual risks are in `docs/threat_model.md`.
* **Masking is not anonymisation.** Build, clothing, gait and context still identify people,
  and the band is placed by geometry rather than by finding a face, so an unusual pose can
  leave part of a face visible. The project does not claim "100 % private".
* **No authentication.** The deciding actor's name is typed, not proven. Stated as an open
  risk (threat T1) rather than papered over with a fake login.

## 8. Maintenance

* Retrain when the site, camera or PPE type changes - the model is specific to what it saw.
* Highest-value next step: **more dark and non-standard hi-vis examples** for the `vest`
  class, then a longer run (50-60 epochs; the 30-epoch curves were still improving and early
  stopping never fired).
* Re-run `python -m scripts.check_dataset` and `python -m training.validate --split test`
  after any dataset change, and update this card with the new measured numbers.

*Card written 24 Sep 2026 for model `ppe4_yolo26n`, trained 23 Sep 2026.*
