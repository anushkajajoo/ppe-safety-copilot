# System evaluation

Every number here was measured on this project, on this laptop or on the Colab T4 that
trained the model. Nothing is copied from a paper. Each row says which command produced it,
so all of it can be re-run.

---

## 1. Detector accuracy

Model `ppe4_yolo26n` (YOLO26n, 2.51 M parameters), trained 30 epochs on 1 000 images.
Command: `python -m training.validate --weights models\ppe4_yolo26n_best.pt --split test`

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

Sources: `detector_val_metrics.json`, `detector_test_metrics.json`, `results.csv`,
`confusion_matrix.png`. Interpretation and limits: `detector_results.md`,
`../model_card.md`.

## 2. Edge performance (this laptop)

Command: `python -m scripts.benchmark_edge` (and `--device cpu` for the second row).
120 frames measured, 10 warm-up frames discarded, 640 px, conf 0.35, ~1.25 people per frame.

| device | detect ms (mean / p95) | rules ms | draw ms | total ms (mean / p95) | FPS mean | FPS p95 |
|---|---|---|---|---|---|---|
| **RTX 3050 laptop GPU** | 21.25 / 30.39 | 0.03 | 0.83 | 22.12 / 31.16 | **45.2** | 32.1 |
| CPU (i7-12650H) | 47.07 / 50.43 | 0.03 | 0.78 | 47.88 / 51.29 | **20.9** | 19.5 |

Three things this table settles:

* **The system is real-time on the laptop GPU** - 45 FPS mean, and still 32 FPS at p95, well
  above the 25 FPS a camera delivers.
* **The GPU is roughly 2.2x the CPU here**, and the CPU alone still manages ~21 FPS. The demo
  survives on a machine with no GPU, which matters for an "edge" claim.
* **The rules cost nothing.** Association plus the compliance decision takes 0.03 ms - about
  0.1 % of the frame. All the time is in the neural network. So making the safety logic more
  careful (temporal windows, zones, confidence bands) is effectively free; only the detector
  is worth optimising.

A separate end-to-end run over `samples/demo_clip.mp4` through `python -m edge.detect`
processed 120 frames in 4.8 s = **25.2 FPS** including video decoding and file I/O.

## 2b. Resources: memory, latency percentiles, sustained rate

Command: `python -m scripts.measure_resources --frames 100`
(RTX 3050 Laptop GPU, 3244 MB free, 640 px, one 640x480 frame reused)

| | value |
|---|---|
| latency p50 | 15.3 ms |
| latency p90 | 18.3 ms |
| latency p95 | **19.1 ms** |
| latency p99 | 22.7 ms |
| worst frame | 23.6 ms |
| sustained rate | **63.9 FPS** |
| memory before the model loaded | 37.2 MB |
| memory after the model loaded | 617.5 MB (**+580 MB for the model**) |
| peak during the run | **1228.9 MB** |

The distribution is the point. p99 is 22.7 ms against a p50 of 15.3 - a spread of 7 ms, with
no stalls. An average would have said "16 ms" and hidden whether that was true every frame or
on average across one freeze. It was true every frame.

580 MB of the footprint is the model and its CUDA context, not the application: the Python
process before loading is 37 MB. Peak 1.2 GB leaves room for the browser and the dashboard on
a 4 GB card, which is what the `auto` device guardrail is protecting (D-022a).

## 2c. Robustness to lighting

Command: `python -m scripts.robustness_lighting --images 30`
Each image is detected untouched, then re-detected under six transforms; detections are
paired with the control by overlap, so the number is "how much of what we could see can we
still see".

| condition | person (n=59) | helmet (n=31) | vest (n=3) | mask (n=0) |
|---|---|---|---|---|
| bright, gamma 0.4 | 88.1 % | 96.8 % | 66.7 % | not measured |
| slightly bright, gamma 0.6 | 94.9 % | 96.8 % | 66.7 % | not measured |
| **as measured (control)** | **100 %** | **100 %** | **100 %** | not measured |
| slightly dark, gamma 1.6 | 94.9 % | 96.8 % | 33.3 % | not measured |
| dark, gamma 2.2 | 91.5 % | 96.8 % | 33.3 % | not measured |
| flat light, contrast 0.5 | 89.8 % | 96.8 % | 66.7 % | not measured |
| **sensor noise, sigma 15** | **39.0 %** | **45.2 %** | 0 % | not measured |

The control reads 100 %, so the measurement is sound.

**What it says.** Helmets are the robust class: 96.8 % survive every brightness, darkness and
flat-light condition tested. People hold up too, 88-95 %. The system does not fall over at
dusk or under a floodlight, which was the open question.

**What it does not say.** `mask` had **zero** detections in the control run, so those cells
are "not measured", not "0 %" - an earlier version of the script printed 0 % there, which
would have been a false claim in this document. `vest` had **three**: those percentages are
anecdote, not measurement, and are reported only because hiding a thin sample is worse than
labelling it.

**The real finding: sensor noise breaks it.** At sigma 15 - a cheap sensor at high gain,
which is what night footage looks like - person detection falls to 39 % and helmet to 45 %.
Brightness the model tolerates; *noise* it does not.

This is **not** an acceptance criterion, deliberately. The system is specified for daylight
site monitoring on a laptop camera, and night operation is out of scope (`problem_brief.md`
§4). Adding a gate the project would fail, for a condition it does not claim to handle, would
be theatre. Recording it as a limitation, with the number, is the honest form - and it names
the first thing to test if anyone ever points this at a night shift.

## 3. Decision logic: temporal filter vs single-frame baseline

Command: `python -m scripts.compare_modes` (seeded, no GPU needed, ~1 second).

| scenario | mode | VIOLATION | UNCERTAIN | COMPLIANT | first flag |
|---|---|---|---|---|---|
| A - detector blinks (worker IS compliant; helmet missed in 15 % of frames) | baseline | **30 false alarms** | 0 | 170 | frame 2 |
| | proposed | **0** | 22 | 178 | never |
| B - real violation (no vest at all) | baseline | 200 correct | 0 | 0 | frame 1 |
| | proposed | 186 correct | 14 | 0 | frame 15 |

**30 of 30 false alarms removed, for 14 frames (0.6 s at 25 FPS) of delay before a genuine
violation is confirmed.** While the window fills, the answer is UNCERTAIN - never COMPLIANT.

Raw numbers: `temporal_comparison.json`.

## 4. Dataset

Command: `python -m scripts.check_dataset` -> **DATASET OK (1143 images)**

| split | images | person | helmet | vest | mask |
|---|---|---|---|---|---|
| train | 1 000 | 3 641 | 1 267 | 1 175 | 631 |
| val | 84 | 188 | 94 | 55 | 27 |
| test | 59 | 175 | 111 | 60 | 29 |

Seeded subset (seed 42) of the Roboflow Universe *Construction Site Safety* dataset,
CC BY 4.0. Checks run: missing labels, missing images, corrupt images, malformed label
lines, boxes outside 0-1, invalid class ids, class distribution. Details:
`dataset_summary.json`, `../decisions.md` D-010 and D-010a.

## 5. Automated tests

Command: `python -m pytest` -> **233 tests**, about 14 seconds

| Area the brief asks about | Covered by | Tests |
|---|---|---|
| Person / helmet / vest detection | `test_detect.py`, `test_edge_logic.py` | 39 |
| PPE-person association | `test_detect.py` (head/torso regions, one-person-per-box), `test_edge_logic.py` | in the 39 |
| Compliance logic | `test_edge_logic.py`, `test_detect.py`, `test_compare_modes.py` | 46 |
| Image API | `test_predict_api.py` | 25 |
| Video API | `test_video_api.py` | 15 |
| Backend / health / schemas | `test_health.py`, `test_schemas.py`, `test_events_api.py` | 18 |
| Dashboard pages | `test_predict_api.py`, `test_live_api.py` | included above |
| Webcam / live streaming | `test_live_api.py` (fake camera) | 10 |
| Temporal filtering | `test_edge_logic.py`, `test_compare_modes.py` | 29 |
| Safety zones | `test_edge_logic.py` | included above |
| Violation logging | `test_violation_log.py` + API tests | 9 |
| Dataset preparation | `test_prepare_ppe4.py`, `test_check_dataset.py` | 19 |
| Training / Colab config | `test_train_cli.py`, `test_colab_bundle.py` | 10 |
| Benchmark helpers | `test_benchmark_edge.py` | 8 |
| Start command | `test_run_py.py` | 6 |
| Event snapshot masking | `test_privacy.py` (incl. reading the saved JPEG back) | 21 |
| Copilot: rules, allow-list, approval gate, audit | `test_copilot.py` | 24 |

No test needs a GPU, a camera or the model file: the detector is replaced by a fake
wherever it would otherwise be required. The whole suite runs in about 15 seconds.

## 6. Privacy checks (technical, not just claimed)

| Claim | How it is enforced | Test |
|---|---|---|
| Uploaded images are not stored | decoded in memory, response says `image_stored: false` | `test_the_image_is_never_stored` |
| Uploaded videos are not kept | temp file deleted in a `finally` block | `test_the_uploaded_clip_is_deleted_from_the_temp_folder` |
| The violation log holds no images | metadata-only writer | `test_no_image_or_frame_is_ever_written` |
| The camera is released | `capture.release()` when the browser disconnects | `test_status_reflects_the_last_frame_and_camera_is_released` |
| The server stays on this machine | `run.py` binds 127.0.0.1, not 0.0.0.0 | `test_defaults_are_local_only` |
| The copilot cannot act on its own | nothing executes without a named actor's decision | `test_a_suggestion_starts_pending_and_does_nothing`, `test_rejecting_carries_nothing_out` |
| The copilot cannot invent an action | `check_action()` gates every path | `test_the_rules_only_ever_produce_allow_listed_actions`, `test_an_unknown_action_is_refused` |
| Decisions are attributable and final | named actor required; second decision refused | `test_a_decision_needs_a_named_actor`, `test_a_suggestion_can_only_be_decided_once` |
| The decision trail survives tampering | hash chain still verifies after a decision | `test_every_proposal_and_decision_reaches_the_audit_chain` |

### 6a. What masking COSTS the evidence (measured)

Storing a masked snapshot is only worth anything if the picture still shows the violation.
`scripts/privacy_utility.py` measures that directly: detect, mask the face bands, detect
again, compare. Command:

```
python -m scripts.privacy_utility --images 40            # with the keep guard
python -m scripts.privacy_utility --images 40 --no-keep  # the band on its own
```

**First run, fixed 8 %-34 % band, no keep guard** - the run that found the bug:

| class | before | after | kept | conf before | conf after |
|---|---|---|---|---|---|
| person | 84 | 71 | 84.5 % | 0.672 | 0.650 |
| helmet | 45 | 15 | **33.3 %** | 0.818 | 0.567 |
| vest | 17 | 10 | 58.8 % | 0.691 | 0.656 |
| mask | 7 | 7 | 100.0 % | 0.796 | 0.777 |

Two thirds of the helmet evidence was gone. `scripts/face_band_geometry.py` then measured
from the dataset labels why: the head sits at a different fraction of the person box
depending on the box's shape (0.15h in a full-body box, 0.60h in a head crop), so no fixed
band can work. See `../decisions.md` D-025 for the full analysis and the fix.

**After the fix**, measured by pairing each original detection with the masked-image
detection of the same class that overlaps it (IoU >= 0.5), 40 validation images:

_Band placed by aspect ratio + helmet/vest boxes restored (the shipped configuration):_

| class | originals | preserved | lost | spurious | conf before | conf after |
|---|---|---|---|---|---|---|
| person | 84 | 44 (52.4 %) | 40 | 46 | 0.672 | 0.652 |
| helmet | 45 | **42 (93.3 %)** | 3 | 2 | 0.818 | 0.809 |
| vest | 17 | **16 (94.1 %)** | 1 | 6 | 0.691 | 0.609 |
| mask | 7 | 6 (85.7 %) | 1 | 1 | 0.796 | 0.773 |

13.07 % of the frame is pixelated on average, across 84 masked face regions.

_Band alone, `--no-keep` - what the geometry does without the restore guard:_

| class | originals | preserved | lost | spurious | conf before | conf after |
|---|---|---|---|---|---|---|
| person | 84 | 34 (40.5 %) | 50 | 46 | 0.672 | 0.659 |
| helmet | 45 | 29 (64.4 %) | 16 | 1 | 0.818 | 0.790 |
| vest | 17 | 6 (35.3 %) | 11 | 1 | 0.691 | 0.750 |
| mask | 7 | 6 (85.7 %) | 1 | 1 | 0.796 | 0.772 |

What the two tables say, read together:

- **The band alone is better than the original but not good enough.** Helmet preservation
  rose from 33.3 % (old fixed band, count-based) to 64.4 %, and helmet confidence from
  0.567 to 0.790. Vest still suffers at 35.3 %, because in crop-shaped person boxes the
  band reaches down over the chest. That residual failure is exactly what the restore
  guard exists for.
- **With the guard, the PPE evidence survives**: helmet 93.3 %, vest 94.1 %, mask 85.7 %.
  This is partly by construction - the guard restores those pixels - which is why both
  tables are published rather than only the flattering one.
- **`mask` was not the casualty I predicted.** 85.7 % preserved either way, confidence
  0.796 -> 0.773. The band evidently overlaps worn face masks less than assumed. Recorded
  as measured, not adjusted to fit the earlier claim.
- **Person boxes are the least stable thing in the picture** - 52.4 % preserved, 46
  spurious. Two reasons: IoU >= 0.5 is strict for tall thin boxes, so a small wobble in the
  box edge counts as a loss; and in crowded site photos the detector's choice between "two
  workers" and "one worker" flips easily once part of the image is pixelated. It matters
  little here: the snapshot exists to show the PPE state, and the person count in the log
  comes from the original frame, not from the stored picture.

Two cautions that belong with any quotation of these figures:

- **Counting detections is not a measure.** An earlier version of this script reported
  129.4 % "retention" for vest, because masking can make the detector invent boxes. Those
  are now counted separately as `spurious`, and `preserved` is matched box-to-box. A
  spurious box never changes a verdict: compliance is decided on the original frame,
  before anything is stored.
- **This measures the detector, not a person.** Pixelation keeps coarse shape and colour,
  so a supervisor can usually read an image the detector can no longer parse. `preserved`
  is a lower bound on human readability, not a measure of it.

## 6b. Acceptance criteria (go / no-go)

"The system performs well" is not a criterion. `configs/acceptance.yaml` names each claim,
the measurement behind it and the threshold below which the claim stops being true;
`python -m scripts.check_acceptance` reads the evaluation reports and prints PASS, FAIL or
SKIPPED for each, then a single verdict. **A missing report is reported as SKIPPED, never as
a pass** - the gate cannot be satisfied by not measuring something.

| Criterion | Required | Measured | Status |
|---|---|---|---|
| Detector mAP50 (test split) | >= 0.70 | 0.804 | PASS |
| Detector recall | >= 0.60 | 0.708 | PASS |
| Helmet recall | >= 0.70 | 0.765 | PASS |
| False alarms after the temporal filter | == 0 | 0 | PASS |
| Helmet evidence preserved through masking | >= 85 % | 93.3 % | PASS |
| Face regions actually masked | > 0 | 84 | PASS |
| Frame latency p95 | <= 200 ms | 19.1 ms | PASS |
| Sustained frame rate | >= 8 FPS | 63.9 FPS | PASS |
| Peak memory | <= 2500 MB | 1228.9 MB | PASS |
| Lighting control condition | == 100 % | 100 % | PASS |
| People found in dim light (gamma 1.6) | >= 60 % | 94.9 % | PASS |

**Current verdict: GO** — all eleven criteria pass.

Two of these sit uncomfortably close to the measured value, and that is deliberate: the
thresholds were written from what the system is *for*, not fitted to the numbers it happens
to produce.

## 7. Honest limitations

1. **Vest detection fails on dark hi-vis jackets** - confidence 0.070 on
   `samples/compliant_candidate.jpg`, below the 0.35 threshold. Not fixed by lowering the
   threshold, because a false "vest present" marks an unprotected worker compliant. See
   `../decisions.md` D-014.
2. **Recall 0.708 < precision 0.896** - the detector misses PPE more often than it invents
   it, which is the dangerous direction for safety. Mitigated by the temporal filter, the
   UNCERTAIN state and human review, not solved.
3. **Small evaluation sets** - 84 val and 59 test images; per-class figures, especially
   `mask` (29 instances), carry wide error bars.
4. **Training images are pre-augmented mosaics**, evaluation images are clean photos - the
   two splits are not identically distributed.
5. **Domain gap** - trained on construction-site photography; an indoor webcam demo performs
   materially worse than section 1 suggests, and those numbers should not be quoted for it.
6. **Masking is not anonymisation** - a pixelated face band removes the most identifying
   region, but build, clothing, context and gait can still identify a worker. The band is
   also placed by geometry, not by finding the face, so an unusual pose can leave part of a
   face visible. See `../decisions.md` D-024 and D-025.
7. **Sensor noise defeats the detector** - at sigma 15 (a cheap sensor at high gain, i.e.
   night footage) person detection falls to 39 % and helmet to 45 %, while brightness and
   darkness are tolerated well. Night operation is out of scope, and this is the number that
   says why.
8. **Video analysis samples frames** (every 5th by default) and therefore does not use the
   temporal filter, which needs consecutive frames.
