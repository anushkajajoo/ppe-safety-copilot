# Detector results - ppe4_yolo26n

Model: YOLO26n fine-tuned from COCO weights (transfer learning) on our 4-class subset.
Trained on Google Colab (Tesla T4), 30 epochs, batch 16, imgsz 640, patience 8, seed 42,
mosaic 0.0. Wall-clock: 560 s (~9.3 min). Ultralytics 8.4.160, torch 2.11.0+cu130.
Full run record: `models/ppe4_yolo26n_info.json`, `docs/evaluation/train_args.yaml`.

## Headline numbers

| split | images | precision | recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| val  | 84 | 0.885 | 0.735 | **0.828** | 0.482 |
| test | 59 | 0.896 | 0.708 | **0.804** | 0.447 |

The test score sits just below the validation score (0.804 vs 0.828 mAP50). That small gap is
what an honest split looks like: the model was selected on val and never saw test.

## Per class (TEST split - the reported numbers)

| class | instances | precision | recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| person | 175 | 0.812 | 0.703 | 0.771 | 0.418 |
| helmet | 111 | 0.955 | 0.765 | 0.874 | 0.523 |
| vest   |  60 | 0.817 | 0.683 | 0.782 | 0.416 |
| mask   |  29 | 1.000 | 0.679 | 0.788 | 0.430 |

## Reading the numbers

* **Precision is high, recall is lower** (0.90 vs 0.71) for every class. The detector rarely
  invents PPE, but it misses some. For a safety system that bias is the wrong way round: a
  missed helmet means a violation is not raised. This is why the pipeline uses a temporal
  filter over 15 frames and a human review step instead of acting on one frame.
* **mask has precision 1.000 and recall 0.679** on 29 instances. Perfect precision on 29 boxes
  is not evidence of a great mask detector - the sample is tiny and one missed mask moves
  recall by 3 points. Always quote the instance count next to this number.
* **person is the weakest class by mAP50 (0.771)** even though it has the most boxes. People
  appear at every scale, are often half-occluded and overlap each other; helmets are compact,
  high-contrast and consistently shaped, which is why helmet scores highest.
* **mAP50-95 (0.447) is far below mAP50 (0.804).** The model finds the right objects but its
  boxes are not tightly placed. That is normal for a nano model at 640 px and matters little
  here: compliance is decided by which PPE box overlaps which person, not by pixel-perfect
  edges.

## Training curves (`results.png`)

Train and validation losses were **still falling at epoch 30**, and mAP was still creeping up.
Early stopping never triggered (patience 8). So the model is **under-trained, not overfitted** -
a longer run (50-60 epochs) would likely add a little accuracy. It was left at 30 because the
project targets a working, fast-to-reproduce system rather than a maximum benchmark score.

## Limitations to state in the report

1. **Small evaluation sets** - 84 val and 59 test images. Every metric carries wide error bars;
   per-class numbers on `mask` (29 instances) especially so.
2. **Training images are Roboflow mosaics** (four photos tiled with cutout squares), while the
   evaluation images are clean single photos. Train and test are therefore not from identical
   distributions; see `docs/decisions.md` D-010a.
3. **Domain gap to the live demo** - the dataset is construction-site photography; the webcam
   demo is an indoor room. Detection quality on the webcam feed will be worse than these
   numbers suggest, and that should be measured separately, not inferred from this table.
4. **No absence classes** - the model does not detect "no helmet". Non-compliance is derived by
   the rule engine from person-PPE association, so a missed PPE box and an absent PPE item look
   the same to the rules. The confidence band and temporal filter exist to soften that.

## Hard case: dark hi-vis jacket (samples/compliant_candidate.jpg)

A construction worker wearing a **red hard hat and a navy hi-vis jacket**. The helmet is
detected at 0.899 and the person at 0.821, but the jacket reaches only **0.070** - below the
0.35 threshold - so the system reports `MISSING_VEST` on a person who is, to a human, wearing
high-visibility clothing.

Lowering `--conf` to 0.01 shows five overlapping vest boxes (0.070, 0.046, 0.035, 0.031,
0.021) spread over the torso: the model sees *something* but cannot settle on it. The best box
sits 70 % inside the torso region, so association would have worked had the box cleared the
gate.

This is kept as evidence rather than fixed by tuning. Dropping the threshold to 0.05 would
pass this image and simultaneously invent vests elsewhere - in a safety system, a false
"vest present" marks an unprotected worker as compliant, which is the failure direction that
actually hurts someone. The honest statement for the report is: *the detector's vest class
generalises to bright mesh vests and fails on dark hi-vis garments, because that is what the
training data contained.*

## Does the temporal filter earn its place? (measured)

`python -m scripts.compare_modes` runs the real `edge/temporal.py` and
`edge/compliance.py` over two simulated scenarios (200 frames each, seed
42, window 15). The detector's output is
simulated rather than run, so the experiment is exact and repeatable.

| Scenario | mode | VIOLATION | UNCERTAIN | COMPLIANT | first flag |
|---|---|---|---|---|---|
| A - detector blinks (worker IS compliant, helmet missed in 15% of frames) | baseline | **30** (false alarms) | 0 | 170 | frame 2 |
| | proposed | **0** | 22 | 178 | never |
| B - real violation (no vest at all) | baseline | 200 (correct) | 0 | 0 | frame 1 |
| | proposed | 186 (correct) | 14 | 0 | frame 15 |

**Result: 30 of 30 false alarms removed, at the cost of
14 frames (0.6 s at 25 FPS) before a genuine violation is confirmed.**

That trade is worth taking here. A supervisor who is shown thirty false alarms in two
hundred frames stops reading the alerts; half a second of delay before a confirmed
violation changes nothing on a construction site. Note also that while the window is
still filling, the proposed mode answers UNCERTAIN, never COMPLIANT - it does not claim
safety on thin evidence.

Raw numbers: `docs/evaluation/temporal_comparison.json`.
