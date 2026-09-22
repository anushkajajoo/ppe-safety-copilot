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
