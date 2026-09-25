# Release notes

Versions are milestones of the capstone build, not published packages. Each entry says what
changed, what it was measured at, and what was knowingly left open - the same information a
reviewer would ask for at a release meeting.

Format: [Keep a Changelog](https://keepachangelog.com/). Dates are the build dates.

---

## [0.5.0] - 2026-09-24 - Governance and evidence

### Added
- **Human-governed copilot** (`shared/copilot.py`, `server/routes/copilot.py`): five
  allow-listed actions, one proposal at a time, approval by a named actor, and every
  proposal and decision appended to the hash-chained audit log. Dashboard panel included.
- `docs/threat_model.md` - assets, trust boundaries, eight STRIDE threats, six misuse cases,
  residual risks.
- `docs/problem_brief.md` - stakeholder map, user stories, scope exclusions, backlog.
- CI pipeline (`.github/workflows/ci.yml`): tests on Python 3.11 and 3.12, ruff, `pip-audit`
  dependency scan, committed-`.env` check, and a container build that must answer `/health`.
- `Dockerfile` - CPU-only, headless OpenCV, non-root, health-checked.

### Changed
- Removed `streamlit`, `plotly` and `pandas` from `requirements.txt`: nothing imported them
  after the dashboard was rebuilt as plain HTML (D-023). An unused dependency is still
  something the scanner has to check.

### Measured
- 233 tests, ~14 s, none requiring a GPU, camera or model file.

---

## [0.4.0] - 2026-09-23 - Privacy, measured and corrected

### Added
- Event snapshot masking (`shared/privacy.py`), off by default, fail-closed.
- `scripts/privacy_utility.py` - measures what masking costs the evidence.
- `scripts/face_band_geometry.py` - measures where PPE sits inside a person box, from labels.

### Fixed
- **The face band was in the wrong place.** The first design destroyed two thirds of helmet
  evidence (33.3 % preserved). The band is now placed by interpolating on the person box's
  aspect ratio, and detected helmet/vest boxes are restored after pixelation. Helmet
  preservation 33.3 % -> 93.3 %. See D-025.
- **The metric was wrong too.** Counting detections reported 129.4 % "retention" for vest,
  because masking can make the detector invent boxes. Detections are now paired by overlap
  and reported as preserved / lost / spurious.

### Known open
- Masking is not anonymisation; person-box preservation is 52.4 % and is not what the
  snapshot is for.

---

## [0.3.0] - 2026-09-22 - Interface and operations

### Added
- Live webcam streaming (MJPEG), video upload analysis, violation log API and dashboard.
- `run.py` - one command, port pre-check, binds 127.0.0.1.
- Adaptive device selection with a VRAM guardrail (`shared/device.py`).

### Changed
- Pages moved to `frontend/`, rebuilt as plain HTML + CSS + JS with no build step (D-023).

---

## [0.2.0] - 2026-09-21 - Decisions, not detections

### Added
- Temporal filter (15-frame window), zones, deterministic compliance rules, event builder,
  hash-chained audit log, SQLite event store.

### Measured
- Temporal filter vs single frame: **30 false alarms -> 0**, at a cost of 14 frames (0.6 s).
- Safety rules cost 0.03 ms per frame (~0.1 % of frame time).

---

## [0.1.0] - 2026-09-20 - Detector

### Added
- YOLO26n transfer-learned on a 1 143-image seeded subset, 4 classes, 30 epochs on a T4.
- Person-PPE association by body region with IoA >= 0.5, one PPE box per person.

### Measured
- Test split: precision 0.896, recall 0.708, mAP50 **0.804**, mAP50-95 0.447.

### Known open
- Vest detection fails on dark/navy hi-vis (confidence 0.070). Kept as a documented failure
  rather than fixed by lowering the threshold (D-014).
