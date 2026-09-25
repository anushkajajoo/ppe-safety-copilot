# Test strategy

What is tested, what is deliberately not, and why the suite can run on a laptop with no GPU,
no camera and no model file.

---

## 1. The principle

Every test must run on any machine, in seconds, with no hardware. That is not a convenience
- it is what makes CI possible at all, and a test suite that only runs on the author's
machine is not evidence.

This forced a design decision early: anything that needs hardware sits behind a seam.

| Hardware thing | The seam | What tests use instead |
|---|---|---|
| The YOLO model | `server/predict_service.py`, `app.state.predict_service` | `FakeService` returning fixed `Detection` objects |
| The webcam | `app.state.camera_factory` | a fake camera yielding synthetic frames |
| The GPU | `shared/device.resolve_device(torch_module=...)` | a fake torch module, so both "GPU present" and "no GPU" branches are covered on one machine |
| The database | `Settings(database_url=...)` | a SQLite file in pytest's `tmp_path` |
| The violation log / outbox | `Settings(violation_log=...)`, monkeypatched `OUTBOX` | files in `tmp_path`, never the real `data/` |

Consequence: **233 tests in about 14 seconds**, and the same suite runs in CI on two Python
versions.

---

## 2. Levels

**Unit** - pure functions with hand-written numbers. Geometry (`ioa`, `iou`, point-in-polygon),
association regions, compliance rules, the temporal window, the copilot rule table, face-band
placement. No I/O at all. These are the tests that make a viva answer checkable: "a helmet
must be 50 % inside the head region" is a line of code *and* an assertion.

**Integration** - the FastAPI app assembled for real (routes, settings, database, audit
chain) with only the detector and camera faked. Covers the image, video, live, events, zones,
violations, audit and copilot APIs, including their error paths.

**Contract** - the things that would silently invalidate everything if they drifted: the class
order `person, helmet, vest, mask`, the event schema version, and the fact that both web pages
exist and are served from `frontend/`.

**Property-style** - a small number of sweeps rather than single examples. The important one:
every combination of status x zone x repeat-count is fed to the copilot and asserted to
produce only allow-listed actions. A single example would not show that.

**Regression** - each bug that was found got a test before it was fixed. The face-band
geometry, the benchmark writing both runs to one file, the duplicate-suggestion guard and the
"never pair one box twice" matcher are all in the suite because they were wrong once.

---

## 3. What is covered, by the brief's own categories

| The brief asks about | Where |
|---|---|
| Object detection plumbing | `test_detect.py`, `test_edge_logic.py` |
| PPE-person association | `test_detect.py`, `test_edge_logic.py` |
| Confidence / compliance rules | `test_edge_logic.py`, `test_compare_modes.py` |
| Event snapshot masking | `test_privacy.py` (21) |
| Alert dashboard and APIs | `test_predict_api.py`, `test_video_api.py`, `test_live_api.py` |
| Human approval boundaries | `test_copilot.py` (24) |
| Excessive tool permissions | `test_copilot.py` - the allow-list sweep |
| Audit / traceability | `test_copilot.py`, `test_events_api.py` (hash-chain verify) |
| Data pipeline reproducibility | `test_prepare_ppe4.py`, `test_check_dataset.py` |
| Deployment surface | `test_run_py.py`, `test_packaging.py` |
| Authentication and roles | `test_auth.py` - hashing, forged and expired sessions, role order |
| Intermittent connectivity | `test_outbox.py` - outage, backoff, restart, poison messages |
| Agent safety | `test_agent_safety.py` - injection, leakage, tool permissions, approval boundary |
| Retention | `test_retention.py` - expiry, dry run, what must never be deleted |
| Acceptance gate | `test_acceptance.py` - a missing report must never read as a pass |

---

## 4. What is NOT tested, and why

- **Model accuracy is not a unit test.** It is measured, versioned and reported in
  `docs/evaluation/`, because accuracy is a distribution property, not a pass/fail assertion.
  A test asserting `mAP > 0.8` would fail for the right reason on new data and teach nothing.
- **Real camera hardware.** Cannot run in CI; covered by the seam above and by manual demo.
- **The trained weights file.** 5 MB of artefact, not source. Tests must pass without it.
- **Browser rendering.** The pages are asserted to exist, be served and contain their panels;
  the visual result is checked by a human. Adding Playwright would be more infrastructure than
  the project needs.
- **Load at scale.** One laptop, one camera. Latency and frame rate are measured
  (`scripts/benchmark_edge.py`), not asserted.

---

## 5. How a change gets accepted

1. The test suite passes locally (`python -m pytest`).
2. CI passes on Python 3.11 and 3.12, ruff is clean, `pip-audit` finds nothing, the container
   builds and answers `/health`.
3. Anything measured is re-measured and the number written into `docs/evaluation/`, not into
   a commit message.
4. Anything decided is written into `docs/decisions.md` with its reason - including the
   decisions that turned out to be wrong.
