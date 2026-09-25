# Administrator guide

For the person who installs this, keeps it running, and is responsible for the data it
holds. The user guide is the README; this is everything that happens after the demo.

---

## 1. Install on a clean machine

```powershell
git clone <your repository>
cd ppe-safety-copilot
python -m venv .venv
.venv\Scripts\activate

# PyTorch first - the CUDA build if the machine has an NVIDIA GPU, otherwise CPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

copy .env.example .env          # then edit it - see section 2
python -m scripts.check_env     # confirms Python, torch, CUDA, OpenCV, the model file
python -m pytest                # 342 tests, none needing a GPU, camera or model
python run.py
```

The model file (`models/ppe4_yolo26n_best.pt`, ~5 MB) is **not** in the repository. Copy it
in, or retrain with `training/colab_train.ipynb`.

**Container instead:** `docker build -t ppe-safety-copilot .` then
`docker run -p 8000:8000 -v ${PWD}/models:/app/models -v ${PWD}/data:/app/data ppe-safety-copilot`.
The image runs as a non-root user and reports healthy only once `/health` answers. It has no
camera access - the container is for the API and the dashboard, not live capture.

---

## 2. Settings that matter

All settings are environment variables prefixed `PPE_`, read from `.env`
(`server/config.py`). The ones with consequences:

| Setting | Default | Why you would change it |
|---|---|---|
| `PPE_EDGE_API_KEY` | `change-me` | **Change this.** It is the shared secret an edge device uses to post events and telemetry. |
| `PPE_STORE_SNAPSHOTS` | `false` | `true` starts keeping masked images of violations. This turns the system into one that holds personal data - see section 5 before you do it. |
| `PPE_EVIDENCE_RETENTION_DAYS` | `30` | How long those images live. Enforced by `scripts.retention`, which you have to schedule (section 4). |
| `PPE_SESSION_SECRET` | empty | Empty means a new secret each start, so everyone signs in again after a restart. Set a long random value to keep sessions across restarts. |
| `PPE_DETECT_DEVICE` | `auto` | `auto` uses the GPU only when it has ≥1500 MB free, else the CPU. Force with `cpu` or `0`. |
| `PPE_DETECT_CONF` | `0.35` | The detection threshold. **Do not lower it to make a demo pass** - see D-014. |
| `PPE_VIOLATION_COOLDOWN_S` | `30` | How often the same person+status may be logged. Lower it and a live camera floods the log. |

`.env` is gitignored and must stay that way. CI fails the build if a `.env` is ever committed.

---

## 3. Accounts and roles

```powershell
python -m scripts.make_user --username ana --role supervisor --display-name "Ana Silva"
python -m scripts.make_user --list        # shows accounts, never hashes
```

| Role | Can |
|---|---|
| `viewer` | see the dashboard, the queue and the log |
| `supervisor` | also approve or reject what the copilot proposes |
| `admin` | also configuration and maintenance |

Accounts live in `configs/users.yaml` as PBKDF2 hashes with a per-user salt. **The file is
per-machine and gitignored** - it is not source, and copying it between machines copies
credentials.

To remove someone, delete their block from `configs/users.yaml`. There is no delete command
on purpose: removing access should be a deliberate edit, not a flag on a script.

**Rotate** the shared `PPE_EDGE_API_KEY` and any account password when someone leaves.
Sessions issued before a restart stop working as soon as the process restarts, unless you
have set a fixed `PPE_SESSION_SECRET`.

---

## 4. Routine jobs

| When | Command | Why |
|---|---|---|
| Weekly | `python -m scripts.retention --dry-run` then without it | Deletes expired snapshots. Nothing deletes them by itself. |
| Weekly | `python -m scripts.check_acceptance` | Confirms every claim still clears its threshold |
| After any change | `python -m pytest` | 342 tests, about 30 seconds |
| After changing the model or the machine | `python -m scripts.measure_resources`, `python -m scripts.robustness_lighting`, `python -m training.validate --split test` | The old numbers describe the old model |
| Monthly | `pip-audit -r requirements.txt` | Known vulnerabilities in pinned dependencies (CI does this per build) |

**Backups.** Everything worth keeping is in two places: `data/` (the violation log, the
database, any stored evidence) and `configs/` (zones, policies, accounts). Copy both. The
model can be retrained; the audit chain cannot be reconstructed.

Verify a backup by checking the chain: `GET /api/v1/audit/verify` must answer
`{"ok": true}`. If it does not, a row was edited or lost, and the response names the first
bad one.

---

## 5. Before you turn snapshots on

`PPE_STORE_SNAPSHOTS=true` is the single most consequential setting in the system: it
changes it from holding metadata to holding pictures of workers. Faces are pixelated before
anything is written, and a masking failure writes nothing at all - but pixelation is not
anonymisation, and build, clothing and context still identify people.

Before enabling it, have in place: a lawful basis and a retention period agreed with
whoever is responsible for staff data; signage telling people the area is monitored;
a scheduled `scripts.retention`; and a decision about who may read `data/evidence/`.

If you cannot answer those, leave it off. The system is fully usable without it.

---

## 6. When something goes wrong

| Symptom | What it means | Fix |
|---|---|---|
| `503` from `/api/v1/predict` with "Model not found" | the weights file is missing | copy `ppe4_yolo26n_best.pt` into `models/` |
| "Could not open webcam" | another program holds the camera | close it; only one program can. `--source 1` for a second camera |
| Port 8000 already in use | an older server is still running | `python run.py --port 8001`, or close the terminal holding it |
| The dashboard is stale after a code change | uvicorn caches the app | restart `run.py`. Editing an HTML page needs only a hard refresh |
| Everyone has to sign in again | the process restarted and the session secret is random | set `PPE_SESSION_SECRET` in `.env` |
| The edge queue keeps growing | the server is unreachable | check `/api/v1/telemetry`; events are safe on disk and deliver when it returns |
| A device disappears from telemetry | the device stopped, not the site going quiet | a heartbeat older than 120 s is flagged `stale` |
| Detections are poor on a new site | domain gap - the model was trained on daylight construction photography | collect and label site images; do not lower the threshold |
| `check_acceptance` says NO-GO | a claim the project makes is no longer supported | read which criterion failed; fix the system or change the claim, never the threshold alone |

**Logs.** The server prints each request to its terminal. Application records are in
`data/violations.jsonl` and `data/server.db`; operational health is at `/api/v1/telemetry`;
decisions are at `/api/v1/audit`.

---

## 7. What this system is not

It is a prototype for a single site on a single machine. Before it could run for real it
needs: HTTPS and an identity provider rather than local accounts; rate limiting on sign-in;
enforced retention as a scheduled service rather than a manual command; and a review of the
detector against footage from the actual site. Those are named in
`docs/threat_model.md` §6 and `docs/problem_brief.md` §6, with the reasons.
