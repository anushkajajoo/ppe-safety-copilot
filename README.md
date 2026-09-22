# Edge Vision Safety Copilot (BAI-01)

Zone-aware edge-AI PPE compliance monitoring with anonymous tracking, privacy-preserving
evidence, human review and an audit trail. Academic MVP, 2-person team.

> Status: **Day 1 — foundation** (environment, dataset, fine-tuning started, backend skeleton).

## Repository layout

```
shared/     schemas.py            event contract used by BOTH edge and server
server/     main.py config.py database.py models.py     FastAPI + SQLAlchemy + SQLite
edge/       smoke_webcam.py       Day 1 camera + YOLO benchmark (full pipeline from Day 2)
training/   prepare_sh17.py train.py validate.py         dataset conversion + fine-tuning
scripts/    check_env.py init_db.py
tests/      pytest tests
docs/       decisions.md, evaluation/ (real measured results only)
datasets/ models/ runs/ data/     gitignored (large or private)
```

## Setup on Windows (both teammates)

Requirements: Windows 10/11, NVIDIA driver installed, **Python 3.11 (64-bit)**, Git.

```powershell
# 1. get the code
git clone <your-repo-url> ppe-safety-copilot
cd ppe-safety-copilot

# 2. virtual environment
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip

# 3. PyTorch WITH CUDA (do this BEFORE requirements.txt, otherwise you may get the CPU build)
#    Check https://pytorch.org/get-started/locally/ for the current command. At the time of writing:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 4. everything else
pip install -r requirements.txt

# 5. verify
python -m scripts.check_env           # writes docs/evaluation/env_report.json
python -m pytest                      # all tests should pass
```

## Person 1 — Day 1 commands

```powershell
python -m edge.smoke_webcam                     # pretrained YOLO on webcam, press Q; prints measured FPS
python -m training.prepare_sh17 --src "D:\datasets\SH17" --out datasets\ppe4
python -m training.train --epochs 3 --fraction 0.1 --name smoke    # 5-10 min sanity run
python -m training.train                        # real run (hours); resume with --resume
```

## Person 2 — Day 1 commands

```powershell
copy .env.example .env
python -m scripts.init_db                       # creates data\server.db and lists tables
uvicorn server.main:app --reload --port 8000    # http://127.0.0.1:8000/health  and  /docs
```

## Training on Google Colab (backup if the laptop is too slow / out of memory)

Zip `datasets/ppe4` (after prepare_sh17), upload it to Google Drive, then in a GPU Colab notebook:

```python
!pip install -q "ultralytics>=8.4.0"
from google.colab import drive; drive.mount('/content/drive')
!unzip -q /content/drive/MyDrive/ppe4.zip -d /content/
# fix the dataset path inside data.yaml
import yaml; d = yaml.safe_load(open('/content/ppe4/data.yaml')); d['path'] = '/content/ppe4'
yaml.safe_dump(d, open('/content/ppe4/data.yaml', 'w'), sort_keys=False)
from ultralytics import YOLO
YOLO('yolo26n.pt').train(data='/content/ppe4/data.yaml', epochs=60, imgsz=640, batch=16, seed=42,
                         project='/content/drive/MyDrive/ppe_runs', name='ppe4_yolo26n_colab')
```
Download `best.pt` into `models/` and record in `docs/decisions.md` that it was trained on Colab.

## Licences

- Code: Ultralytics is AGPL-3.0, so this repository should be public/open-source under a compatible licence.
- Dataset: SH17 is CC BY-NC-SA 4.0 (non-commercial, share-alike). Academic use is fine; cite the paper.
