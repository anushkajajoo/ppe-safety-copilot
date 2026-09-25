# Technology notes — viva preparation

For every technology: **WHAT** it is, **WHY** we use it, **HOW** it works,
and **WHY it suits this project**.

---

## 1. Python

* **What** — a high-level, interpreted programming language.
* **Why** — it has the largest ecosystem for machine learning and computer
  vision, and readable syntax we can explain line by line.
* **How** — we write `.py` files; the interpreter executes them. Code is
  grouped into *packages* (folders with `__init__.py`) so we can run
  `python -m edge.detect` and import `shared.config` from anywhere.
* **Why suitable** — every library this project needs (Ultralytics, PyTorch,
  OpenCV, FastAPI) is a Python library, so one language covers training,
  inference and the web backend.

## 2. Virtual environment (`.venv`)

* **What** — a private folder holding a copy of Python plus only this
  project's libraries.
* **Why** — the laptop also has Anaconda and other projects; without
  isolation, one `pip install` can break another project.
* **How** — `python -m venv .venv` creates it; activating it puts
  `.venv\Scripts` first on PATH, so `python` and `pip` mean *this* project's.
* **Why suitable** — makes the project **reproducible**: `requirements.txt`
  plus a fresh `.venv` recreates the exact environment on any machine.

## 3. Computer vision

* **What** — the field that extracts meaning from images and video.
* **Why** — PPE compliance is a *visual* question: is this person wearing a
  helmet?
* **How** — an image is a grid of pixel numbers (height × width × 3 colour
  channels). A convolutional neural network slides small learned filters over
  that grid; early layers respond to edges and colours, later layers to whole
  objects such as a helmet.
* **Why suitable** — the safety rule we are automating is exactly what a
  supervisor checks by eye, so vision is the natural sensor. No wearables or
  RFID tags are needed.

## 4. Object detection vs. classification

* **Classification** answers *what is in this image?* (one label).
* **Detection** answers *what is in this image, and where?* — it returns a
  **bounding box** plus a class plus a confidence for every object.
* We need detection, because compliance depends on *which* helmet belongs to
  *which* person — positions matter.

## 5. YOLO (You Only Look Once)

* **What** — a family of single-stage object detectors. We use
  **YOLOv8-nano** through the Ultralytics library.
* **Why** — it is fast, small (~6 MB), accurate enough, and has a clean
  Python API for both training and inference.
* **How** — the whole image goes through the network **once**. The network
  divides the image into a grid and predicts, at several scales, box
  coordinates + objectness + class scores. Non-Maximum Suppression (NMS) then
  removes duplicate overlapping boxes, keeping the most confident one.
  ("You only look once" = one forward pass, unlike older two-stage detectors
  such as Faster R-CNN which first propose regions and then classify them.)
* **Why suitable** — real-time speed on a laptop GPU/CPU, a nano variant that
  trains in minutes on ~1000 images, and it runs at the edge without a server.

## 6. PyTorch

* **What** — the deep-learning framework YOLO is built on.
* **Why** — it provides tensors, automatic differentiation, GPU acceleration
  and the optimiser; Ultralytics is a wrapper over it.
* **How** — data flows through layers as **tensors**; PyTorch records the
  operations in a computation graph, `loss.backward()` computes gradients by
  the chain rule, and the optimiser (SGD/AdamW) updates the weights. With
  CUDA, those tensor operations run on the GPU's thousands of cores.
* **Why suitable** — CUDA support lets the RTX 3050 train the model, and the
  same `.pt` checkpoint loads later for inference. `torch.cuda.is_available()`
  is also how we prove whether GPU training is possible.

## 7. Transfer learning

* **What** — starting from a model already trained on a huge dataset
  (COCO, 80 classes, ~118k images) instead of random weights.
* **Why** — we only have ~1000 images. Training from scratch would need
  100× more data and time, and would overfit badly.
* **How** — the pretrained backbone already encodes generic visual features
  (edges, textures, shapes, "person-ness"). We keep those weights, replace the
  detection head with one that outputs our 4 classes, and fine-tune the whole
  network with a small learning rate for a few dozen epochs.
* **Why suitable** — this is the single reason a small dataset + 30 epochs can
  give a usable PPE detector on a student laptop.

## 8. YOLO annotation format

* **What** — one `.txt` label file per image, same filename, one line per
  object:
  `class_id  x_center  y_center  width  height`
* All four numbers are **normalised to 0–1** (divided by image width/height),
  so labels survive resizing.
* Example: `1 0.512 0.233 0.104 0.087` = a *helmet* (class 1) centred slightly
  right of middle, near the top, about 10% of the image wide.
* **Why suitable** — it is plain text, human-readable, tiny, and the format
  Ultralytics expects. Our class IDs are fixed in `shared/config.py`
  (0 person, 1 helmet, 2 vest, 3 mask) and locked by a unit test, because a
  silent reordering would train the model on wrong labels.
* `data.yaml` tells YOLO where train/val/test folders are and lists the class
  names **in ID order**.

## 9. Dataset splits

* **Train** — the model learns from these images.
* **Validation** — used *during* training to pick the best epoch and trigger
  early stopping. The model never learns from it, but we tune against it.
* **Test** — touched **once**, at the very end, to report honest performance.
* Mixing validation and test results is a classic mistake and a favourite
  viva question — that is why `training/validate.py` takes an explicit
  `--split`.

## 10. Evaluation metrics

* **Precision** — of the objects the model reported, how many were real.
* **Recall** — of the real objects, how many the model found.
  For safety, recall on `person` matters most: a missed person is a missed
  violation.
* **IoU** (Intersection over Union) — overlap between predicted and true box;
  a prediction counts as correct above an IoU threshold.
* **mAP50** — mean Average Precision at IoU 0.5 (the usual headline number).
* **mAP50-95** — averaged over IoU 0.5 → 0.95 in steps of 0.05; stricter,
  rewards precise box placement.

## 11. OpenCV

* **What** — the standard computer-vision library (`cv2`).
* **Why** — we need to read images/video, open the webcam, draw boxes and
  labels, blur faces, and show a live window.
* **How** — images are NumPy arrays in **BGR** order (not RGB — a common
  bug). `cv2.VideoCapture(0)` opens the default camera and `.read()` returns
  frames in a loop; `cv2.rectangle` / `cv2.putText` draw the overlay;
  `cv2.imshow` displays it.
* **Why suitable** — it is fast C++ under a Python API, works offline, and
  handles image, video file and webcam with the *same* interface, which is
  exactly what `--source` needs.

## 12. Edge inference

* **What** — running the model where the data is created (this laptop, or a
  camera-side device) instead of sending frames to a server.
* **Why** — privacy, latency, bandwidth and offline operation.
* **How** — the trained weights are loaded once into memory; each frame is
  resized to 640×640, normalised, passed through the network, and the boxes
  are mapped back to the original frame size. Nothing is transmitted.
* **Why suitable** — this is the core claim of the project title: camera
  footage of workers is sensitive, and the safest place for it is the machine
  that captured it.

## 13. PPE compliance logic (deterministic rule engine)

* **What** — plain Python rules in `shared/compliance.py` that turn raw
  detections into a decision.
* **Why separate from YOLO?** — the neural network is a *statistical* model;
  the safety decision must be **deterministic, auditable and explainable**.
  Keeping the rules in their own module means we can unit-test them without a
  GPU, and change the site's rules by editing one list.
* **How** —
  1. take all detections for a frame;
  2. for each `person` box, find PPE boxes that overlap it (≥50% of the PPE
     box inside the person box);
  3. compare the set of associated PPE against `REQUIRED_PPE`;
  4. output `COMPLIANT`, `MISSING_HELMET`, `MISSING_VEST` or
     `MISSING_HELMET_AND_VEST` (plus `UNCERTAIN` when confidence is in the
     low band).
* **Why suitable** — simple, explainable in one sentence to an examiner, and
  configurable: change `REQUIRED_PPE` and the whole system's policy changes.

## 14. FastAPI

* **What** — a modern Python web framework for building APIs.
* **Why** — it gives the project a backend and a dashboard without the weight
  of Django or a React app.
* **How** — you declare functions decorated with `@app.get("/health")` or
  `@app.post("/predict")`; FastAPI validates the request against **Pydantic**
  models (type hints become validation rules) and generates interactive
  documentation at `/docs` automatically. **Uvicorn** is the ASGI server that
  actually runs it.
* **Why suitable** — few lines of code, automatic docs that make the API easy
  to demonstrate in a viva, and it still runs locally, so the privacy story
  is unchanged.

## 15. Pytest

* **What** — the testing framework.
* **Why** — to prove the project works and to catch silent breakages (e.g.
  someone reordering the class list).
* **How** — any function named `test_*` is collected and run; a plain
  `assert` that fails is reported with the values involved.
* **Why suitable** — tiny syntax, no boilerplate classes, and it is the
  automated-testing evidence the project rubric asks for.

## 16. Git / GitHub

* **What** — version control and remote hosting.
* **Why** — history, branches, pull requests and proof of individual
  contribution.
* **How** — `git add`/`commit` records snapshots; `.gitignore` keeps the
  dataset and `.pt` weights out of the repository (they are large binaries).
* **Why suitable** — the project must be reproducible from the repo alone:
  code + `requirements.txt` + the dataset preparation script, not the data
  itself.
