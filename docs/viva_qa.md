# Viva questions and answers

Every number here is from this project's own measurements. If you are asked something not
on this list, the honest answer - "I measured X, I did not measure Y" - is always better
than a guess: examiners test whether you know the difference.

---

## A. The problem and the idea

**1. What does your project do, in one sentence?**
It watches a camera feed on a construction site, detects each worker and the PPE they are
wearing, and decides whether the required PPE (helmet and vest) is present - doing all the
inference on the local machine so camera footage never leaves it.

**2. Why is this worth building?**
PPE checks on a site are done by a supervisor walking around. That is intermittent, and the
moment a violation happens is usually not the moment someone is watching. A camera watches
continuously. The system does not replace the supervisor - it tells them where to look.

**3. Why "privacy-preserving"? Everyone says that.**
Three concrete things, not a slogan: inference runs in the local process (no cloud vision
API anywhere in the code path), the system detects *a person* but never *which* person (no
face recognition, no identity database; tracker IDs reset every session), and the violation
log stores metadata only - a test asserts no image content reaches it. The server also binds
127.0.0.1, not 0.0.0.0, so it is not reachable from the network by default.

**4. Is it 100% private?**
No, and I do not claim that. If the operator enabled evidence snapshots, images would exist
on that machine. The defensible claim is: *inference is local and raw camera data is not
transmitted by this system*.

---

## B. Dataset

**5. What data did you train on?**
A 1 143-image subset of the Roboflow Universe *Construction Site Safety* dataset (CC BY 4.0):
1 000 train, 84 validation, 59 test. Four classes - person, helmet, vest, mask - remapped
from the source's ten.

**6. Why only 1 143 images? Isn't more better?**
More is better for accuracy, but the project is optimised for a working, reproducible system
on a laptop. 1 000 images with transfer learning trains in 9 minutes and reaches 0.80 mAP50
on the test split. Building the subset is scripted and seeded, so anyone can reproduce the
exact same dataset.

**7. How did you choose those images?**
`training/prepare_ppe4.py` with seed 42. It groups images by the ORIGINAL photo (Roboflow
names augmented copies `<photo>_jpg.rf.<hash>`), so the same photo can never land in two
splits, then samples class-aware: rarest classes first, so `mask` and `vest` survive. Six
photos appeared in two source splits; those went to the evaluation split.

**8. Why did you drop the NO-Hardhat / NO-Safety-Vest classes?**
Because absence of PPE is a *decision*, not an *object*. The detector says what it sees; the
rule engine derives "missing helmet" from a person with no associated helmet. That keeps the
model smaller and the safety decision auditable and explainable.

**9. What is wrong with your dataset?**
Two things I state openly. The source's training images are pre-augmented mosaics - four
photos tiled with cutout squares - while validation and test are clean photos, so the splits
are not identically distributed. And `vest` instances are overwhelmingly bright mesh vests,
which causes the failure in question 22.

---

## C. Model and training

**10. Which model, and why that one?**
Ultralytics YOLO26n - the nano variant, 2.51 M parameters, 5.9 GFLOPs. It is a single-stage
detector: one forward pass produces boxes, classes and confidences. Nano because it must run
in real time on a laptop; larger variants gain little on 1 000 images and cost VRAM I do not
have.

**11. Did you train from scratch?**
No - transfer learning from COCO weights. 612 of 708 weight tensors transferred, and the COCO
`person` head row was reused by name. Training from scratch on 1 000 images would overfit
badly; the pretrained backbone already knows edges, textures and "person-ness".

**12. What training settings, and why?**
30 epochs, batch 16, 640 px, patience 8, seed 42, mosaic 0.0, on a Colab T4 - 560 seconds.
Mosaic is OFF because the training images are *already* mosaics; letting Ultralytics mosaic
them again would produce mosaics-of-mosaics and shrink every object to a few pixels.

**13. Why train on Colab and not your laptop?**
To keep the laptop free, and because a T4 does in 9 minutes what a 4 GB RTX 3050 would take
much longer to do. Inference, the backend and the demo all run on the laptop.

**14. Is the model finished improving?**
No. Both losses were still falling at epoch 30 and early stopping never fired, so it is
under-trained rather than overfitted. A 50-epoch run is the obvious cheap improvement; I
prepared the workflow but Colab's free GPU quota was exhausted that day.

---

## D. Detection, association and compliance

**15. How do you decide that a helmet belongs to a particular person?**
`edge/association.py` builds body regions from each person box - a head region (top of the
box, extended slightly above it) and a torso region. A helmet or mask must overlap the head
region, a vest the torso region, with at least 50 % of the PPE box inside it (intersection
over the PPE box's own area). Each PPE box goes to at most one person - the best overlap - so
one helmet cannot make two people compliant.

**16. What if two people overlap?**
Then a PPE box can be credited to the wrong person. I state it as a limitation. The temporal
filter and the human review gate exist partly for this reason - a one-frame mis-assignment
rarely survives 15 frames.

**17. Where does the compliance decision live, and why not inside the model?**
`edge/compliance.py`, a deterministic rule engine, entirely separate from YOLO. The network
is statistical and its output is a probability; a safety decision must be explainable,
auditable and testable. Because the rules are separate, they are unit-tested with
hand-written boxes - no GPU, no camera. Measured bonus: the rules cost 0.03 ms per frame,
about 0.1 % of the time, so keeping them separate costs nothing.

**18. What statuses can the system produce?**
Single-frame: COMPLIANT, MISSING_HELMET, MISSING_VEST, MISSING_HELMET_AND_VEST. The full
pipeline adds UNCERTAIN and POTENTIAL_VIOLATION, because over time the honest answer is
sometimes "not enough evidence yet".

---

## E. Temporal logic and zones

**19. Why not just decide on every frame?**
I measured what happens if you do. With a detector that misses the helmet in 15 % of frames -
motion blur, a hand in the way - the single-frame rule raised **30 false alarms in 200
frames**. The 15-frame filter raised **zero**, and still caught a genuine missing vest, just
14 frames (0.6 s at 25 FPS) later. A supervisor shown 30 false alarms stops reading the
alerts; half a second of delay changes nothing on a site.

**20. How does the temporal filter work?**
Per worker, per PPE item, a sliding window of the last 15 frames: 1 if the item was detected,
0 if not. 10 or more missing out of 15 is a violation; 5-9 is UNCERTAIN; fewer than 5 is
treated as worn - the detector blinked. The window resets when the worker changes zone,
because the requirements changed.

**21. What are the safety zones and why pixels rather than GPS?**
Polygons drawn in the camera image (`configs/zones.yaml`), because a webcam has no GPS and a
site's "welding area" is a region of the view. A worker's position is the foot point - the
bottom-centre of their box - because that is where they stand. Each zone type carries its own
required PPE: GENERAL needs helmet + vest, CAUTION adds mask, RESTRICTED needs authorisation.

---

## F. Results and failures

**22. Your own demo image says MISSING_VEST when the worker is wearing hi-vis. Explain.**
That is a real, documented model failure and I keep it deliberately. The worker wears a
**navy** hi-vis jacket. Raw model output at conf 0.01: helmet 0.899, person 0.821, and vest
at only **0.070**, in five fragmented boxes over the correct torso area. So the model sees
*something* but cannot commit. It is not an association bug - the best vest box lies 70 %
inside the torso region, well over the 0.5 threshold - and not a class-mapping bug. The cause
is training data: 1 175 vest instances, nearly all bright mesh vests.

**23. Why not lower the threshold to 0.05 and make it pass?**
Because at 0.07 the model would accept almost any activation as a vest, and a false "vest
present" marks an *unprotected* worker as compliant. In a safety system that is the failure
that actually hurts someone. I would rather report a false alarm than miss a bare-chested
worker. The fix is more dark-vest training data, not a threshold.

**24. What are your test numbers?**
Test split, 59 images: precision 0.896, recall 0.708, mAP50 0.804, mAP50-95 0.447. Per class
mAP50: person 0.771, helmet 0.874, vest 0.782, mask 0.788. Validation was 0.828 mAP50 - the
test score sitting slightly below it is what an honest split looks like.

**25. Your mask precision is 1.000. Is mask detection solved?**
No. That is 1.000 on **29 instances**. One missed mask moves recall by 3 points. I always
quote the instance count next to it.

**26. Precision 0.90 but recall 0.71 - is that good?**
It is the wrong way round for safety. The detector rarely invents PPE but misses about three
in ten items, and a missed helmet means a real violation is never raised. That is exactly why
the pipeline has a temporal filter, an UNCERTAIN state and a human review step rather than
acting on one frame.

**27. Why is mAP50-95 so much lower than mAP50?**
The boxes are found but not tightly placed - normal for a nano model at 640 px. It matters
little here, because compliance depends on *which* PPE box overlaps *which* person, not on
pixel-perfect edges.

---

## G. System, performance and engineering

**28. How fast is it, and on what?**
Measured with `scripts/benchmark_edge.py`, 120 frames, warm-up discarded: on the RTX 3050,
22.12 ms per frame (31.16 ms at p95) = **45.2 FPS**; on the CPU, 47.88 ms = **20.9 FPS**. A
camera delivers 25-30 FPS, so the system is real-time on the GPU and keeps up on the CPU.

**29. Does it need a GPU?**
No. `shared/device.py` picks the GPU only when at least 1500 MB of VRAM is free, and falls
back to the CPU otherwise, printing which it chose and why. The fallback is a slowdown
(21 FPS), never a failure - and never an out-of-memory crash mid-demo.

**30. What is the architecture?**
Camera or upload → YOLO detection → person-PPE association → zone lookup → temporal filter →
compliance rules → status and event → FastAPI backend → dashboard and violation log. The
diagram is in `docs/architecture/architecture.md`. The privacy boundary is the line between
the edge pipeline and the backend: only structured events cross it, never frames.

**31. How is it tested?**
233 automated tests, running in about 14 seconds, none of which need a GPU, a camera or the
model file - the detector is replaced by a fake wherever it would be required. That covers
detection plumbing, association geometry, compliance rules, the temporal filter, zones, all
three APIs, the dashboard pages, the violation log, dataset preparation and the start command.

**32. What happens when something goes wrong?**
Missing model: a plain sentence on the CLI, HTTP 503 with the same sentence from the API - the
server still starts. Camera busy: "Could not open camera 0" with the fix. Port in use:
`run.py` says so and suggests `--port 8001`. Not a traceback in sight for the normal failures.

**33. Why FastAPI, SQLite and a JSONL log rather than something bigger?**
Each one is the smallest thing that does the job. FastAPI gives typed request validation and
automatic API docs in a few lines. A violation record is written once and read in order -
that is what an append-only text file is for. The SQLite database holds the tracked pipeline's
reviewed events, where relational queries genuinely help. No microservices, no message queue,
no Kubernetes: none of them would earn their complexity here.

---

## H. The copilot and governance

**H1. The project is called a copilot. Where is the copilot?**
`shared/copilot.py` and the **Copilot review queue** panel in the dashboard. When a worker
is non-compliant, the system proposes exactly one action - flag for review, notify the
supervisor, escalate a restricted-zone breach, or ask for a second look - with the rule that
produced it and the evidence it saw. It then waits. A named person approves or rejects it,
and only then does anything happen.

**H2. Why not let it act automatically when it is confident?**
Because confidence is the model's opinion about pixels, not about whether someone should be
pulled off a job. Recall is 0.708, so the system is wrong often enough that an automatic
action would sometimes be an unjustified one - and the person who has to defend that action
is the supervisor, not the model. There is deliberately no threshold anywhere that bypasses
a human.

**H3. How do you know it cannot do something it was not designed to do?**
`ACTIONS` is a five-item tuple, and every path that stores or executes goes through
`check_action()`, which raises on anything else - including an action id sent in an API
request. A property test sweeps every status x zone x repeat-count combination and asserts
no input produces an action outside the list. The list is also published at
`/api/v1/copilot/catalog` and shown in the dashboard, so a user can read the copilot's whole
vocabulary instead of trusting me about it.

**H4. Why is there no LLM in a system called a copilot?**
"Missing helmet in a restricted zone → escalate" is a policy, not a judgement. A rule table
is auditable, deterministic, runs in microseconds and cannot be argued into proposing
something off the list. An LLM would cost all four properties and buy nicer wording. If an
examiner wants the LLM version, the honest answer is that it belongs in the report's future
work, behind the same approval gate.

**H5. What stops the copilot from being gamed?**
Rule R4 escalates when a worker has been seen repeatedly. That count is read from the
database, not taken from the request, so a caller cannot claim a high count to trigger a
supervisor alert. And a suggestion can be decided only once - a second attempt returns 409
naming who decided first.

**H6. What does "approved" actually do?**
The two alerting actions append a line to `data/outbox.jsonl`. Nothing is sent anywhere -
which is the honest implementation for a system whose claim is that data stays local. In a
real deployment that file is the seam where a siren or SMS gateway attaches.

**H7. How would you know if the copilot were useless?**
The table stores every approval and rejection with the rule that fired. If supervisors
reject most of what a rule proposes, that rule is wrong. That is the next measurement in
the backlog, and it is the number that would justify keeping or removing the layer.

---

## I. Scope and next steps

**34. What would you do with another month?**
In order of value: collect dark and non-standard hi-vis examples to fix the vest failure;
train 50-60 epochs since the model is still improving; measure the system on real site
footage rather than stock photography; add the review workflow so a supervisor can dismiss a
flagged event and have that recorded in the audit log.

**35. What would you NOT add, and why?**
Face recognition - it would break the privacy design and change the consent and governance
the system would need. Cloud inference - it would undermine the edge claim. An LLM anywhere in
the decision path - the safety decision must stay deterministic and auditable.

**36. What did you learn?**
That separating the statistical model from the deterministic rules is what made the project
testable - and that it costs 0.03 ms. And that most of the honest work in an AI project is
measurement: the temporal filter, the device fallback and the vest failure are all things I
can quantify rather than assert.

**37. Your privacy step had a bug. Talk me through it.**
I claimed the face band sat below the helmet line. `scripts/privacy_utility.py` measured it
and showed helmet detections falling to 33 %, so the masking was destroying the evidence the
snapshot exists to show. I measured where PPE boxes actually sit using the dataset labels
(`scripts/face_band_geometry.py`, no model, no GPU) and found the head is at a different
fraction of the person box depending on the box's shape - 0.15 in a full-body box, 0.60 in a
head crop. The band is now placed by interpolating on aspect ratio, and detected helmet and
vest boxes are restored after pixelation so masking cannot erase them at all. Helmet
preservation went from 33 % to 93 %. Then the *metric* turned out to be wrong too - it
reported 129 % retention for vest, because masking can make the detector invent boxes - so
detections are now paired by overlap and reported as preserved / lost / spurious. Both are
in D-025, including the numbers that embarrassed me.

**38. Which part of the project are you most confident about, and which least?**
Most: the separation between the model and the rules, because it is what made everything
testable and it costs 0.03 ms per frame. Least: the vest class on dark hi-vis jackets, which
fails at 0.07 confidence and is kept in `samples/` as a hard case rather than hidden.
