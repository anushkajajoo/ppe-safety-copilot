"""
Unit tests for the edge decision logic, using fake detections (no camera, no GPU).
These tests ARE the specification: if you change a rule, a test here should change too.
"""
from datetime import datetime
from pathlib import Path

import pytest

from edge.association import associate, body_regions
from edge.compliance import ComplianceEngine
from edge.events import EventBuilder
from edge.geometry import ioa, iou, point_in_polygon
from edge.pipeline import EdgePipeline
from edge.temporal import TemporalFilter
from edge.types import Detection, Worker
from edge.workers import WorkerRegistry
from edge.zones import ZoneManager

ROOT = Path(__file__).resolve().parent.parent
ZONES, POLICIES = ROOT / "configs" / "zones.yaml", ROOT / "configs" / "policies.yaml"
W, H = 1280, 720


# ---------------------------------------------------------------- geometry
def test_point_in_polygon_square_and_concave():
    sq = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert point_in_polygon(5, 5, sq)
    assert not point_in_polygon(15, 5, sq)
    concave = [(0, 0), (10, 0), (10, 10), (5, 5), (0, 10)]   # a "V" cut into the top
    assert point_in_polygon(2, 2, concave)
    assert not point_in_polygon(5, 8, concave)


def test_ioa():
    assert ioa((0, 0, 10, 10), (0, 0, 20, 20)) == 1.0
    assert ioa((0, 0, 10, 10), (5, 0, 20, 20)) == pytest.approx(0.5)
    assert ioa((0, 0, 10, 10), (50, 50, 60, 60)) == 0.0


def test_iou():
    """Used by the privacy evaluation to ask 'is this the same detection?'."""
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (0, 0, 20, 20)) == pytest.approx(0.25)
    assert iou((0, 0, 10, 10), (50, 50, 60, 60)) == 0.0
    assert iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0


# ---------------------------------------------------------------- zones
def test_zones_load_and_priority():
    zm = ZoneManager.from_files(ZONES, POLICIES)
    assert zm.zone_at(880, 400, W, H).zone_id == "welding-a"     # inside welding (priority 2) AND general
    assert zm.zone_at(200, 700, W, H).zone_id == "restricted-c"
    assert zm.zone_at(100, 100, W, H).zone_id == "general-a"
    welding = zm.zone_at(880, 400, W, H)
    assert welding.required_ppe == ["helmet", "vest", "mask"]


def test_zones_scale_with_resolution():
    zm = ZoneManager.from_files(ZONES, POLICIES)
    # same physical spot at 640x360 (half size) must land in the same zone
    assert zm.zone_at(440, 200, 640, 360).zone_id == "welding-a"


def test_restricted_authorization_window():
    zm = ZoneManager.from_files(ZONES, POLICIES)
    rz = next(z for z in zm.zones if z.zone_id == "restricted-c")
    assert not rz.is_authorized_now(datetime(2026, 9, 22, 15, 30))
    rz.authorization_windows = [{"start": "15:00", "end": "16:00"}]
    assert rz.is_authorized_now(datetime(2026, 9, 22, 15, 30))
    assert not rz.is_authorized_now(datetime(2026, 9, 22, 16, 30))


# ---------------------------------------------------------------- association
def person(tid, x1, y1, x2, y2, conf=0.9):
    return Worker(track_id=tid, display_id=f"Worker-{tid:03d}", box=(x1, y1, x2, y2), conf=conf)


def test_helmet_goes_to_the_right_person():
    a, b = person(1, 100, 100, 200, 400), person(2, 300, 100, 400, 400)
    helmet_on_b = Detection("helmet", 0.8, (320, 85, 380, 130))
    vest_on_a = Detection("vest", 0.7, (110, 180, 190, 300))
    associate([a, b], [helmet_on_b, vest_on_a])
    assert "helmet" in b.ppe and "helmet" not in a.ppe
    assert "vest" in a.ppe and "vest" not in b.ppe


def test_helmet_at_feet_is_not_worn():
    a = person(1, 100, 100, 200, 400)
    helmet_held_low = Detection("helmet", 0.9, (120, 360, 180, 400))   # carried by the legs
    _, unassigned = associate([a], [helmet_held_low])
    assert "helmet" not in a.ppe and len(unassigned) == 1


def test_one_helmet_cannot_serve_two_people():
    a, b = person(1, 100, 100, 200, 400), person(2, 160, 100, 260, 400)   # overlapping people
    helmet = Detection("helmet", 0.9, (115, 90, 175, 130))                # mostly over a's head
    associate([a, b], [helmet])
    assert ("helmet" in a.ppe) != ("helmet" in b.ppe)
    assert "helmet" in a.ppe


def test_regions_shape():
    r = body_regions((0, 0, 100, 200))
    assert r["head"][1] < 0 and r["head"][3] == pytest.approx(70)
    assert r["torso"][1] == pytest.approx(40) and r["torso"][3] == pytest.approx(150)


# ---------------------------------------------------------------- tracker ids
def test_worker_ids_sequential_and_forgotten():
    reg = WorkerRegistry(forget_after_s=2.0)
    assert reg.display_id(57, now=0) == "Worker-001"
    assert reg.display_id(99, now=0) == "Worker-002"
    assert reg.display_id(57, now=1) == "Worker-001"     # same track -> same id
    assert reg.forget_stale(now=2.5) == ["Worker-002"]
    assert reg.display_id(99, now=3) == "Worker-003"     # came back -> new id (documented limitation)


# ---------------------------------------------------------------- temporal + compliance
def run_frames(engine, zone, pattern, required_item="mask"):
    """pattern: string like '110011...' where 1 = item seen on that frame."""
    tf = TemporalFilter(engine.window)
    w = person(1, 700, 200, 800, 560)
    d = None
    for ch in pattern:
        present = {"helmet", "vest"} | ({required_item} if ch == "1" else set())
        w.ppe = {k: Detection(k, 0.8, (0, 0, 1, 1)) for k in present}
        h = tf.update(w.display_id, zone.zone_id, present, 0.9)
        d = engine.decide(w, zone, h)
    return d


@pytest.fixture()
def zm():
    return ZoneManager.from_files(ZONES, POLICIES)


def test_one_bad_frame_is_not_a_violation(zm):
    welding = next(z for z in zm.zones if z.zone_id == "welding-a")
    eng = ComplianceEngine(15, 10, 5)
    d = run_frames(eng, welding, "111111101111111")   # mask missed in 1 frame
    assert d.status == "COMPLIANT"


def test_consistent_missing_mask_is_violation(zm):
    welding = next(z for z in zm.zones if z.zone_id == "welding-a")
    eng = ComplianceEngine(15, 10, 5)
    d = run_frames(eng, welding, "110010000000000")   # missing 12 of 15
    assert d.status == "POTENTIAL_VIOLATION"
    assert d.reasons[:1] == ["MISSING_MASK"] and "ZONE_REQUIREMENT_NOT_MET" in d.reasons
    assert d.frames_missing == 12 and d.item_state["mask"] == "MISSING"
    assert 0 < d.confidence <= 1


def test_flickering_evidence_is_uncertain(zm):
    welding = next(z for z in zm.zones if z.zone_id == "welding-a")
    eng = ComplianceEngine(15, 10, 5)
    d = run_frames(eng, welding, "101010101010101")   # missing 7 of 15
    assert d.status == "UNCERTAIN" and d.reasons == ["LOW_CONFIDENCE"]


def test_not_enough_frames_is_uncertain(zm):
    welding = next(z for z in zm.zones if z.zone_id == "welding-a")
    d = run_frames(ComplianceEngine(15, 10, 5), welding, "0000")
    assert d.status == "UNCERTAIN" and d.reasons == ["INSUFFICIENT_EVIDENCE"]


def test_mask_not_required_in_general_zone(zm):
    general = next(z for z in zm.zones if z.zone_id == "general-a")
    d = run_frames(ComplianceEngine(15, 10, 5), general, "000000000000000")
    assert d.status == "COMPLIANT" and d.item_state["mask"] == "N/A"


def test_restricted_zone_entry(zm):
    rz = next(z for z in zm.zones if z.zone_id == "restricted-c")
    d = run_frames(ComplianceEngine(15, 10, 5), rz, "111111111111111")
    assert d.status == "POTENTIAL_VIOLATION" and d.reasons[0] == "RESTRICTED_ZONE_ENTRY"


def test_zone_change_resets_evidence(zm):
    tf = TemporalFilter(15)
    for _ in range(15):
        tf.update("Worker-001", "general-a", {"helmet", "vest"}, 0.9)
    h = tf.update("Worker-001", "welding-a", {"helmet", "vest"}, 0.9)
    assert h.n == 1


def test_baseline_flags_single_bad_frame(zm):
    welding = next(z for z in zm.zones if z.zone_id == "welding-a")
    w = person(1, 700, 200, 800, 560)
    w.ppe = {k: Detection(k, 0.8, (0, 0, 1, 1)) for k in ("helmet", "vest")}
    d = ComplianceEngine(15, 10, 5).decide_baseline(w, welding)
    assert d.status == "POTENTIAL_VIOLATION" and d.reasons == ["MISSING_MASK"]


def test_bad_thresholds_rejected():
    with pytest.raises(ValueError):
        ComplianceEngine(15, 20, 5)


# ---------------------------------------------------------------- full pipeline + events
def frame_dets(with_mask: bool):
    # person standing inside welding-a (foot point ~ (880, 560))
    persons = [Detection("person", 0.9, (830, 250, 930, 560), track_id=7)]
    ppe = [Detection("helmet", 0.85, (850, 235, 910, 280)), Detection("vest", 0.8, (840, 330, 920, 450))]
    if with_mask:
        ppe.append(Detection("mask", 0.7, (865, 285, 895, 310)))
    return persons, ppe


def make_pipe(zm, mode="proposed"):
    b = EventBuilder("EDGE-01", "CAM-01", "test-session", "unit-test", zm.rules_version, cooldown_s=30)
    return EdgePipeline(zm, b, mode=mode)


def test_pipeline_emits_exactly_one_event_for_sustained_violation(zm):
    pipe = make_pipe(zm)
    all_events = []
    for i in range(60):                       # 60 frames without a mask
        _, decisions, events = pipe.step(*frame_dets(with_mask=False), W, H, now=i * 0.04)
        all_events += events
    assert decisions["Worker-001"].zone_id == "welding-a"
    violations = [e for e in all_events if e["status"] == "POTENTIAL_VIOLATION"]
    assert len(violations) == 1
    e = violations[0]
    assert e["worker_display_id"] == "Worker-001" and "MISSING_MASK" in e["reasons"]
    assert {d["object_class"] for d in e["detections"]} == {"person", "helmet", "vest"}


def test_pipeline_no_event_when_compliant(zm):
    pipe = make_pipe(zm)
    events = []
    for i in range(60):
        _, decisions, ev = pipe.step(*frame_dets(with_mask=True), W, H, now=i * 0.04)
        events += ev
    assert decisions["Worker-001"].status == "COMPLIANT"
    assert events == []


def test_baseline_vs_proposed_on_flicker(zm):
    """Mask detector 'blinks' off every 5th frame: baseline raises alerts, proposed stays quiet."""
    base, prop = make_pipe(zm, "baseline"), make_pipe(zm, "proposed")
    base_flips = prop_events = 0
    last = None
    for i in range(100):
        dets = frame_dets(with_mask=(i % 5 != 0))
        _, db, _ = base.step(*dets, W, H, now=i * 0.04)
        _, _, pe = prop.step(*dets, W, H, now=i * 0.04)
        s = db["Worker-001"].status
        base_flips += int(last is not None and s != last)
        last = s
        prop_events += len([e for e in pe if e["status"] == "POTENTIAL_VIOLATION"])
    assert base_flips >= 30          # baseline flips status constantly
    assert prop_events == 0          # proposed: no false violation
