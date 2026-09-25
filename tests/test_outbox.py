"""
Tests for the store-and-forward outbox and device telemetry - the project's answer to
"what happens when the network is not there?".

Nothing here touches a network: `send` is a callable, so an outage is just a function that
returns False.
"""
import json

import pytest
from fastapi.testclient import TestClient

from edge.outbox import (EVENT, MAX_BACKOFF_S, TELEMETRY, Outbox, backoff_for)
from server.config import Settings
from server.main import create_app

KEY = "telemetry-key"


def event(n=1):
    return {"event_id": f"evt-{n:04d}", "status": "POTENTIAL_VIOLATION", "camera_id": "CAM-01"}


@pytest.fixture()
def outbox(tmp_path):
    return Outbox(tmp_path / "outbox.jsonl")


# ------------------------------------------------------------------- queueing
def test_an_event_is_queued_the_moment_it_is_made(outbox):
    outbox.add(event(1))
    assert outbox.depth() == 1
    assert outbox.pending()[0].body["event_id"] == "evt-0001"


def test_the_queue_survives_a_restart(tmp_path):
    """The file IS the queue - that is the whole point of writing it down immediately."""
    first = Outbox(tmp_path / "q.jsonl")
    first.add(event(1))
    first.add(event(2))

    reopened = Outbox(tmp_path / "q.jsonl")          # as if the device had rebooted
    assert [item.body["event_id"] for item in reopened.pending()] == ["evt-0001", "evt-0002"]


def test_queueing_the_same_event_twice_does_nothing(outbox):
    outbox.add(event(1))
    outbox.add(event(1))
    assert outbox.depth() == 1


def test_a_torn_line_is_skipped_rather_than_fatal(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text(json.dumps({"kind": EVENT, "body": event(1), "item_id": "a",
                                "queued_at": 1.0}) + "\n{ this line was cut off by a power c",
                    encoding="utf-8")
    assert Outbox(path).depth() == 1


def test_the_queue_is_bounded_and_says_what_it_dropped(tmp_path):
    """A full disk takes the whole device down; the newest observation is the one to keep."""
    small = Outbox(tmp_path / "q.jsonl", max_items=3)
    for n in range(5):
        small.add(event(n))
    assert small.depth() == 3
    assert small.dropped == 2
    assert [item.body["event_id"] for item in small.pending()] == ["evt-0002", "evt-0003", "evt-0004"]


# -------------------------------------------------------------------- sending
def test_a_successful_drain_empties_the_queue(outbox):
    outbox.add(event(1))
    outbox.add(event(2))
    result = outbox.drain(lambda item: True, now=100.0)
    assert result.sent == 2 and result.ok
    assert outbox.depth() == 0


def test_an_outage_keeps_everything_and_backs_off(outbox):
    outbox.add(event(1))
    result = outbox.drain(lambda item: False, now=100.0)
    assert result.failed == 1 and not result.ok
    [item] = outbox.pending()
    assert item.attempts == 1
    assert item.next_attempt == 100.0 + backoff_for(1)       # 2 seconds later


def test_an_exception_from_the_sender_is_an_outage_not_a_crash(outbox):
    def dead_link(item):
        raise ConnectionError("connection refused")

    outbox.add(event(1))
    result = outbox.drain(dead_link, now=100.0)
    assert result.failed == 1
    assert "ConnectionError" in outbox.pending()[0].last_error


def test_nothing_is_retried_before_its_backoff_expires(outbox):
    outbox.add(event(1))
    outbox.drain(lambda item: False, now=100.0)          # fails, next attempt at 102
    attempted = []
    result = outbox.drain(lambda item: attempted.append(item) or True, now=101.0)
    assert attempted == [] and result.sent == 0 and result.skipped == 1
    result = outbox.drain(lambda item: True, now=103.0)  # now it is due
    assert result.sent == 1


def test_backoff_grows_and_then_stops_growing():
    """A device back from an hour offline must not hammer the server - nor wait an hour."""
    assert backoff_for(0) == 0.0
    assert backoff_for(1) == 2.0
    assert backoff_for(2) == 4.0
    assert backoff_for(3) == 8.0
    assert backoff_for(50) == MAX_BACKOFF_S


def test_one_stuck_item_does_not_block_the_ones_behind_it(outbox):
    outbox.add(event(1))
    outbox.add(event(2))

    def only_the_second(item):
        return item.body["event_id"] == "evt-0002"

    result = outbox.drain(only_the_second, now=100.0)
    assert result.sent == 1 and result.failed == 1
    assert [item.body["event_id"] for item in outbox.pending()] == ["evt-0001"]


def test_order_is_preserved(outbox):
    for n in range(1, 6):
        outbox.add(event(n))
    seen = []
    outbox.drain(lambda item: seen.append(item.body["event_id"]) or True, now=100.0)
    assert seen == ["evt-0001", "evt-0002", "evt-0003", "evt-0004", "evt-0005"]


def test_stats_describe_the_queue_for_telemetry(outbox):
    outbox.add(event(1), now=100.0)
    outbox.drain(lambda item: False, now=100.0)
    stats = outbox.stats(now=160.0)
    assert stats["depth"] == 1 and stats["retrying"] == 1
    assert stats["oldest_age_s"] == 60.0


# ------------------------------------------------------- telemetry rides along
def test_telemetry_is_queued_like_anything_else(outbox):
    outbox.add({"device_id": "EDGE-01", "fps": 21.4}, kind=TELEMETRY)
    assert outbox.pending()[0].kind == TELEMETRY


@pytest.fixture()
def client(tmp_path):
    s = Settings(database_url=f"sqlite:///{(tmp_path / 't.db').as_posix()}",
                 evidence_dir=tmp_path / "ev", edge_api_key=KEY,
                 violation_log=tmp_path / "violations.jsonl")
    with TestClient(create_app(s)) as c:
        yield c


def test_a_heartbeat_needs_the_api_key(client):
    r = client.post("/api/v1/telemetry", json={"device_id": "EDGE-01"})
    assert r.status_code == 401


def test_a_heartbeat_is_stored_and_read_back(client):
    r = client.post("/api/v1/telemetry", headers={"X-API-Key": KEY},
                    json={"device_id": "EDGE-01", "camera_id": "CAM-01", "fps": 21.4,
                          "queue_depth": 7, "queue_dropped": 2, "device_label": "cpu"})
    assert r.status_code == 201

    body = client.get("/api/v1/telemetry").json()
    assert body["count"] == 1
    beat = body["telemetry"][0]
    assert beat["device_id"] == "EDGE-01" and beat["queue_depth"] == 7
    assert beat["stale"] is False and beat["age_s"] is not None


def test_a_device_that_stops_reporting_is_marked_stale(client):
    client.post("/api/v1/telemetry", headers={"X-API-Key": KEY},
                json={"device_id": "EDGE-01", "occurred_at": "2020-01-01T00:00:00Z"})
    beat = client.get("/api/v1/telemetry").json()["telemetry"][0]
    assert beat["stale"] is True, "an old heartbeat must not look healthy"


def test_a_heartbeat_carries_no_personal_data(client):
    """Health only: the payload has no field that could hold a frame or a worker label."""
    from server.routes.telemetry import Heartbeat
    fields = set(Heartbeat.model_fields)
    assert not (fields & {"frame", "image", "detections", "worker_id", "person_id", "snapshot"})


# --------------------------------------------------------- delivery (edge/sync.py)
class FakeResponse:
    def __init__(self, status_code):
        self.status_code = status_code


class FakeClient:
    """Stands in for httpx: records the posts, returns whatever the test wants."""
    def __init__(self, codes):
        self.codes = list(codes)
        self.posts = []

    def post(self, url, json=None, headers=None):
        self.posts.append({"url": url, "json": json, "headers": headers})
        return FakeResponse(self.codes.pop(0) if self.codes else 201)


def sender(codes, **kw):
    from edge.sync import HttpSender
    client = FakeClient(codes)
    return HttpSender(base_url="http://localhost:8000", api_key=KEY, client=client, **kw), client


def test_a_stored_event_counts_as_delivered(outbox):
    send, client = sender([201])
    outbox.add(event(1))
    assert outbox.drain(send, now=1.0).sent == 1
    assert client.posts[0]["url"].endswith("/api/v1/events")
    assert client.posts[0]["headers"]["X-API-Key"] == KEY


def test_a_duplicate_also_counts_as_delivered(outbox):
    """200 = 'I already have this event'. That idempotency is what makes retrying safe."""
    send, _ = sender([200])
    outbox.add(event(1))
    assert outbox.drain(send, now=1.0).sent == 1
    assert outbox.depth() == 0


def test_a_server_that_is_down_keeps_the_event(outbox):
    send, _ = sender([503])
    outbox.add(event(1))
    assert outbox.drain(send, now=1.0).failed == 1
    assert outbox.depth() == 1


def test_a_payload_the_server_will_never_accept_is_dropped_and_counted(outbox):
    """Retrying a poison message forever is how a queue dies - and it evicts good events."""
    send, _ = sender([422])
    outbox.add(event(1))
    outbox.drain(send, now=1.0)
    assert outbox.depth() == 0
    assert send.rejected == 1 and "dropped" in send.last_error


def test_telemetry_goes_to_the_telemetry_endpoint(outbox):
    from edge.outbox import TELEMETRY as T
    send, client = sender([201])
    outbox.add({"device_id": "EDGE-01"}, kind=T)
    outbox.drain(send, now=1.0)
    assert client.posts[0]["url"].endswith("/api/v1/telemetry")


# ------------------------------------------------------------------ the worker
def make_worker(outbox, send):
    from edge.sync import SyncWorker
    return SyncWorker(outbox=outbox, send=send, device_id="EDGE-01",
                      drain_every_s=5.0, heartbeat_every_s=30.0, started_at=0.0)


def test_the_worker_queues_a_heartbeat_rather_than_posting_it(outbox):
    """A heartbeat raised during an outage must survive the outage and arrive after it."""
    send, _ = sender([503] * 10)
    worker = make_worker(outbox, send)
    worker.step(now=30.0, frames=300)
    beats = [item for item in outbox.pending() if item.kind == TELEMETRY]
    assert len(beats) == 1
    assert beats[0].body["device_id"] == "EDGE-01"
    assert beats[0].body["fps"] == 10.0            # 300 frames in 30 s


def test_the_heartbeat_reports_the_queue_it_could_not_send(outbox):
    send, _ = sender([503] * 10)
    outbox.add(event(1))
    outbox.drain(send, now=1.0)                    # one failed event sitting in the queue
    worker = make_worker(outbox, send)
    body = worker.heartbeat_body(now=60.0)
    assert body["queue_depth"] == 1
    assert body["uptime_s"] == 60.0


def test_the_worker_does_nothing_between_timers(outbox):
    send, client = sender([201] * 10)
    worker = make_worker(outbox, send)
    worker.step(now=0.5, frames=1)
    assert client.posts == [] and outbox.depth() == 0


def test_everything_queued_during_an_outage_arrives_when_it_ends(outbox):
    """The end-to-end promise of the whole module, in one test."""
    down, _ = sender([503] * 20)
    worker = make_worker(outbox, down)
    for n in range(1, 4):
        outbox.add(event(n))
    worker.step(now=10.0, frames=100)              # tries, fails, keeps everything
    assert outbox.depth() >= 3

    up, client = sender([201] * 20)
    worker.send = up
    worker.step(now=400.0, frames=100)             # well past every backoff window
    assert outbox.depth() == 0
    sent_ids = [post["json"].get("event_id") for post in client.posts if "event_id" in post["json"]]
    assert sent_ids == ["evt-0001", "evt-0002", "evt-0003"]
