"""
Delivery: how queued events and heartbeats actually reach the server.

The queue itself (edge/outbox.py) knows nothing about HTTP - it takes a `send` callable that
returns True when an item was accepted. This module is that callable, plus the small loop
that drains on a timer and reports the device's own health.

TWO KINDS OF FAILURE, TREATED DIFFERENTLY
    * The link is down, the server is restarting, a timeout - a TRANSIENT failure. The item
      stays queued and is retried with a growing backoff.
    * The server says the payload is wrong (400, 422) - a PERMANENT failure. Retrying a
      malformed item forever is how a queue dies: it blocks nothing (the queue keeps going)
      but it never empties, and it eventually pushes out good events when the cap is hit.
      Such an item is dropped and COUNTED, so it shows up in telemetry rather than silently.
    * 200 and 201 both mean success. 200 is the server saying "I already have this event" -
      the idempotency that makes retrying safe in the first place.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from edge.outbox import EVENT, TELEMETRY, Item, Outbox

TRANSIENT_CODES = {401, 403, 408, 425, 429, 500, 502, 503, 504}
PATHS = {EVENT: "/api/v1/events", TELEMETRY: "/api/v1/telemetry"}


class PermanentRejection(Exception):
    """The server will never accept this payload. Dropping it is the correct outcome."""


@dataclass
class HttpSender:
    """
    Posts one queued item. Returns True if it is done with (stored, or a duplicate).

    `client` is injected so tests never touch a network: anything with a `.post()` works.
    """
    base_url: str
    api_key: str
    client: Optional[object] = None
    timeout: float = 5.0
    rejected: int = 0
    delivered: int = 0
    last_error: Optional[str] = None

    def _client(self):
        if self.client is None:
            import httpx
            self.client = httpx.Client(timeout=self.timeout)
        return self.client

    def __call__(self, item: Item) -> bool:
        url = self.base_url.rstrip("/") + PATHS.get(item.kind, PATHS[EVENT])
        response = self._client().post(url, json=item.body,
                                       headers={"X-API-Key": self.api_key})
        code = getattr(response, "status_code", 0)
        if code in (200, 201):
            self.delivered += 1
            self.last_error = None
            return True
        if code in TRANSIENT_CODES:
            self.last_error = f"HTTP {code}"
            return False                       # keep it, back off, try again
        # 400, 409, 422 and friends: the payload itself is the problem.
        self.rejected += 1
        self.last_error = f"HTTP {code} (dropped, will never be accepted)"
        return True


@dataclass
class SyncWorker:
    """
    Drains the outbox on a timer and queues a heartbeat on a slower one.

    Deliberately not a thread: it is stepped from the frame loop, so there is no concurrency
    to reason about on a device whose main job is to not miss frames. `step()` returns
    quickly and does nothing at all when neither timer is due.
    """
    outbox: Outbox
    send: Callable[[Item], bool]
    device_id: str = "EDGE-01"
    camera_id: Optional[str] = None
    session_id: Optional[str] = None
    drain_every_s: float = 5.0
    heartbeat_every_s: float = 30.0
    model_version: Optional[str] = None
    rules_version: Optional[str] = None
    device_label: Optional[str] = None

    started_at: float = field(default_factory=time.time)
    _last_drain: float = 0.0
    _last_heartbeat: float = 0.0
    frames_processed: int = 0
    last_error: Optional[str] = None

    def heartbeat_body(self, now: float) -> dict:
        stats = self.outbox.stats(now=now)
        elapsed = max(now - self.started_at, 1e-6)
        return {"device_id": self.device_id, "camera_id": self.camera_id,
                "session_id": self.session_id,
                "fps": round(self.frames_processed / elapsed, 2),
                "frames_processed": self.frames_processed,
                "queue_depth": int(stats["depth"]), "queue_dropped": int(stats["dropped"]),
                "uptime_s": round(elapsed, 1), "model_version": self.model_version,
                "rules_version": self.rules_version, "device_label": self.device_label,
                "last_error": self.last_error}

    def step(self, now: Optional[float] = None, frames: int = 0) -> dict:
        """Call once per frame. Returns what it did, for the on-screen status line."""
        now = time.time() if now is None else now
        self.frames_processed += frames
        did = {"heartbeat": False, "sent": 0, "failed": 0}

        if now - self._last_heartbeat >= self.heartbeat_every_s:
            self._last_heartbeat = now
            # queued, not posted: a heartbeat during an outage must survive the outage
            self.outbox.add(self.heartbeat_body(now), kind=TELEMETRY,
                            item_id=f"hb-{self.device_id}-{int(now)}", now=now)
            did["heartbeat"] = True

        if now - self._last_drain >= self.drain_every_s:
            self._last_drain = now
            result = self.outbox.drain(self.send, now=now)
            did["sent"], did["failed"] = result.sent, result.failed
            if result.errors:
                self.last_error = result.errors[-1][:250]
        return did
