"""
Store-and-forward outbox: what the edge does when the network is not there.

THE PROBLEM
    A site camera loses connectivity - a dropped Wi-Fi link, a server restart, a van parked
    in front of the antenna. A safety event that happens during those minutes is exactly the
    kind that must not be lost, and a supervisor must not have to know that the link was down.

THE DESIGN, AND WHY IT IS THIS SMALL
    Events are appended to a file the moment they are created, then drained to the server
    when it answers. The file is the queue. That gives durability across a process restart
    (and a power cut) for the price of one `open(..., "a")`, with no broker, no daemon and no
    dependency - which matters on a device whose whole job is to run unattended.

    Retry is exponential with a cap: 2s, 4s, 8s ... 300s. A device that has been offline for
    an hour must not hammer the server the instant it returns, and must not wait an hour
    either.

WHY RETRYING IS SAFE
    `event_id` is generated on the edge and is the server's primary key, so re-sending an
    event that actually arrived (but whose response was lost) is a no-op: the API answers
    200 with `duplicate=true` instead of storing it twice. Without that, a retry queue would
    quietly corrupt the record every time a connection dropped mid-request - the failure
    mode that makes naive retry worse than no retry.

BOUNDED BY DESIGN
    The queue is capped (`max_items`). When it is full the OLDEST unsent item is dropped and
    counted, because a disk that fills up takes the whole device down, and in a safety system
    the newest observation is the one worth keeping. The drop count is reported in telemetry,
    so "we lost events" is visible rather than silent.

TELEMETRY RIDES THE SAME PIPE
    Heartbeats (queue depth, frame rate, last error) are queued and drained exactly like
    events, so the fact that a device was offline is itself reported once it reconnects.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

EVENT = "event"
TELEMETRY = "telemetry"

FIRST_BACKOFF_S = 2.0
MAX_BACKOFF_S = 300.0
MAX_ITEMS = 5000


@dataclass
class Item:
    """One queued thing. `attempts` and `next_attempt` are what make the retry polite."""
    kind: str
    body: dict
    item_id: str
    queued_at: float
    attempts: int = 0
    next_attempt: float = 0.0
    last_error: Optional[str] = None

    def as_dict(self) -> dict:
        return {"kind": self.kind, "body": self.body, "item_id": self.item_id,
                "queued_at": self.queued_at, "attempts": self.attempts,
                "next_attempt": self.next_attempt, "last_error": self.last_error}

    @staticmethod
    def from_dict(raw: dict) -> "Item":
        return Item(kind=raw.get("kind", EVENT), body=raw.get("body", {}),
                    item_id=str(raw.get("item_id", "")), queued_at=float(raw.get("queued_at", 0.0)),
                    attempts=int(raw.get("attempts", 0)),
                    next_attempt=float(raw.get("next_attempt", 0.0)),
                    last_error=raw.get("last_error"))


def backoff_for(attempts: int) -> float:
    """2, 4, 8, ... capped at 300 seconds. Deterministic, so it can be tested."""
    if attempts <= 0:
        return 0.0
    return min(FIRST_BACKOFF_S * (2 ** (attempts - 1)), MAX_BACKOFF_S)


@dataclass
class DrainResult:
    sent: int = 0
    failed: int = 0
    skipped: int = 0            # not due yet - still inside its backoff window
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.failed == 0


class Outbox:
    """
    A durable queue in one file. Not a message broker - deliberately.

    The whole file is rewritten after a drain. At the sizes this queue lives at (thousands of
    small JSON lines, drained every few seconds) that costs microseconds and buys code that
    can be read in one sitting, which is worth more here than an append-only log with
    compaction.
    """

    def __init__(self, path: Path, max_items: int = MAX_ITEMS) -> None:
        self.path = Path(path)
        self.max_items = int(max_items)
        self.dropped = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ storage
    def _read(self) -> List[Item]:
        if not self.path.exists():
            return []
        items: List[Item] = []
        for line in self.path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                items.append(Item.from_dict(json.loads(line)))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue          # a torn line from a power cut is skipped, never fatal
        return items

    def _write(self, items: List[Item]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = "".join(json.dumps(item.as_dict()) + "\n" for item in items)
        # Write beside, then replace: a crash mid-write cannot leave a half queue.
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(self.path)

    # ------------------------------------------------------------------- adding
    def add(self, body: dict, kind: str = EVENT, item_id: Optional[str] = None,
            now: Optional[float] = None) -> Item:
        now = time.time() if now is None else now
        identifier = item_id or str(body.get("event_id") or f"{kind}-{now:.6f}")
        item = Item(kind=kind, body=body, item_id=identifier, queued_at=now)
        with self._lock:
            items = self._read()
            if any(existing.item_id == identifier for existing in items):
                return item                        # already queued; queueing twice helps nobody
            items.append(item)
            if len(items) > self.max_items:
                overflow = len(items) - self.max_items
                items = items[overflow:]           # drop the OLDEST, keep the newest
                self.dropped += overflow
            self._write(items)
        return item

    # ------------------------------------------------------------------ reading
    def pending(self) -> List[Item]:
        with self._lock:
            return self._read()

    def depth(self) -> int:
        return len(self.pending())

    # ----------------------------------------------------------------- draining
    def drain(self, send: Callable[[Item], bool], now: Optional[float] = None,
              limit: int = 100) -> DrainResult:
        """
        Try to deliver what is due. `send` returns True when the server accepted the item.

        Order is preserved: items are attempted oldest first, and one that fails stays in
        the queue with a longer backoff rather than blocking the ones behind it.
        """
        now = time.time() if now is None else now
        result = DrainResult()
        with self._lock:
            items = self._read()
            keep: List[Item] = []
            attempted = 0
            for item in items:
                if attempted >= limit or item.next_attempt > now:
                    result.skipped += 0 if attempted >= limit else 1
                    keep.append(item)
                    continue
                attempted += 1
                try:
                    delivered = bool(send(item))
                    error = None
                except Exception as exc:                       # a dead link, not a bug
                    delivered, error = False, f"{type(exc).__name__}: {exc}"
                if delivered:
                    result.sent += 1
                    continue
                item.attempts += 1
                item.last_error = error or "rejected by the server"
                item.next_attempt = now + backoff_for(item.attempts)
                result.failed += 1
                if error:
                    result.errors.append(error)
                keep.append(item)
            self._write(keep)
        return result

    def stats(self, now: Optional[float] = None) -> Dict[str, object]:
        """What telemetry reports about the queue itself."""
        now = time.time() if now is None else now
        items = self.pending()
        oldest = min((item.queued_at for item in items), default=None)
        return {"depth": len(items), "dropped": self.dropped,
                "oldest_age_s": round(now - oldest, 1) if oldest else 0.0,
                "retrying": sum(1 for item in items if item.attempts > 0),
                "max_items": self.max_items}
