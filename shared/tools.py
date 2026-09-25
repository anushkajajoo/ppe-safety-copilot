"""
The tool allow-list: everything the copilot layer is permitted to call, and nothing else.

WHY A REGISTRY AND NOT JUST FUNCTIONS
    The capstone brief requires that an assistant layer cannot reach arbitrary capability -
    no shell, no filesystem, no code execution, no ad-hoc API calls. The way to make that
    checkable rather than promised is to put every callable behind one table and one gate,
    so "what can this thing do?" is answered by reading a list instead of auditing a codebase.

    Anything not in TOOLS is DENIED, and the denial is logged with what was asked for, when,
    and why it was refused. A denial is evidence, not just a rejection.

READ AND WRITE ARE NOT THE SAME THING
    Each tool declares `writes`. Read-only tools (camera status, latest detection, compliance
    figures) can run for anyone signed in. Tools that change something - creating an event,
    saving a snapshot, acknowledging an alert - are marked, and the API requires a supervisor
    for those. Nothing here executes a safety DECISION: the rule engine does that
    (shared/decision.py), and these tools only report or record what it already decided.

WHAT IS DELIBERATELY ABSENT
    There is no run_command, no read_file, no write_file, no query_database, no http_request.
    Their absence is the point, and `test_agent_safety.py` asserts it stays that way.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

ALLOWED, DENIED = "ALLOWED", "DENIED"


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    writes: bool = False          # True -> needs a supervisor, and is recorded


TOOLS: Dict[str, Tool] = {t.name: t for t in (
    Tool("get_camera_status", "Is a camera running, at what frame rate, with what verdict."),
    Tool("get_latest_detection", "The newest frame's people, statuses and decision."),
    Tool("get_compliance_status", "Compliance counts and percentage from recorded events."),
    Tool("get_system_health", "Frame rate, latency, memory, queue depth, storage."),
    Tool("list_recent_events", "The most recent violation-log entries."),
    Tool("create_event", "Record a safety event from the current verdict.", writes=True),
    Tool("save_snapshot", "Store a face-masked snapshot of the current frame.", writes=True),
    Tool("save_recording", "Write the buffered evidence clip for an event.", writes=True),
    Tool("acknowledge_alert", "Mark an alert as seen by a named person.", writes=True),
    Tool("generate_report", "Produce a compliance summary for a period.", writes=True),
)}


class ToolDenied(PermissionError):
    """The tool is not on the allow-list, or the caller may not use it."""


@dataclass
class ToolCall:
    name: str
    outcome: str
    reason: str
    actor: Optional[str] = None
    at: str = ""

    def as_dict(self) -> dict:
        return {"tool": self.name, "outcome": self.outcome, "reason": self.reason,
                "actor": self.actor, "at": self.at}


class ToolLog:
    """
    Append-only record of every tool request - allowed and denied alike.

    Denials are the interesting half: a system that only logs what it did cannot show what it
    refused, and refusing is most of what this layer is for.
    """

    def __init__(self, path: Path, limit: int = 2000) -> None:
        self.path = Path(path)
        self.limit = limit
        self._lock = threading.Lock()

    def record(self, name: str, outcome: str, reason: str,
               actor: Optional[str] = None) -> ToolCall:
        call = ToolCall(name=name, outcome=outcome, reason=reason, actor=actor,
                        at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(call.as_dict()) + "\n")
        except OSError:
            pass                    # logging must never break the request it is describing
        return call

    def recent(self, limit: int = 50) -> List[dict]:
        if not self.path.exists():
            return []
        rows: List[dict] = []
        for line in self.path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return list(reversed(rows))[:max(1, limit)]

    def denials(self, limit: int = 50) -> List[dict]:
        return [row for row in self.recent(limit=self.limit) if row.get("outcome") == DENIED][:limit]


def check_tool(name: str, can_write: bool = False) -> Tool:
    """
    The gate. Raises ToolDenied with the reason; callers log it either way.

    `can_write` is the caller's authority, not the tool's: a viewer asking for a write tool
    is denied by role, an unknown name is denied by the allow-list.
    """
    tool = TOOLS.get(name)
    if tool is None:
        raise ToolDenied(f"'{name}' is not on the tool allow-list")
    if tool.writes and not can_write:
        raise ToolDenied(f"'{name}' changes state and needs the supervisor role")
    return tool


def catalog() -> List[dict]:
    return [{"name": t.name, "description": t.description, "writes": t.writes}
            for t in TOOLS.values()]
