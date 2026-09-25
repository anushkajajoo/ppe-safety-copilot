"""
Compliance analytics: counted from what actually happened, never from example numbers.

WHY A SEPARATE LOG
    `violations.jsonl` records violations only, which is the right thing for an evidence
    trail and the wrong thing for a compliance rate: you cannot divide by a denominator you
    never wrote down. This file records one line per EVALUATION - including the ones that
    came out clean - so "compliance %" has a real numerator and a real denominator.

    It stays small: one line per uploaded image or analysed clip, and for a live camera at
    most one line every `min_gap_s` seconds per source, plus one whenever the decision
    changes. A 10 FPS camera therefore produces a handful of lines a minute, not six hundred.

WHAT IS STORED
    Time, source, the decision (GO / REVIEW / STOP), how many people were in frame, how many
    were non-compliant, and which required items were missing. No identity, no image, no box
    coordinates - the evidence lives elsewhere and this is the counting layer.
"""
from __future__ import annotations

import json
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

GO, REVIEW, STOP = "GO", "REVIEW", "STOP"


class DecisionLog:
    def __init__(self, path: Path, min_gap_s: float = 5.0) -> None:
        self.path = Path(path)
        self.min_gap_s = float(min_gap_s)
        self._lock = threading.Lock()
        self._last: Dict[str, tuple] = {}        # source -> (when, decision)

    # ------------------------------------------------------------------ writing
    def record(self, source: str, decision: str, people: int = 0, violations: int = 0,
               missing: Sequence[str] = (), required: Sequence[str] = (),
               throttle: bool = True, now: Optional[float] = None,
               event_id: Optional[str] = None) -> bool:
        """
        Append one evaluation. Returns True if a line was written.

        Throttling keeps a live camera from flooding the file, but a CHANGE of decision is
        always written: the moment compliance turns into a violation is the one a supervisor
        will look for, and it must not be the line that got skipped.
        """
        now = now if now is not None else datetime.now(timezone.utc).timestamp()
        if throttle:
            with self._lock:
                previous = self._last.get(source)
                if previous is not None:
                    when, last_decision = previous
                    if decision == last_decision and (now - when) < self.min_gap_s:
                        return False
                self._last[source] = (now, decision)

        record = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "source": source, "decision": decision, "people": int(people),
                  "violations": int(violations), "missing": list(missing),
                  "required": list(required), "event_id": event_id}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
        except OSError:
            return False
        return True

    # ------------------------------------------------------------------ reading
    def rows(self, limit: Optional[int] = None) -> List[dict]:
        if not self.path.exists():
            return []
        out: List[dict] = []
        for line in self.path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue                      # a torn line is skipped, never fatal
        return out[-limit:] if limit else out

    def summary(self, limit: Optional[int] = None) -> Dict[str, object]:
        """
        Compliance figures computed from the rows. Every number here is counted; none is
        assumed. With no data the rate is None rather than 0 %, because "no evaluations yet"
        and "nothing was compliant" are different statements.
        """
        rows = self.rows(limit)
        decisions = Counter(row.get("decision", REVIEW) for row in rows)
        missing = Counter()
        for row in rows:
            for item in row.get("missing", []):
                missing[item] += 1

        total = len(rows)
        compliant = decisions.get(GO, 0)
        rate = round(compliant / total * 100, 1) if total else None
        return {
            "total_evaluations": total,
            "compliant": compliant,
            "violations": decisions.get(STOP, 0),
            "review": decisions.get(REVIEW, 0),
            "compliance_pct": rate,
            "by_item": {"helmet": missing.get("helmet", 0),
                        "vest": missing.get("vest", 0),
                        "mask": missing.get("mask", 0)},
            "people_seen": sum(int(row.get("people", 0)) for row in rows),
            "sources": sorted({row.get("source", "") for row in rows if row.get("source")}),
            "first": rows[0]["at"] if rows else None,
            "last": rows[-1]["at"] if rows else None,
        }
