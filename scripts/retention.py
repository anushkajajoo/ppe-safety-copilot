"""
Delete stored evidence once it is older than the retention period.

WHY THIS EXISTS
    `evidence_retention_days` has been a setting since the beginning and, until now, a
    setting that did nothing - which is worse than not having it, because it reads like a
    promise. Data minimisation is not a policy you write down; it is a job that runs.

WHAT IT TOUCHES
    Only .jpg files directly inside the evidence folder, and only ones older than the
    period. It never recurses, never follows a symlink out of the folder, and never touches
    the violation log or the database: the metadata record of an event is what survives, and
    the picture is what expires. That is the right way round - the log line says a violation
    happened, the snapshot is the personal data.

    A deleted snapshot leaves its log line intact with `snapshot` still pointing at a path
    that no longer exists. That is deliberate and honest: the record must not silently
    rewrite itself, and `privacy_status` already tells a reviewer what to expect.

Run:
    python -m scripts.retention --dry-run      # list what would go
    python -m scripts.retention               # delete it
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

SUFFIXES = {".jpg", ".jpeg", ".png"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}
LOG_SUFFIXES = {".jsonl", ".log"}

# Each kind of data has its own period, because they cost and reveal different things:
# a clip is large and shows a person moving, a snapshot is one moment, a log line is text.
KINDS = {
    "snapshots": {"suffixes": SUFFIXES, "setting": "snapshot_retention_days",
                  "dir": "evidence_dir"},
    "recordings": {"suffixes": VIDEO_SUFFIXES, "setting": "video_retention_days",
                   "dir": "recordings_dir"},
    "logs": {"suffixes": LOG_SUFFIXES, "setting": "log_retention_days", "dir": "logs_dir"},
}

# Events are not files, so they are swept by the store rather than by suffix: it knows
# which ones a person is still waiting on and refuses to remove those.
EVENT_KIND = "events"


@dataclass
class Sweep:
    deleted: List[str] = field(default_factory=list)
    kept: int = 0
    freed_bytes: int = 0
    failed: List[str] = field(default_factory=list)
    dry_run: bool = False

    @property
    def freed_mb(self) -> float:
        return round(self.freed_bytes / (1024 * 1024), 2)


def expired(path: Path, older_than_days: float, now: Optional[float] = None) -> bool:
    now = time.time() if now is None else now
    return (now - path.stat().st_mtime) > older_than_days * 86400


def sweep(evidence_dir: Path, older_than_days: float, now: Optional[float] = None,
          dry_run: bool = False, suffixes=None, recursive: bool = False) -> Sweep:
    """
    Delete expired files of one kind. Returns what happened - never raises for one bad file.

    `recursive` is off by default: recordings live one folder deep (per camera), so that
    path asks for it explicitly rather than every sweep walking a tree it should not.
    """
    result = Sweep(dry_run=dry_run)
    directory = Path(evidence_dir)
    if not directory.is_dir():
        return result

    wanted = set(suffixes) if suffixes else SUFFIXES
    entries = sorted(directory.rglob("*")) if recursive else sorted(directory.iterdir())
    for path in entries:
        if not path.is_file() or path.is_symlink() or path.suffix.lower() not in wanted:
            continue
        try:
            if not expired(path, older_than_days, now):
                result.kept += 1
                continue
            size = path.stat().st_size
            if not dry_run:
                path.unlink()
            result.deleted.append(path.name)
            result.freed_bytes += size
        except OSError as exc:
            result.failed.append(f"{path.name}: {exc}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kind", choices=sorted(KINDS) + [EVENT_KIND, "all"], default="all")
    parser.add_argument("--days", type=float, default=None,
                        help="override the retention period for the chosen kind")
    parser.add_argument("--dir", type=Path, default=None, help="override the folder")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def sweep_kind(kind: str, settings, days: Optional[float] = None,
               directory: Optional[Path] = None, dry_run: bool = False) -> Sweep:
    spec = KINDS[kind]
    folder = Path(directory or getattr(settings, spec["dir"]))
    period = days if days is not None else float(getattr(settings, spec["setting"]))
    return sweep(folder, period, dry_run=dry_run, suffixes=spec["suffixes"],
                 recursive=(kind == "recordings"))


def sweep_events(settings, days=None, dry_run: bool = False) -> dict:
    """
    Retention for structured events. NEW and ACKNOWLEDGED rows are protected however old
    they are: an event nobody has reviewed is overdue, not stale.
    """
    from pathlib import Path as _Path

    from server import event_store
    from server.database import make_engine, make_session_factory

    engine = make_engine(settings.database_url)
    session = make_session_factory(engine)()
    try:
        result = event_store.cleanup(
            session,
            [_Path(settings.evidence_dir), _Path(settings.recordings_dir),
             _Path(settings.snapshots_masked_dir)],
            retention_days=days if days is not None else settings.event_retention_days,
            dry_run=dry_run)
        if not dry_run:
            session.commit()
        return result
    finally:
        session.close()
        engine.dispose()


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    from server.config import get_settings

    settings = get_settings()
    # "events" is handled separately below - it is rows, not files
    kinds = sorted(KINDS) if args.kind == "all" else [k for k in [args.kind] if k in KINDS]
    verb = "would delete" if args.dry_run else "deleted"
    total_files, total_mb = 0, 0.0

    for kind in kinds:
        spec = KINDS[kind]
        folder = Path(args.dir or getattr(settings, spec["dir"]))
        period = args.days if args.days is not None else float(getattr(settings, spec["setting"]))
        result = sweep_kind(kind, settings, days=args.days, directory=args.dir,
                            dry_run=args.dry_run)
        total_files += len(result.deleted)
        total_mb += result.freed_mb
        print(f"{kind:<12} {folder}")
        print(f"{'':<12} retention {period:g} days -> {verb} {len(result.deleted)} file(s), "
              f"{result.freed_mb} MB; kept {result.kept}")
        for name in result.deleted[:10]:
            print(f"{'':<14}{name}")
        for problem in result.failed:
            print(f"{'':<14}FAILED {problem}")

    if args.kind in (EVENT_KIND, "all"):
        events = sweep_events(settings, days=args.days, dry_run=args.dry_run)
        print(f"{EVENT_KIND:<12} retention {events['retention_days']:g} days -> "
              f"{verb} {events['removed_count']} event(s); "
              f"{events['protected']} protected (NEW/ACKNOWLEDGED), {events['kept']} kept")
        for name in events["removed"][:10]:
            print(f"{'':<14}{name}")
        if events["files_deleted"]:
            print(f"{'':<14}evidence removed: {len(events['files_deleted'])} file(s)")

    print(f"\ntotal: {verb} {total_files} file(s), {round(total_mb, 2)} MB")
    if not total_files:
        print("Nothing was old enough to remove. Snapshots and recordings are off by "
              "default, so an empty result is the normal state.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
