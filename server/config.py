"""
Server settings, read from environment variables or the .env file.

WHY: secrets and paths must NOT be hard-coded (and must never be committed to
GitHub). Copy .env.example -> .env and edit values there.
"""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The web pages: plain HTML + CSS + JS, no build step, served by FastAPI.
FRONTEND_DIR = PROJECT_ROOT / "frontend"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", env_prefix="PPE_", extra="ignore")

    api_version: str = "0.1.0"
    # SQLite file inside ./data (gitignored). Absolute path so it works from any folder.
    database_url: str = f"sqlite:///{(PROJECT_ROOT / 'data' / 'server.db').as_posix()}"
    evidence_dir: Path = PROJECT_ROOT / "data" / "snapshots" / "masked"
    evidence_retention_days: int = 30      # kept as the alias of snapshot_retention_days
    # Used from Day 3 for edge -> server authentication. Never commit the real value.
    edge_api_key: str = "change-me"
    sql_echo: bool = False

    # Detector used by POST /api/v1/predict (override with PPE_WEIGHTS, PPE_DETECT_CONF, ...)
    weights: Path = PROJECT_ROOT / "models" / "ppe4_yolo26n_best.pt"
    detect_conf: float = 0.35
    detect_imgsz: int = 640
    # "auto" = use the GPU only if it has enough free VRAM right now, else the CPU
    # (shared/device.py). "cpu" or "0" force the choice. See docs/decisions.md D-022a.
    detect_device: str = "auto"

    # Local violation log (Phase 12). Plain JSONL: metadata only, never images.
    violation_log: Path = PROJECT_ROOT / "data" / "violations.jsonl"
    violation_cooldown_s: float = 30.0   # same person+status logged at most this often

    # Sign-in (server/auth.py). users_file holds PBKDF2 hashes - never plaintext.
    # session_secret empty = a fresh random secret each start, so no secret is ever
    # committed; the cost is signing in again after a restart. See D-027.
    users_file: Path = PROJECT_ROOT / "configs" / "users.yaml"
    session_secret: str = ""

    # ---- the safety decision layer (shared/decision.py) ----------------------
    # The threshold a PPE item must clear to count as present. Separate from
    # detect_conf, which is the detector's own cut-off: this one is site POLICY and
    # is the number quoted in a decision's reason.
    ppe_confidence_threshold: float = 0.50
    # Between (threshold - review_margin) and threshold an item is UNCERTAIN, which
    # produces REVIEW rather than STOP. Set to 0 to remove the band entirely.
    review_margin: float = 0.15
    # What the system answers when it cannot see (no model, no frame, detector error).
    # "REVIEW" or "STOP" only - "GO" is refused by shared.decision.validate_fail_safe.
    fail_safe_decision: str = "REVIEW"

    # ---- storage layout (every path configurable, nothing hard-coded) --------
    storage_root: Path = PROJECT_ROOT / "data"
    recordings_dir: Path = PROJECT_ROOT / "data" / "recordings"
    snapshots_masked_dir: Path = PROJECT_ROOT / "data" / "snapshots" / "masked"
    snapshots_raw_dir: Path = PROJECT_ROOT / "data" / "snapshots" / "raw"
    logs_dir: Path = PROJECT_ROOT / "data" / "logs"
    reports_dir: Path = PROJECT_ROOT / "data" / "reports"

    # ---- structured events (shared/events.py, server/event_store.py) --------
    # A violation must be seen on this many consecutive inference frames before an event
    # record is created. 1 disables the debounce; keep it small - this is a guard against
    # one bad frame, not a second temporal filter.
    violation_confirmation_frames: int = 3
    # A continuing problem seen again within this many seconds extends the open event
    # instead of creating another one.
    event_merge_window_s: float = 30.0
    events_dir: Path = PROJECT_ROOT / "data" / "events"

    # ---- retention, per kind of data ----------------------------------------
    event_retention_days: int = 30         # CLOSED events only; open ones are protected
    video_retention_days: int = 7          # clips are the biggest and least reusable
    snapshot_retention_days: int = 30
    log_retention_days: int = 90           # metadata is small and is the audit record

    # ---- event-based recording ----------------------------------------------
    # OFF by default, like snapshots. When on, a rolling buffer keeps the last few
    # seconds in memory and a clip is written only when an event fires.
    record_events: bool = False
    clip_seconds_before: float = 5.0
    clip_seconds_after: float = 5.0
    clip_fps: float = 10.0
    max_clip_bytes: int = 25 * 1024 * 1024

    # Event snapshots are OFF by default - the system stores metadata only.
    # Switch on with PPE_STORE_SNAPSHOTS=true; faces are then ALWAYS masked first
    # (shared/privacy.py), and a masking failure means no file is written at all.
    store_snapshots: bool = False
    # Raw (unmasked) snapshots are a separate, deliberately awkward switch. Leave off.
    store_raw_snapshots: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
