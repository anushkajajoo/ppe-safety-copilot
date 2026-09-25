"""
FastAPI application entry point.

Run (from the project root, with the venv active):
    uvicorn server.main:app --reload --port 8000

Then open:
    http://127.0.0.1:8000/health   -> JSON health check
    http://127.0.0.1:8000/docs     -> interactive API docs (Swagger UI)
"""
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from server.config import FRONTEND_DIR, Settings, get_settings
from server.database import init_db, make_engine, make_session_factory, ping
from server.predict_service import PredictService
from server.routes import audit as audit_routes
from server.routes import auth as auth_routes
from server.routes import copilot as copilot_routes
from server.routes import history as history_routes
from server.routes import operations as operations_routes
from server.routes import telemetry as telemetry_routes
from server.routes import events as event_routes
from server.routes import predict as predict_routes
from server.routes import live as live_routes
from server.routes import violations as violation_routes
from server.routes import video as video_routes
from server.routes import zones as zone_routes
from shared.schemas import CONTRACT_VERSION, HealthResponse, utc_now
from shared.analytics import DecisionLog
from shared.tools import ToolLog
from shared.violation_log import ViolationLog


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    """App factory: tests call create_app(Settings(database_url=...)) to use a temporary DB."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine = make_engine(settings.database_url, echo=settings.sql_echo)
        init_db(engine)                                   # create tables on startup
        settings.evidence_dir.mkdir(parents=True, exist_ok=True)
        app.state.engine = engine
        app.state.session_factory = make_session_factory(engine)
        app.state.settings = settings
        # The detector is created here but the weights load on the FIRST /predict
        # request, so the server starts instantly and still starts without a model.
        # Who may sign in, and the key their session cookies are signed with.
        from secrets import token_hex
        from server.auth import load_users
        app.state.users = load_users(settings.users_file)
        app.state.session_secret = settings.session_secret or token_hex(32)
        if not app.state.users:
            print("NOTE: no configs/users.yaml - nobody can sign in, so nothing can be "
                  "approved. Create a user with: python -m scripts.make_user")

        # Storage layout: every folder created up front so a first run never fails on a
        # missing directory, and so the structure is visible before anything is written.
        for folder in (settings.storage_root, settings.recordings_dir,
                       settings.snapshots_masked_dir, settings.logs_dir, settings.reports_dir):
            Path(folder).mkdir(parents=True, exist_ok=True)

        # Counted evaluations (analytics) and every tool request (allowed and denied).
        Path(settings.events_dir).mkdir(parents=True, exist_ok=True)
        # One debouncer for the process: a violation must be seen on N
        # consecutive frames before it becomes a record (shared/events.py).
        from shared.events import Debouncer
        app.state.debouncer = Debouncer(frames=settings.violation_confirmation_frames)
        app.state.decision_log = DecisionLog(Path(settings.logs_dir) / "decisions.jsonl")
        app.state.tool_log = ToolLog(Path(settings.logs_dir) / "tool_calls.jsonl")

        app.state.violation_log = ViolationLog(settings.violation_log,
                                               cooldown_s=settings.violation_cooldown_s)
        app.state.predict_service = PredictService(weights=settings.weights,
                                                   conf=settings.detect_conf,
                                                   imgsz=settings.detect_imgsz,
                                                   device=settings.detect_device or None)
        if settings.edge_api_key in ("change-me", "change-me-to-a-long-random-string"):
            print("WARNING: PPE_EDGE_API_KEY is still the default. Set a long random value in .env")
        yield
        engine.dispose()

    app = FastAPI(
        title="Edge Vision Safety Copilot API",
        version=settings.api_version,
        description=f"Backend for BAI-01. Event contract v{CONTRACT_VERSION}.",
        lifespan=lifespan,
    )

    @app.get("/", tags=["meta"])
    def root():
        return {"service": "ppe-safety-copilot", "docs": "/docs", "health": "/health",
                "dashboard": "/dashboard", "predict": "/api/v1/predict",
                "detect_image": "/detect/image", "detect_video": "/detect/video",
                "live": "/live", "violations": "/api/v1/violations"}

    @app.get("/dashboard", include_in_schema=False)
    def dashboard():
        """The simple upload-and-check page (plain HTML + JS, no build step)."""
        return FileResponse(FRONTEND_DIR / "dashboard.html")

    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    def health():
        db_ok = ping(app.state.engine)
        return HealthResponse(
            status="ok" if db_ok else "degraded",
            database="ok" if db_ok else "unreachable",
            api_version=settings.api_version,
            server_time=utc_now(),
        )

    # frontend/ is mounted so the pages can share theme.css (and any future asset).
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="assets")

    app.include_router(predict_routes.router)
    app.include_router(video_routes.router)
    app.include_router(live_routes.router)
    app.include_router(violation_routes.router)
    app.include_router(event_routes.router)
    app.include_router(zone_routes.router)
    app.include_router(audit_routes.router)
    app.include_router(copilot_routes.router)
    app.include_router(telemetry_routes.router)
    app.include_router(auth_routes.router)
    app.include_router(operations_routes.router)
    app.include_router(history_routes.router)
    return app


app = create_app()
