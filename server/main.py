"""
FastAPI application entry point.

Run (from the project root, with the venv active):
    uvicorn server.main:app --reload --port 8000

Then open:
    http://127.0.0.1:8000/health   -> JSON health check
    http://127.0.0.1:8000/docs     -> interactive API docs (Swagger UI)
"""
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI

from server.config import Settings, get_settings
from server.database import init_db, make_engine, make_session_factory, ping
from shared.schemas import CONTRACT_VERSION, HealthResponse, utc_now


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
        return {"service": "ppe-safety-copilot", "docs": "/docs", "health": "/health"}

    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    def health():
        db_ok = ping(app.state.engine)
        return HealthResponse(
            status="ok" if db_ok else "degraded",
            database="ok" if db_ok else "unreachable",
            api_version=settings.api_version,
            server_time=utc_now(),
        )

    # Day 2: app.include_router(events_router, prefix="/api/v1")
    return app


app = create_app()
