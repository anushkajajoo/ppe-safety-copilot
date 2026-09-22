"""
Database connection helpers (SQLAlchemy 2.0).

WHAT: creates the "engine" (the connection to the SQLite file) and a session
factory (a session = one unit of work / one conversation with the database).
"""
from pathlib import Path
from typing import Iterator

from fastapi import Request
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    """All ORM tables inherit from this."""


def make_engine(database_url: str, echo: bool = False) -> Engine:
    connect_args = {}
    if database_url.startswith("sqlite"):
        # FastAPI may use the connection from different threads.
        connect_args["check_same_thread"] = False
        # Make sure the folder for the .db file exists.
        path = database_url.replace("sqlite:///", "", 1)
        if path and path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(database_url, echo=echo, connect_args=connect_args, future=True)

    if database_url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _record):
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")   # SQLite ignores FKs unless told
            cur.execute("PRAGMA journal_mode=WAL")  # readers don't block the writer
            cur.close()

    return engine


def make_session_factory(engine: Engine) -> sessionmaker:
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db(engine: Engine) -> None:
    """Create all tables that don't exist yet. (Alembic migrations are out of scope for the MVP.)"""
    from server import models  # noqa: F401  (import registers the tables on Base.metadata)
    Base.metadata.create_all(engine)


def ping(engine: Engine) -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def get_db(request: Request) -> Iterator[Session]:
    """FastAPI dependency: one session per request, always closed afterwards."""
    session: Session = request.app.state.session_factory()
    try:
        yield session
    finally:
        session.close()
