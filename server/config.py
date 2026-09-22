"""
Server settings, read from environment variables or the .env file.

WHY: secrets and paths must NOT be hard-coded (and must never be committed to
GitHub). Copy .env.example -> .env and edit values there.
"""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", env_prefix="PPE_", extra="ignore")

    api_version: str = "0.1.0"
    # SQLite file inside ./data (gitignored). Absolute path so it works from any folder.
    database_url: str = f"sqlite:///{(PROJECT_ROOT / 'data' / 'server.db').as_posix()}"
    evidence_dir: Path = PROJECT_ROOT / "data" / "evidence"
    evidence_retention_days: int = 30
    # Used from Day 3 for edge -> server authentication. Never commit the real value.
    edge_api_key: str = "change-me"
    sql_echo: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
