"""Database engine/session management.

PostgreSQL + PostGIS is the target store (docker compose). SQLite is supported for
single-machine demos and automated tests so the POC can run without containers.
Only portable SQL types are used in the ORM; PostGIS spatial columns/indexes are
added by database/init/*.sql when PostGIS is present.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import settings


class Base(DeclarativeBase):
    pass


_engine: Engine | None = None
SessionLocal = sessionmaker(autoflush=False, expire_on_commit=False)


def init_engine(url: str | None = None) -> Engine:
    global _engine
    url = url or settings.database_url
    if url.startswith("sqlite"):
        path = url.replace("sqlite:///", "")
        if path and path != ":memory:":
            from pathlib import Path
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        eng = create_engine(url, connect_args={"check_same_thread": False, "timeout": 30})

        @event.listens_for(eng, "connect")
        def _pragmas(dbapi_conn, _):  # pragma: no cover - driver hook
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA busy_timeout=30000")
            cur.close()
    else:
        eng = create_engine(url, pool_pre_ping=True, pool_size=10, max_overflow=20)
    _engine = eng
    SessionLocal.configure(bind=eng)
    return eng


def get_engine() -> Engine:
    if _engine is None:
        init_engine()
    assert _engine is not None
    return _engine


def create_schema() -> None:
    from . import models  # noqa: F401  (register mappers)
    eng = get_engine()
    Base.metadata.create_all(eng)
    if eng.dialect.name == "postgresql":
        _postgis_extras(eng)


def _postgis_extras(eng: Engine) -> None:
    """Best-effort PostGIS enrichment: spatial views for GIS clients / future ENC layers."""
    try:
        with eng.begin() as c:
            c.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
            c.execute(text("""
                CREATE OR REPLACE VIEW gis_assets AS
                SELECT id, asset_code, asset_type, ST_SetSRID(ST_MakePoint(lon, lat), 4326) AS geom
                FROM assets WHERE lat IS NOT NULL"""))
            c.execute(text("""
                CREATE OR REPLACE VIEW gis_vessels AS
                SELECT id, vessel_code, name, ST_SetSRID(ST_MakePoint(lon, lat), 4326) AS geom
                FROM vessels WHERE lat IS NOT NULL"""))
            c.execute(text("""
                CREATE OR REPLACE VIEW gis_incidents AS
                SELECT id, code, family, status, ST_SetSRID(ST_MakePoint(lon, lat), 4326) AS geom
                FROM incidents WHERE lat IS NOT NULL"""))
    except Exception:  # PostGIS not installed: the platform still works with lat/lon columns
        pass


def db_session() -> Iterator[Session]:
    """FastAPI dependency."""
    s = SessionLocal()
    try:
        yield s
    finally:
        s.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    s = SessionLocal()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()
