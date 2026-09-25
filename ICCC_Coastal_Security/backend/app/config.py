"""Runtime configuration, read once from environment variables.

Every setting has a safe POC default so the platform runs with zero configuration.
Production deployments MUST override SECRET_KEY and DEMO_PASSWORD.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    database_url: str = field(default_factory=lambda: os.getenv(
        "DATABASE_URL", f"sqlite:///{Path(os.getenv('DATA_DIR', str(PROJECT_ROOT / 'data'))) / 'iccc_poc.db'}"))
    secret_key: str = field(default_factory=lambda: os.getenv(
        "SECRET_KEY", "POC-ONLY-CHANGE-ME-this-key-is-not-secret"))
    jwt_ttl_minutes: int = field(default_factory=lambda: int(os.getenv("JWT_TTL_MINUTES", "480")))
    session_idle_minutes: int = field(default_factory=lambda: int(os.getenv("SESSION_IDLE_MINUTES", "30")))
    max_failed_logins: int = field(default_factory=lambda: int(os.getenv("MAX_FAILED_LOGINS", "5")))
    lockout_minutes: int = field(default_factory=lambda: int(os.getenv("LOCKOUT_MINUTES", "15")))
    pbkdf2_iterations: int = field(default_factory=lambda: int(os.getenv("PBKDF2_ITERATIONS", "210000")))
    demo_password: str = field(default_factory=lambda: os.getenv("DEMO_PASSWORD", "Demo@2026"))
    # Simulation engine: moves assets/vessels and runs analytics on a timer.
    sim_enabled: bool = field(default_factory=lambda: _bool("SIM_ENABLED", True))
    sim_tick_seconds: float = field(default_factory=lambda: float(os.getenv("SIM_TICK_SECONDS", "3")))
    # Simulated seconds that elapse per real second (movement only). 20 => a 6 NM transit at 18 kn takes ~1 min.
    sim_time_factor: float = field(default_factory=lambda: float(os.getenv("SIM_TIME_FACTOR", "20")))
    seed_on_start: bool = field(default_factory=lambda: _bool("SEED_ON_START", True))
    seed_random: int = field(default_factory=lambda: int(os.getenv("SEED_RANDOM", "20260924")))
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", str(PROJECT_ROOT / "data"))))
    frontend_dist: Path = field(default_factory=lambda: Path(os.getenv(
        "FRONTEND_DIST", str(PROJECT_ROOT / "frontend" / "out"))))
    static_chart_dir: Path = field(default_factory=lambda: Path(os.getenv(
        "STATIC_CHART_DIR", str(PROJECT_ROOT / "assets" / "static_nautical_chart"))))
    cors_origins: str = field(default_factory=lambda: os.getenv("CORS_ORIGINS", "http://localhost:3000"))
    map_tile_url: str = field(default_factory=lambda: os.getenv("MAP_TILE_URL", "https://tile.openstreetmap.org/{z}/{x}/{y}.png"))
    seamark_tile_url: str = field(default_factory=lambda: os.getenv(
        "SEAMARK_TILE_URL", "https://tiles.openseamap.org/seamark/{z}/{x}/{y}.png"))
    stale_after_seconds: int = field(default_factory=lambda: int(os.getenv("STALE_AFTER_SECONDS", "300")))

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")


settings = Settings()
