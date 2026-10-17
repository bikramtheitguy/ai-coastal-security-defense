"""Runtime configuration, read once from environment variables.

Demo launchers explicitly select APP_MODE=demo. Direct startup defaults to
restricted mode, which requires a signing secret and prohibits synthetic seeding
and simulation. Restricted mode is a configuration guard, not production approval.
"""
from __future__ import annotations

import os
import secrets
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
    app_mode: str = field(default_factory=lambda: os.getenv("APP_MODE", "restricted").strip().lower())
    database_url: str = field(default_factory=lambda: os.getenv(
        "DATABASE_URL", f"sqlite:///{Path(os.getenv('DATA_DIR', str(PROJECT_ROOT / 'data'))) / 'iccc_poc.db'}"))
    secret_key: str = field(default_factory=lambda: os.getenv(
        "SECRET_KEY", ""))
    jwt_ttl_minutes: int = field(default_factory=lambda: int(os.getenv("JWT_TTL_MINUTES", "480")))
    session_idle_minutes: int = field(default_factory=lambda: int(os.getenv("SESSION_IDLE_MINUTES", "30")))
    max_failed_logins: int = field(default_factory=lambda: int(os.getenv("MAX_FAILED_LOGINS", "5")))
    lockout_minutes: int = field(default_factory=lambda: int(os.getenv("LOCKOUT_MINUTES", "15")))
    pbkdf2_iterations: int = field(default_factory=lambda: int(os.getenv("PBKDF2_ITERATIONS", "210000")))
    demo_password: str = field(default_factory=lambda: os.getenv("DEMO_PASSWORD", "Demo@2026"))
    # Simulation engine: moves assets/vessels and runs analytics on a timer.
    sim_enabled: bool = field(default_factory=lambda: _bool("SIM_ENABLED", os.getenv("APP_MODE", "restricted").strip().lower() == "demo"))
    sim_tick_seconds: float = field(default_factory=lambda: float(os.getenv("SIM_TICK_SECONDS", "3")))
    # Simulated seconds that elapse per real second (movement only). 20 => a 6 NM transit at 18 kn takes ~1 min.
    sim_time_factor: float = field(default_factory=lambda: float(os.getenv("SIM_TIME_FACTOR", "20")))
    seed_on_start: bool = field(default_factory=lambda: _bool("SEED_ON_START", os.getenv("APP_MODE", "restricted").strip().lower() == "demo"))
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

    def __post_init__(self):
        if self.app_mode not in {"demo", "restricted"}:
            raise ValueError("APP_MODE must be demo or restricted")
        if self.is_demo and not self.secret_key:
            # Ephemeral per-process key; set SECRET_KEY for stable demo sessions across restarts.
            self.secret_key = secrets.token_urlsafe(48)

    def validate_startup(self):
        if not self.is_demo:
            if (len(self.secret_key.strip()) < 32 or self.secret_key.startswith("POC-ONLY")
                    or self.secret_key.startswith("synthetic-ci") or "change_me" in self.secret_key.lower()
                    or "change-me" in self.secret_key.lower()):
                raise ValueError("Restricted mode requires an independently generated SECRET_KEY (32+ characters)")
            if self.seed_on_start or self.sim_enabled:
                raise ValueError("Restricted mode forbids SEED_ON_START and SIM_ENABLED")

    @property
    def is_demo(self) -> bool:
        return self.app_mode == "demo"

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")


settings = Settings()
