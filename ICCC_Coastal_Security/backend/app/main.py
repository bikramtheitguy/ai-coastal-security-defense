"""FastAPI application entry point.

Run locally:   uvicorn app.main:app --port 8000   (from backend/)
Serves the REST API under /api and the statically exported Next.js frontend at /.
"""
from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from .config import PROJECT_ROOT, settings
from .db import SessionLocal, create_schema, init_engine

if str(PROJECT_ROOT) not in sys.path:  # make the top-level seed/ package importable
    sys.path.insert(0, str(PROJECT_ROOT))

log = logging.getLogger("iccc")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def bootstrap() -> None:
    init_engine()
    create_schema()
    if settings.seed_on_start:
        from seed.generate import seed_if_empty
        with SessionLocal() as db:
            res = seed_if_empty(db, settings.seed_random, settings.demo_password)
            if res:
                log.info("Seeded synthetic POC dataset: %s", res)


@asynccontextmanager
async def lifespan(app: FastAPI):
    bootstrap()
    task = None
    if settings.sim_enabled:
        from .services import simulator
        task = asyncio.create_task(simulator.run_forever())
    yield
    if task:
        from .services import simulator
        simulator.stop()
        task.cancel()


def create_app() -> FastAPI:
    app = FastAPI(title="ICCC Coastal Security & MDA Platform (POC)", version="0.1.0",
                  description="AI-Enabled Integrated Coastal Security & Maritime Domain Awareness Platform — "
                              "Proof of Concept. ALL DATA IS SIMULATED / POC DATA.", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
                       allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        resp = await call_next(request)
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("X-Frame-Options", "DENY")
        resp.headers.setdefault("Referrer-Policy", "no-referrer")
        resp.headers.setdefault("Permissions-Policy", "geolocation=(self), camera=(), microphone=(self)")
        if request.url.path.startswith("/api/"):
            resp.headers.setdefault("Cache-Control", "no-store")
        return resp

    from .routers import admin, assets, auth, chat, command, cop, incidents, intel, personnel, readiness, system
    for r in (auth, cop, readiness, personnel, assets, incidents, intel, chat, command, system, admin):
        app.include_router(r.router)

    @app.get("/api/public/info")
    def info():
        return {"name": "ICCC Coastal Security & MDA Platform", "mode": "PROOF OF CONCEPT",
                "data_label": "SIMULATED / POC DATA", "map_label": "LIVE PUBLIC MAP + SIMULATED OPERATIONAL DATA — NOT FOR NAVIGATION",
                "chart_label": "LOCAL STATIC NAUTICAL REFERENCE — NOT FOR NAVIGATION",
                "static_chart": _chart_file() is not None, "demo_accounts": True,
                "map_tile_url": settings.map_tile_url, "seamark_tile_url": settings.seamark_tile_url,
                "coastline": _coastline()}

    @app.get("/api/public/static-chart")
    def static_chart():
        f = _chart_file()
        if f is None:
            return JSONResponse({"detail": "No static chart supplied — see assets/static_nautical_chart/README.md"}, 404)
        return FileResponse(f)

    dist = settings.frontend_dist

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path.startswith("api/"):
            return JSONResponse({"detail": "Not found"}, 404)
        if not dist.exists():
            return JSONResponse({"detail": "Frontend not built. Run the frontend build (see README)."}, 503)
        target = (dist / path).resolve()
        if dist.resolve() not in target.parents and target != dist.resolve():
            return JSONResponse({"detail": "Not found"}, 404)
        for cand in (target, target / "index.html", Path(str(target) + ".html")):
            if cand.is_file():
                return FileResponse(cand)
        return FileResponse(dist / "index.html")

    return app


def _coastline():
    from seed.geography import COASTLINE
    return COASTLINE


def _chart_file() -> Path | None:
    d = settings.static_chart_dir
    if not d.exists():
        return None
    for ext in (".png", ".jpg", ".jpeg", ".webp", ".svg"):
        files = sorted(p for p in d.iterdir() if p.suffix.lower() == ext and not p.name.startswith("placeholder"))
        if files:
            return files[0]
    ph = d / "placeholder_no_chart_supplied.svg"
    return ph if ph.exists() else None


app = create_app()
