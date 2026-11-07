"""FastAPI application factory."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse

from . import __version__
from .agent import AgentRunner
from .api import routes_admin, routes_agent, routes_public
from .config import Settings
from .db import Database
from .errors import HarnessError
from .live import LiveRunRegistry
from .llm import LLMClient, ModelRouter
from .models import ModelCatalog
from .sandbox import create_sandbox
from .sandbox.base import Sandbox

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


def create_app(
    settings: Settings | None = None,
    *,
    llm: LLMClient | None = None,
    sandbox: Sandbox | None = None,
    models: ModelCatalog | None = None,
    live: LiveRunRegistry | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if settings.admin_token_generated:
        logger.warning(
            "HARNESS_ADMIN_TOKEN was not set; generated one for this process: %s",
            settings.admin_token,
        )

    db = Database(settings.database_path)
    db.init()

    sandbox = sandbox or create_sandbox(settings)
    catalog = models or ModelCatalog.from_settings(settings)
    llm = llm or ModelRouter(catalog, settings)
    live = live or LiveRunRegistry()
    runner = AgentRunner(
        db=db,
        llm=llm,
        sandbox=sandbox,
        settings=settings,
        models=catalog,
        live=live,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await sandbox.start()

        async def purge_call_logs_loop() -> None:
            while True:
                try:
                    removed = db.purge_call_logs(settings.call_log_retention_days)
                    if removed:
                        logger.info("purged %s expired call-log bodies", removed)
                except Exception:
                    logger.exception("call-log retention cleanup failed")
                await asyncio.sleep(3600)

        purge_task: asyncio.Task | None = None
        if settings.call_log_enabled and settings.call_log_retention_days > 0:
            purge_task = asyncio.create_task(purge_call_logs_loop())

        logger.info(
            "mini-harness %s ready (sandbox=%s, models=%s, default_model=%s, call_log_days=%s)",
            __version__,
            sandbox.name,
            len(catalog.list()),
            catalog.default_alias,
            settings.call_log_retention_days if settings.call_log_enabled else "disabled",
        )
        try:
            yield
        finally:
            if purge_task is not None:
                purge_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await purge_task
            await sandbox.close()
            await llm.close()
            db.close()

    app = FastAPI(
        title="mini-harness",
        version=__version__,
        description=(
            "A small LLM agent harness with a pluggable sandbox and per-user "
            "token quotas. Use `Authorization: Bearer hx_...` for the user API "
            "and `X-Admin-Token` for the admin API."
        ),
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.db = db
    app.state.sandbox = sandbox
    app.state.llm = llm
    app.state.models = catalog
    app.state.live = live
    app.state.runner = runner

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        return response

    @app.exception_handler(HarnessError)
    async def harness_error_handler(request: Request, exc: HarnessError) -> JSONResponse:
        headers = {"WWW-Authenticate": "Bearer"} if exc.status_code == 401 else None
        return JSONResponse(status_code=exc.status_code, content=exc.to_dict(), headers=headers)

    app.include_router(routes_public.router)
    app.include_router(routes_agent.router)
    app.include_router(routes_admin.router)

    @app.get("/admin", include_in_schema=False)
    async def admin_redirect() -> RedirectResponse:
        return RedirectResponse(url="/admin/ui", status_code=307)

    @app.get("/admin/ui", include_in_schema=False)
    async def admin_ui() -> FileResponse:
        return FileResponse(
            STATIC_DIR / "admin.html",
            media_type="text/html",
            headers={
                "Cache-Control": "no-store",
                "Content-Security-Policy": (
                    "default-src 'self'; img-src 'self' data: https: http:; "
                    "style-src 'self' 'unsafe-inline'; "
                    "script-src 'self' 'unsafe-inline'; connect-src 'self'"
                ),
            },
        )

    return app
