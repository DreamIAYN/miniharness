"""Public, user-facing routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request

from .. import __version__
from ..api.deps import Principal, get_principal
from ..api.schemas import (
    LiveRunListOut,
    LiveRunOut,
    ModelListOut,
    ModelOut,
    QuotaSummary,
    UsageEventOut,
    UsageOut,
    UserOut,
)
from ..db import Database
from ..errors import NotFoundError

router = APIRouter()


@router.get("/", tags=["meta"])
async def index(request: Request) -> dict:
    settings = request.app.state.settings
    catalog = request.app.state.models
    return {
        "name": "mini-harness",
        "version": __version__,
        "docs": "/docs",
        "admin_ui": "/admin/ui",
        "llm_configured": bool(catalog.list()),
        "default_model": catalog.default_alias,
        "models": len(catalog.list()),
        "sandbox": settings.sandbox_backend,
    }


@router.get("/healthz", tags=["meta"])
async def healthz(request: Request) -> dict:
    catalog = request.app.state.models
    return {
        "status": "ok",
        "version": __version__,
        "llm_configured": bool(catalog.list()),
        "default_model": catalog.default_alias,
        "models": len(catalog.list()),
        "sandbox": request.app.state.sandbox.name,
    }


@router.get("/v1/models", response_model=ModelListOut, tags=["user"])
async def list_models(
    request: Request,
    principal: Principal = Depends(get_principal),
) -> ModelListOut:
    catalog = request.app.state.models
    return ModelListOut(
        default=catalog.default_alias,
        items=[ModelOut(**spec.public()) for spec in catalog.list()],
    )


@router.get("/v1/runs/live", response_model=LiveRunListOut, tags=["user"])
async def my_live_runs(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
    principal: Principal = Depends(get_principal),
) -> LiveRunListOut:
    snapshots = request.app.state.live.list(include_finished=False, limit=200)
    items = [
        LiveRunOut(**snapshot)
        for snapshot in snapshots
        if snapshot["user_id"] == principal.user["id"]
    ][:limit]
    return LiveRunListOut(items=items)


@router.get("/v1/runs/{run_id}/live", response_model=LiveRunOut, tags=["user"])
async def my_live_run(
    run_id: str,
    request: Request,
    principal: Principal = Depends(get_principal),
) -> LiveRunOut:
    snapshot = request.app.state.live.snapshot(run_id)
    if snapshot is None or snapshot["user_id"] != principal.user["id"]:
        raise NotFoundError(f"Live run {run_id!r} not found.")
    return LiveRunOut(**snapshot)


@router.get("/v1/me", response_model=UserOut, tags=["user"])
async def me(request: Request, principal: Principal = Depends(get_principal)) -> UserOut:
    db: Database = request.app.state.db
    record = db.get_user_with_usage(principal.user["id"])
    assert record is not None
    return UserOut.from_record(record)


@router.get("/v1/usage", response_model=UsageOut, tags=["user"])
async def my_usage(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
    principal: Principal = Depends(get_principal),
) -> UsageOut:
    db: Database = request.app.state.db
    user_id = principal.user["id"]
    summary = db.quota_summary(user_id)
    assert summary is not None
    events = db.list_usage_events(user_id, limit=limit)
    return UsageOut(
        summary=QuotaSummary(**summary),
        events=[UsageEventOut(**event) for event in events],
    )
