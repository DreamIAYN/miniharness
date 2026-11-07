"""Admin routes: users, API keys and quotas.

All routes require the ``X-Admin-Token`` header.  This is intentionally a
small, single-tenant admin surface; put it behind your own SSO/reverse proxy
for multi-operator deployments.
"""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse

from ..api.deps import require_admin
from ..api.schemas import (
    CallBodyOut,
    CallEventOut,
    CallHistoryOut,
    KeyCreate,
    KeyCreatedOut,
    KeyOut,
    LiveRunListOut,
    LiveRunOut,
    QuotaSummary,
    StatsOut,
    UsageEventOut,
    UsageOut,
    UserCreate,
    UserCreatedOut,
    UserListOut,
    UserOut,
    UserUpdate,
)
from ..db import Database
from ..errors import HarnessError, NotFoundError

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_admin)])


def _db(request: Request) -> Database:
    return request.app.state.db


def _user_or_404(db: Database, user_id: str) -> dict:
    record = db.get_user_with_usage(user_id)
    if record is None:
        raise NotFoundError(f"User {user_id!r} not found.")
    return record


@router.get("/stats", response_model=StatsOut)
async def stats(request: Request) -> StatsOut:
    return StatsOut(**_db(request).stats())


@router.get("/usage", response_model=CallHistoryOut)
async def call_history(
    request: Request,
    q: str = Query(default="", max_length=100),
    user_id: str | None = Query(default=None),
    run_id: str | None = Query(default=None),
    since: str | None = Query(default=None, max_length=40),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> CallHistoryOut:
    rows, total = _db(request).list_all_usage(
        limit=limit,
        offset=offset,
        query=q,
        user_id=user_id,
        run_id=run_id,
        since=since,
    )
    items = []
    for row in rows:
        row["is_subagent"] = row["run_id"].startswith("sub_")
        items.append(CallEventOut(**row))
    return CallHistoryOut(items=items, total=total, limit=limit, offset=offset)


@router.get("/usage/{usage_event_id}/body", response_model=CallBodyOut)
async def call_body(usage_event_id: int, request: Request) -> CallBodyOut:
    row = _db(request).get_call_log(usage_event_id)
    if row is None:
        raise NotFoundError("Call body not found (expired, disabled, or never recorded).")

    request_messages: list[dict] = []
    request_text: str | None = None
    try:
        parsed = json.loads(row["request_json"])
        if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
            request_messages = parsed
        else:
            request_text = row["request_json"]
    except (TypeError, json.JSONDecodeError):
        request_text = row["request_json"]

    try:
        tool_calls = json.loads(row["tool_calls_json"] or "[]")
        if not isinstance(tool_calls, list):
            tool_calls = []
    except (TypeError, json.JSONDecodeError):
        tool_calls = []

    return CallBodyOut(
        usage_event_id=row["usage_event_id"],
        run_id=row["run_id"],
        email=row["email"],
        name=row["name"],
        model=row["model"],
        created_at=row["created_at"],
        request_messages=request_messages,
        request_text=request_text,
        response_text=row["response_text"] or "",
        tool_calls=tool_calls,
        reasoning_text=row["reasoning_text"] or "",
        truncated=bool(row["truncated"]),
    )


@router.get("/users", response_model=UserListOut)
async def list_users(
    request: Request,
    q: str = Query(default="", max_length=100),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> UserListOut:
    rows, total = _db(request).list_users(limit=limit, offset=offset, query=q)
    return UserListOut(items=[UserOut.from_record(row) for row in rows], total=total)


@router.post("/users", response_model=UserCreatedOut, status_code=201)
async def create_user(body: UserCreate, request: Request) -> UserCreatedOut:
    db = _db(request)
    settings = request.app.state.settings
    quota = settings.default_quota_tokens if body.quota_tokens is None else body.quota_tokens
    user = db.create_user(email=body.email, name=body.name, quota_tokens=quota)
    raw_key: str | None = None
    if body.issue_key:
        raw_key, _ = db.create_api_key(user["id"], name=body.key_name)
    record = _user_or_404(db, user["id"])
    return UserCreatedOut(user=UserOut.from_record(record), api_key=raw_key)


@router.get("/users/{user_id}", response_model=UserOut)
async def get_user(user_id: str, request: Request) -> UserOut:
    return UserOut.from_record(_user_or_404(_db(request), user_id))


@router.patch("/users/{user_id}", response_model=UserOut)
async def update_user(user_id: str, body: UserUpdate, request: Request) -> UserOut:
    db = _db(request)
    _user_or_404(db, user_id)
    fields = body.model_dump(exclude_none=True)
    if not fields:
        raise HarnessError("No fields to update.", code="nothing_to_update")
    db.update_user(user_id, **fields)
    return UserOut.from_record(_user_or_404(db, user_id))


@router.post("/users/{user_id}/keys", response_model=KeyCreatedOut, status_code=201)
async def create_key(user_id: str, body: KeyCreate, request: Request) -> KeyCreatedOut:
    db = _db(request)
    _user_or_404(db, user_id)
    raw, record = db.create_api_key(user_id, name=body.name)
    return KeyCreatedOut(api_key=raw, key=KeyOut(**record))


@router.get("/users/{user_id}/keys", response_model=list[KeyOut])
async def list_keys(user_id: str, request: Request) -> list[KeyOut]:
    db = _db(request)
    _user_or_404(db, user_id)
    return [KeyOut(**record) for record in db.list_keys(user_id)]


@router.delete("/users/{user_id}/keys/{key_id}")
async def revoke_key(user_id: str, key_id: str, request: Request) -> dict:
    db = _db(request)
    _user_or_404(db, user_id)
    if not db.revoke_key(user_id, key_id):
        raise NotFoundError("Active API key not found.")
    return {"revoked": True, "id": key_id}


@router.get("/users/{user_id}/usage", response_model=UsageOut)
async def get_usage(
    user_id: str,
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
) -> UsageOut:
    db = _db(request)
    _user_or_404(db, user_id)
    summary = db.quota_summary(user_id)
    assert summary is not None
    events = db.list_usage_events(user_id, limit=limit)
    return UsageOut(
        summary=QuotaSummary(**summary),
        events=[UsageEventOut(**event) for event in events],
    )


@router.post("/users/{user_id}/usage/reset", response_model=QuotaSummary)
async def reset_usage(user_id: str, request: Request) -> QuotaSummary:
    db = _db(request)
    _user_or_404(db, user_id)
    return QuotaSummary(**db.reset_usage(user_id))


@router.get("/runs/live", response_model=LiveRunListOut)
async def live_runs(
    request: Request,
    include_finished: bool = False,
    limit: int = Query(default=50, ge=1, le=200),
) -> LiveRunListOut:
    items = request.app.state.live.list(include_finished=include_finished, limit=limit)
    return LiveRunListOut(items=[LiveRunOut(**item) for item in items])


@router.get("/runs/{run_id}/live", response_model=LiveRunOut)
async def live_run(run_id: str, request: Request) -> LiveRunOut:
    snapshot = request.app.state.live.snapshot(run_id)
    if snapshot is None:
        raise NotFoundError(f"Live run {run_id!r} not found (finished or unknown).")
    return LiveRunOut(**snapshot)


@router.get("/runs/{run_id}/live/stream")
async def live_run_stream(run_id: str, request: Request) -> StreamingResponse:
    live = request.app.state.live
    if live.snapshot(run_id) is None:
        raise NotFoundError(f"Live run {run_id!r} not found.")

    async def generate():
        last_content = ""
        last_reasoning = ""
        last_steps_len = -1
        last_status: str | None = None
        last_output = ""
        last_error = ""
        last_truncated = False

        while True:
            snapshot = live.snapshot(run_id)
            if snapshot is None:
                yield 'data: {"type":"error","message":"live run expired"}\n\n'
                yield 'data: {"type":"done"}\n\n'
                return

            content = snapshot["content"]
            reasoning = snapshot["reasoning"]
            content_reset = not content.startswith(last_content)
            content_delta = content if content_reset else content[len(last_content):]
            reasoning_reset = not reasoning.startswith(last_reasoning)
            reasoning_delta = reasoning if reasoning_reset else reasoning[len(last_reasoning):]
            steps = snapshot["steps"]
            steps_changed = len(steps) != last_steps_len

            changed = (
                last_status is None
                or bool(content_delta)
                or bool(reasoning_delta)
                or steps_changed
                or snapshot["status"] != last_status
                or snapshot["output"] != last_output
                or snapshot["error"] != last_error
                or snapshot["truncated"] != last_truncated
            )
            if changed:
                event = {
                    "type": "snapshot",
                    "run_id": run_id,
                    "status": snapshot["status"],
                    "current_step": snapshot["current_step"],
                    "model": snapshot["model"],
                    "email": snapshot["email"],
                    "elapsed_ms": snapshot["elapsed_ms"],
                    "content_delta": content_delta,
                    "content_reset": content_reset,
                    "reasoning_delta": reasoning_delta,
                    "reasoning_reset": reasoning_reset,
                    "steps": steps if steps_changed else None,
                    "output": snapshot["output"],
                    "error": snapshot["error"],
                    "truncated": snapshot["truncated"],
                }
                yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"

            last_content = content
            last_reasoning = reasoning
            last_steps_len = len(steps)
            last_status = snapshot["status"]
            last_output = snapshot["output"]
            last_error = snapshot["error"]
            last_truncated = snapshot["truncated"]

            if snapshot["status"] != "running":
                yield 'data: {"type":"done"}\n\n'
                return
            if await request.is_disconnected():
                return
            await asyncio.sleep(0.15)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
