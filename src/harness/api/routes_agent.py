"""Agent execution routes (blocking JSON and SSE streaming)."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from ..agent import AgentRunner
from ..api.deps import Principal, get_principal
from ..api.schemas import RunRequest, RunResponse, TokenUsageOut
from ..errors import HarnessError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/agent", tags=["agent"])


def _new_run_id() -> str:
    return f"run_{uuid.uuid4().hex}"


def _run_kwargs(req: RunRequest, principal: Principal, run_id: str) -> dict[str, Any]:
    return {
        "user": principal.user,
        "api_key_id": principal.api_key_id,
        "run_id": run_id,
        "prompt": req.prompt,
        "system": req.system,
        "history": [m.model_dump() for m in req.history],
        "images": [image.model_dump() for image in req.images] or None,
        "model": req.model,
        "max_steps": req.max_steps,
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
    }


@router.post("/run", response_model=RunResponse)
async def run_agent(
    body: RunRequest,
    request: Request,
    principal: Principal = Depends(get_principal),
) -> RunResponse:
    runner: AgentRunner = request.app.state.runner
    run_id = _new_run_id()
    final: dict[str, Any] | None = None
    async for event in runner.run_events(**_run_kwargs(body, principal, run_id)):
        if event["type"] == "final":
            final = event
    if final is None:
        raise HarnessError(
            "Agent run ended without a final answer.",
            status_code=500,
            code="agent_failed",
        )
    return RunResponse(
        run_id=final["run_id"],
        model=final.get("model"),
        output=final["output"],
        steps=final["steps"],
        usage=TokenUsageOut(**final["usage"]),
        remaining_tokens=final["remaining_tokens"],
        stopped_reason=final["stopped_reason"],
    )


@router.post("/stream")
async def stream_agent(
    body: RunRequest,
    request: Request,
    principal: Principal = Depends(get_principal),
) -> StreamingResponse:
    runner: AgentRunner = request.app.state.runner
    run_id = _new_run_id()
    kwargs = _run_kwargs(body, principal, run_id)

    async def generate():
        yield ": connected\n\n"
        try:
            async for event in runner.run_events(**kwargs):
                if await request.is_disconnected():
                    logger.info("client disconnected from run %s", run_id)
                    return
                yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
        except HarnessError as exc:
            payload = {
                "type": "error",
                "code": exc.code,
                "message": exc.message,
                "details": exc.details,
            }
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        except Exception:
            logger.exception("unexpected error in streaming run %s", run_id)
            payload = {
                "type": "error",
                "code": "internal_error",
                "message": "Unexpected server error.",
            }
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        yield 'data: {"type": "done"}\n\n'

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
