"""In-memory registry for in-progress runs and partial model output.

The registry is deliberately process-local: it is a live view, not an audit
log.  Completed call bodies still go to ``call_logs`` for the 7-day history.
A second harness instance (blue/green) can serve new runs while the old
process keeps its already-running tasks untouched.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _clip(text: str, limit: int) -> tuple[str, bool]:
    if limit <= 0 or len(text) <= limit:
        return text, False
    return text[:limit] + "\n… [live view truncated]", True


@dataclass(slots=True)
class LiveRunState:
    run_id: str
    user_id: str
    email: str
    model: str
    depth: int
    started_ts: float
    started_at: str
    status: str = "running"
    current_step: int = 0
    content: str = ""
    reasoning: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
    output: str = ""
    error: str = ""
    truncated: bool = False
    updated_at: str = ""
    finished_ts: float | None = None

    def snapshot(self) -> dict[str, Any]:
        now = time.time()
        end = self.finished_ts or now
        return {
            "run_id": self.run_id,
            "user_id": self.user_id,
            "email": self.email,
            "model": self.model,
            "depth": self.depth,
            "is_subagent": self.depth > 0 or self.run_id.startswith("sub_"),
            "status": self.status,
            "current_step": self.current_step,
            "content": self.content,
            "reasoning": self.reasoning,
            "tool_calls": self.tool_calls,
            "steps": self.steps,
            "output": self.output,
            "error": self.error,
            "truncated": self.truncated,
            "started_at": self.started_at,
            "updated_at": self.updated_at or self.started_at,
            "elapsed_ms": int((end - self.started_ts) * 1000),
            "chars": len(self.content),
        }


class LiveRunRegistry:
    def __init__(self, *, max_chars: int = 200_000, ttl_seconds: int = 3600) -> None:
        self.max_chars = max_chars
        self.ttl_seconds = ttl_seconds
        self._runs: dict[str, LiveRunState] = {}
        self._lock = threading.RLock()

    def create(
        self,
        *,
        run_id: str,
        user_id: str,
        email: str,
        model: str,
        depth: int,
    ) -> None:
        now = time.time()
        with self._lock:
            self._runs[run_id] = LiveRunState(
                run_id=run_id,
                user_id=user_id,
                email=email,
                model=model,
                depth=depth,
                started_ts=now,
                started_at=_now_iso(),
                updated_at=_now_iso(),
            )

    def start_step(self, run_id: str, step: int) -> None:
        with self._lock:
            state = self._runs.get(run_id)
            if state is None:
                return
            state.current_step = step
            state.content = ""
            state.reasoning = ""
            state.tool_calls = []
            state.updated_at = _now_iso()

    def add_delta(self, run_id: str, kind: str, text: str) -> None:
        if not text or kind not in {"content", "reasoning"}:
            return
        with self._lock:
            state = self._runs.get(run_id)
            if state is None or state.status != "running":
                return
            current = state.content if kind == "content" else state.reasoning
            current += text
            current, clipped = _clip(current, self.max_chars)
            if kind == "content":
                state.content = current
            else:
                state.reasoning = current
            state.truncated = state.truncated or clipped
            state.updated_at = _now_iso()

    def finish_step(
        self,
        run_id: str,
        *,
        step: int,
        content: str,
        reasoning: str,
        tool_calls: list[dict[str, Any]],
        usage: dict[str, Any],
    ) -> None:
        with self._lock:
            state = self._runs.get(run_id)
            if state is None:
                return
            content, c1 = _clip(content or "", self.max_chars)
            reasoning, c2 = _clip(reasoning or "", self.max_chars)
            state.content = content
            state.reasoning = reasoning
            state.tool_calls = tool_calls
            state.truncated = state.truncated or c1 or c2
            state.steps.append(
                {
                    "step": step,
                    "content": content,
                    "reasoning": reasoning,
                    "tool_calls": tool_calls,
                    "usage": usage,
                    "truncated": c1 or c2,
                }
            )
            state.updated_at = _now_iso()

    def finish(
        self,
        run_id: str,
        *,
        status: str,
        output: str = "",
        error: str = "",
    ) -> None:
        with self._lock:
            state = self._runs.get(run_id)
            if state is None:
                return
            if state.status != "running":
                return
            state.status = status
            state.output, clipped = _clip(output or "", self.max_chars)
            state.truncated = state.truncated or clipped
            state.error = error
            state.finished_ts = time.time()
            state.updated_at = _now_iso()

    def finish_if_running(self, run_id: str, status: str = "cancelled") -> None:
        with self._lock:
            state = self._runs.get(run_id)
            if state is not None and state.status == "running":
                self.finish(run_id, status=status, error="run ended without a final event")

    def snapshot(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            state = self._runs.get(run_id)
            return state.snapshot() if state else None

    def list(
        self,
        *,
        include_finished: bool = False,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        now = time.time()
        with self._lock:
            stale = [
                run_id
                for run_id, state in self._runs.items()
                if state.finished_ts is not None and now - state.finished_ts > self.ttl_seconds
            ]
            for run_id in stale:
                self._runs.pop(run_id, None)
            states = [
                state
                for state in self._runs.values()
                if include_finished or state.status == "running"
            ]
            states.sort(key=lambda state: state.started_ts, reverse=True)
            return [state.snapshot() for state in states[:limit]]

    def active_count(self) -> int:
        with self._lock:
            return sum(1 for state in self._runs.values() if state.status == "running")
