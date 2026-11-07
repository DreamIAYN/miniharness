"""The agent loop: plan → call LLM → execute tools → repeat → final answer.

Events are emitted as plain dictionaries so the same loop powers both the
blocking JSON endpoint and the SSE endpoint.

Two advanced behaviours live here:

* **Context compaction** — before a model call, if the estimated prompt exceeds
  ``agent_context_window_tokens * agent_context_compress_ratio``, older messages
  are summarized by the model (or dropped if summarization is unavailable).
* **Subagents** — each run exposes a ``spawn_subagent`` tool that starts a
  nested run at ``depth + 1``, sharing the parent's workspace and quota.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from .config import Settings
from .db import Database
from .errors import HarnessError, NotFoundError, QuotaExceeded, RunTimeout
from .live import LiveRunRegistry
from .llm import (
    LLMClient,
    LLMResponse,
    TokenUsage,
    estimate_messages_tokens,
    estimate_tokens,
)
from .models import ModelCatalog
from .sandbox.base import Sandbox
from .tools.base import SpawnSubagent, ToolContext, ToolRegistry, build_default_registry
from .workspace import WorkspaceManager

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a capable assistant running inside a sandboxed agent harness.

You have tools for running shell commands and Python inside an isolated sandbox, and for reading, writing and listing files in a fresh per-run workspace. Tool paths are relative to that workspace. Each top-level run gets a clean workspace, and the workspace is discarded when the run ends.

Guidelines:
- Use tools instead of guessing for calculations, file inspection and anything that must be exact.
- Prefer run_python for computation and data processing; use run_shell for filesystem inspection and command-line tools.
- For a large, well-scoped subtask, you may delegate it with spawn_subagent. The subagent shares this workspace and its tokens count against the same user quota. Do not delegate trivial work, and do not create deep chains of subagents.
- Older context may be automatically compressed. Treat the summary as reliable, but re-check critical facts with tools if needed.
- Keep tool calls purposeful. Do not run destructive commands. Never try to access paths outside the workspace or probe the host.
- If a tool fails, read the error and adapt; do not repeat the identical failing call.
- When you have enough information, stop calling tools and give a concise final answer.
- Answer in the same language as the user unless asked otherwise."""

SUBAGENT_SYSTEM_PROMPT = """You are a focused subagent working on a task delegated by a parent agent.

You share the parent agent's workspace. Use tools when they make the result more accurate. Keep your work scoped to the delegated task; do not ask the user questions. When done, reply with a concise result and the key evidence (values, file names, errors). If you spawn another subagent, do so only when the task is genuinely decomposable."""

SUMMARY_SYSTEM_PROMPT = """You compress conversation history for an AI agent that must continue a task.

Produce a compact, factual summary that preserves:
- the user's goal, constraints and acceptance criteria
- decisions made and why
- tool results that matter: computed values, file names, errors, command outcomes
- open questions and the next step

Omit greetings, redundant reasoning and raw logs that no longer matter. Output only the summary, with no preamble."""


def _tool_args_for_display(raw: str) -> Any:
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw


def _content_to_text_for_transcript(content: Any) -> str:
    """Flatten multimodal content for compaction transcripts without inlining base64."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                parts.append(str(part))
                continue
            kind = part.get("type")
            if kind == "text":
                parts.append(str(part.get("text") or ""))
            elif kind in {"image_url", "input_image"}:
                image = part.get("image_url") or {}
                url = image.get("url") if isinstance(image, dict) else image
                url = str(url or "")
                if len(url) > 100:
                    url = url[:97] + "..."
                parts.append(f"[image: {url}]")
            elif kind == "file":
                parts.append(f"[file: {part.get('file_id') or 'inline'}]")
            else:
                parts.append(json.dumps(part, ensure_ascii=False))
        return "\n".join(parts)
    return str(content or "")


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… [truncated, {len(text) - limit} chars omitted]"


def _limit_log(text: str, limit: int) -> tuple[str, bool]:
    if limit <= 0 or len(text) <= limit:
        return text, False
    return text[:limit] + "\n… [log truncated by CALL_LOG_MAX_CHARS]", True


def _truncate_transcript(text: str, token_limit: int) -> str:
    """Keep the head and tail of a transcript within an estimated token budget."""
    if token_limit <= 0 or estimate_tokens(text) <= token_limit:
        return text
    half = max(1, token_limit // 2)

    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if estimate_tokens(text[:mid]) <= half:
            low = mid
        else:
            high = mid - 1
    head = text[:low]

    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if estimate_tokens(text[-mid:]) <= half:
            low = mid
        else:
            high = mid - 1
    tail = text[-low:] if low else ""
    return head + "\n… [middle of transcript omitted during compaction] …\n" + tail


class AgentRunner:
    def __init__(
        self,
        *,
        db: Database,
        llm: LLMClient,
        sandbox: Sandbox,
        settings: Settings,
        registry: ToolRegistry | None = None,
        workspaces: WorkspaceManager | None = None,
        models: ModelCatalog | None = None,
        live: LiveRunRegistry | None = None,
    ) -> None:
        self.db = db
        self.llm = llm
        self.sandbox = sandbox
        self.settings = settings
        self.models = models or ModelCatalog.from_settings(settings)
        self.live = live
        self.registry = registry or build_default_registry()
        self.workspaces = workspaces or WorkspaceManager(
            settings.workspace_root, keep=settings.keep_workspaces
        )

    # -- quota / limits ----------------------------------------------------
    def _budget(self, user_id: str) -> dict[str, Any]:
        summary = self.db.quota_summary(user_id)
        if summary is None:
            raise NotFoundError("User not found.")
        if not summary["unlimited"] and (summary["remaining_tokens"] or 0) <= 0:
            raise QuotaExceeded(used=summary["used_tokens"], quota=summary["quota_tokens"])
        return summary

    @staticmethod
    def _check_deadline(started: float, limit: float) -> None:
        # limit <= 0 means "no wall-clock limit".
        if limit > 0 and time.monotonic() - started > limit:
            raise RunTimeout(f"Agent run exceeded the {limit:.0f}s wall-clock limit.")

    # -- main loop ---------------------------------------------------------
    async def run_events(
        self,
        *,
        user: dict[str, Any],
        api_key_id: str | None,
        run_id: str,
        prompt: str,
        system: str | None = None,
        history: list[dict[str, Any]] | None = None,
        images: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_steps: int | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        workspace: Path | None = None,
        depth: int = 0,
    ) -> AsyncIterator[dict[str, Any]]:
        settings = self.settings
        user_id = user["id"]
        model_spec = self.models.resolve(model)
        model_alias = model_spec.alias
        model_context_window = model_spec.context_window or settings.agent_context_window_tokens
        max_steps = min(
            max_steps or settings.agent_default_max_steps, settings.agent_max_steps_limit
        )
        max_steps = max(1, max_steps)
        temperature = settings.agent_temperature if temperature is None else temperature
        max_tokens = max_tokens or settings.agent_max_output_tokens
        if model_spec.max_output_tokens > 0:
            max_tokens = min(max_tokens, model_spec.max_output_tokens)

        # Fail fast before consuming the user's remaining budget.
        self._budget(user_id)

        owns_workspace = workspace is None
        if workspace is None:
            workspace = self.workspaces.create(run_id)

        spawn = self._make_spawn(
            user=user,
            api_key_id=api_key_id,
            workspace=workspace,
            depth=depth,
            model=model_alias,
        )
        ctx = ToolContext(
            workspace=workspace,
            sandbox=self.sandbox,
            settings=settings,
            run_id=run_id,
            depth=depth,
            spawn_subagent=spawn,
        )
        started = time.monotonic()
        run_usage = TokenUsage()
        steps_used = 0
        last_summary: dict[str, Any] | None = None

        system_text = (system or SYSTEM_PROMPT).strip()
        if depth >= settings.agent_max_subagent_depth:
            system_text += (
                "\n\nYou are at the maximum subagent depth. "
                "Complete the task yourself; do not attempt to spawn another subagent."
            )
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_text}]
        for item in history or []:
            messages.append({"role": item["role"], "content": item["content"]})

        if images:
            user_content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            for image in images:
                image_url: dict[str, Any] = {"url": image["url"]}
                if image.get("detail"):
                    image_url["detail"] = image["detail"]
                user_content.append({"type": "image_url", "image_url": image_url})
            messages.append({"role": "user", "content": user_content})
        else:
            messages.append({"role": "user", "content": prompt})

        if self.live is not None:
            self.live.create(
                run_id=run_id,
                user_id=user_id,
                email=str(user.get("email") or ""),
                model=model_alias,
                depth=depth,
            )

        def on_delta(event: dict[str, Any]) -> None:
            if self.live is not None:
                self.live.add_delta(
                    run_id,
                    str(event.get("type") or "content"),
                    str(event.get("text") or ""),
                )

        logger.info(
            "agent run %s started user=%s depth=%s max_steps=%s",
            run_id,
            user_id,
            depth,
            max_steps,
        )
        try:
            yield {
                "type": "start",
                "run_id": run_id,
                "max_steps": max_steps,
                "sandbox": self.sandbox.name,
                "depth": depth,
            }

            for step in range(1, max_steps + 1):
                self._check_deadline(started, settings.agent_run_timeout_s)

                messages, comp_usage, comp_event = await self._maybe_compress(
                    messages=messages,
                    user_id=user_id,
                    api_key_id=api_key_id,
                    run_id=run_id,
                    step=step,
                    context_window=model_context_window,
                    model=model_alias,
                )
                if comp_event is not None:
                    run_usage = run_usage + comp_usage
                    last_summary = self.db.quota_summary(user_id) or last_summary
                    yield comp_event

                summary = self._budget(user_id)
                call_cap = max_tokens
                if not summary["unlimited"]:
                    call_cap = max(1, min(max_tokens, summary["remaining_tokens"] or 1))

                if self.live is not None:
                    self.live.start_step(run_id, step)

                response = await self.llm.complete(
                    messages=messages,
                    tools=self._tool_schemas(depth),
                    tool_choice="auto",
                    temperature=temperature,
                    max_tokens=call_cap,
                    model=model_alias,
                    on_delta=on_delta,
                )
                steps_used += 1
                if self.live is not None:
                    self.live.finish_step(
                        run_id,
                        step=step,
                        content=response.content,
                        reasoning=response.reasoning_content,
                        tool_calls=[
                            {"id": call.id, "name": call.name, "arguments": call.arguments}
                            for call in response.tool_calls
                        ],
                        usage=response.usage.as_dict(),
                    )
                run_usage = run_usage + response.usage
                last_summary = self._record(
                    user_id=user_id,
                    api_key_id=api_key_id,
                    run_id=run_id,
                    response=response,
                    messages=messages,
                )
                yield self._usage_event(
                    step=step,
                    response=response,
                    run_usage=run_usage,
                    summary=last_summary,
                )

                if not response.tool_calls:
                    output = response.content.strip() or "(the model returned an empty answer)"
                    if self.live is not None:
                        self.live.finish(run_id, status="completed", output=output)
                    yield self._final_event(
                        run_id=run_id,
                        output=output,
                        steps=steps_used,
                        usage=run_usage,
                        summary=last_summary,
                        stopped_reason="completed",
                        model=model_alias,
                    )
                    return

                # The answer still needs more turns, but the budget is gone.
                if not last_summary["unlimited"] and (last_summary["remaining_tokens"] or 0) <= 0:
                    raise QuotaExceeded(
                        used=last_summary["used_tokens"],
                        quota=last_summary["quota_tokens"],
                    )

                messages.append(
                    {
                        "role": "assistant",
                        "content": response.content or "",
                        "tool_calls": [
                            {
                                "id": call.id,
                                "type": "function",
                                "function": {"name": call.name, "arguments": call.arguments},
                            }
                            for call in response.tool_calls
                        ],
                    }
                )

                for call in response.tool_calls:
                    self._check_deadline(started, settings.agent_run_timeout_s)
                    yield {
                        "type": "tool_call",
                        "step": step,
                        "call_id": call.id,
                        "tool": call.name,
                        "arguments": _tool_args_for_display(call.arguments),
                    }
                    result = await self.registry.execute(call.name, call.arguments, ctx)
                    child_usage = result.meta.get("subagent_usage")
                    if child_usage:
                        run_usage = run_usage + TokenUsage(
                            prompt_tokens=int(child_usage.get("prompt_tokens") or 0),
                            completion_tokens=int(child_usage.get("completion_tokens") or 0),
                            total_tokens=int(child_usage.get("total_tokens") or 0),
                        )
                    payload = result.content
                    if result.is_error and not payload.startswith("ERROR:"):
                        payload = f"ERROR: {payload}"
                    payload = _clip(payload, settings.tool_output_chars)
                    yield {
                        "type": "tool_result",
                        "step": step,
                        "call_id": call.id,
                        "tool": call.name,
                        "is_error": result.is_error,
                        "output": payload,
                        "meta": result.meta,
                    }
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": payload,
                        }
                    )

                # Subagents may have consumed quota since the last usage event.
                refreshed = self.db.quota_summary(user_id)
                if refreshed is not None:
                    last_summary = refreshed

            # Tool-call budget exhausted: ask once more, without tools, for the
            # best possible final answer.
            self._check_deadline(started, settings.agent_run_timeout_s)
            messages, comp_usage, comp_event = await self._maybe_compress(
                messages=messages,
                user_id=user_id,
                api_key_id=api_key_id,
                run_id=run_id,
                step=steps_used + 1,
                context_window=model_context_window,
                model=model_alias,
            )
            if comp_event is not None:
                run_usage = run_usage + comp_usage
                last_summary = self.db.quota_summary(user_id) or last_summary
                yield comp_event

            summary = self._budget(user_id)
            call_cap = max_tokens
            if not summary["unlimited"]:
                call_cap = max(1, min(max_tokens, summary["remaining_tokens"] or 1))
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "You have reached the tool-call limit for this run. "
                        "Do not call any more tools. Give your best final answer "
                        "now, based on everything above."
                    ),
                }
            )
            final_step = steps_used + 1
            if self.live is not None:
                self.live.start_step(run_id, final_step)
            response = await self.llm.complete(
                messages=messages,
                tools=None,
                tool_choice=None,
                temperature=temperature,
                max_tokens=call_cap,
                model=model_alias,
                on_delta=on_delta,
            )
            steps_used += 1
            if self.live is not None:
                self.live.finish_step(
                    run_id,
                    step=final_step,
                    content=response.content,
                    reasoning=response.reasoning_content,
                    tool_calls=[
                        {"id": call.id, "name": call.name, "arguments": call.arguments}
                        for call in response.tool_calls
                    ],
                    usage=response.usage.as_dict(),
                )
            run_usage = run_usage + response.usage
            last_summary = self._record(
                user_id=user_id,
                api_key_id=api_key_id,
                run_id=run_id,
                response=response,
                messages=messages,
            )
            yield self._usage_event(
                step=steps_used,
                response=response,
                run_usage=run_usage,
                summary=last_summary,
            )
            output = response.content.strip() or (
                "The run hit its tool-call limit before a final answer was produced."
            )
            if self.live is not None:
                self.live.finish(run_id, status="completed", output=output)
            yield self._final_event(
                run_id=run_id,
                output=output,
                steps=steps_used,
                usage=run_usage,
                summary=last_summary,
                stopped_reason="max_steps",
                model=model_alias,
            )
        except HarnessError as exc:
            if self.live is not None:
                self.live.finish(run_id, status="failed", error=exc.message)
            raise
        except Exception as exc:
            if self.live is not None:
                self.live.finish(run_id, status="failed", error=str(exc))
            raise
        finally:
            if self.live is not None:
                self.live.finish_if_running(run_id, "cancelled")
            if owns_workspace:
                self.workspaces.cleanup(workspace)
            logger.info(
                "agent run %s finished user=%s depth=%s tokens=%s",
                run_id,
                user_id,
                depth,
                run_usage.total_tokens,
            )

    # -- subagents ---------------------------------------------------------
    def _tool_schemas(self, depth: int) -> list[dict[str, Any]]:
        schemas = self.registry.schemas()
        if depth >= self.settings.agent_max_subagent_depth:
            schemas = [
                schema
                for schema in schemas
                if (schema.get("function") or {}).get("name") != "spawn_subagent"
            ]
        return schemas

    def _make_spawn(
        self,
        *,
        user: dict[str, Any],
        api_key_id: str | None,
        workspace: Path,
        depth: int,
        model: str,
    ) -> SpawnSubagent:
        parent_model = model

        async def spawn(
            *,
            prompt: str,
            system: str | None = None,
            max_steps: int | None = None,
            model: str | None = None,
        ) -> AsyncIterator[dict[str, Any]]:
            child_run_id = f"sub_{uuid.uuid4().hex}"
            logger.info(
                "subagent %s spawned at depth %s model=%s",
                child_run_id,
                depth + 1,
                model or parent_model,
            )
            async for event in self.run_events(
                user=user,
                api_key_id=api_key_id,
                run_id=child_run_id,
                prompt=prompt,
                system=system or SUBAGENT_SYSTEM_PROMPT,
                max_steps=max_steps,
                model=model or parent_model,
                workspace=workspace,
                depth=depth + 1,
            ):
                yield event

        return spawn

    # -- context compaction ------------------------------------------------
    async def _maybe_compress(
        self,
        *,
        messages: list[dict[str, Any]],
        user_id: str,
        api_key_id: str | None,
        run_id: str,
        step: int,
        context_window: int,
        model: str,
    ) -> tuple[list[dict[str, Any]], TokenUsage, dict[str, Any] | None]:
        settings = self.settings
        window = context_window
        if window <= 0:
            return messages, TokenUsage(), None

        before_tokens = estimate_messages_tokens(messages)
        threshold = int(window * settings.agent_context_compress_ratio)
        if before_tokens < threshold or len(messages) <= 1:
            return messages, TokenUsage(), None

        body = messages[1:]
        keep_start = self._find_keep_start(body)
        if keep_start <= 0:
            return messages, TokenUsage(), None

        old = body[:keep_start]
        keep = body[keep_start:]
        if len(old) == 1 and str(old[0].get("content") or "").startswith("[Compressed context"):
            # Nothing new to summarize; avoid a compaction loop.
            return messages, TokenUsage(), None

        # Do not pay for a summary call when the dropped prefix is tiny: the
        # summary itself could be larger than what we remove.
        old_tokens = estimate_messages_tokens(old)
        min_old_tokens = max(1_024, int(threshold * 0.05))
        if old_tokens < min_old_tokens:
            return messages, TokenUsage(), None

        transcript = self._format_transcript(old)
        transcript = _truncate_transcript(transcript, settings.agent_context_compress_input_tokens)

        usage = TokenUsage()
        summarized = False
        summary_text = ""
        try:
            summary = self._budget(user_id)
            cap = settings.agent_context_summary_max_tokens
            if not summary["unlimited"]:
                cap = min(cap, summary["remaining_tokens"] or 1)
            summary_messages = [
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Summarize the following agent transcript so the agent can "
                        "continue the same task. Keep concrete values, file names, "
                        "errors and decisions.\n\n" + transcript
                    ),
                },
            ]
            response = await self.llm.complete(
                messages=summary_messages,
                tools=None,
                tool_choice=None,
                temperature=0.1,
                max_tokens=max(1, cap),
                model=model,
            )
            usage = response.usage
            self._record(
                user_id=user_id,
                api_key_id=api_key_id,
                run_id=run_id,
                response=response,
                messages=summary_messages,
            )
            summary_text = response.content.strip()
            summarized = bool(summary_text)
        except HarnessError as exc:
            logger.warning("context compaction summary failed: %s", exc)
        except Exception:
            logger.exception("unexpected context compaction failure")

        if not summary_text:
            summary_text = (
                f"[Earlier context omitted to fit the context window: "
                f"{len(old)} messages, about {estimate_tokens(transcript)} tokens.]"
            )

        new_messages = [
            messages[0],
            {
                "role": "user",
                "content": f"[Compressed context from earlier in this run]\n{summary_text}",
            },
            *keep,
        ]
        after_tokens = estimate_messages_tokens(new_messages)
        event = {
            "type": "compaction",
            "step": step,
            "model": model,
            "before_tokens": before_tokens,
            "after_tokens": after_tokens,
            "dropped_messages": len(old),
            "summarized": summarized,
            "usage": usage.as_dict(),
        }
        logger.info(
            "compacted run %s context: %s -> %s tokens (summarized=%s)",
            run_id,
            before_tokens,
            after_tokens,
            summarized,
        )
        return new_messages, usage, event

    @staticmethod
    def _find_keep_start(body: list[dict[str, Any]]) -> int:
        # Keep the latest assistant tool-call round plus its tool results, so
        # the compacted history never starts with an orphaned tool message.
        for index in range(len(body) - 1, -1, -1):
            message = body[index]
            if message.get("role") == "assistant" and message.get("tool_calls"):
                return index
        for index in range(len(body) - 1, -1, -1):
            if body[index].get("role") == "user":
                return index
        return max(0, len(body) - 2)

    @staticmethod
    def _format_transcript(messages: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for message in messages:
            role = message.get("role", "unknown")
            content = _content_to_text_for_transcript(message.get("content"))
            if role == "assistant" and message.get("tool_calls"):
                calls = []
                for call in message["tool_calls"]:
                    function = call.get("function") or {}
                    calls.append(f"{function.get('name')}({function.get('arguments')})")
                content = (content + "\n" if content else "") + "tool_calls: " + "; ".join(calls)
            parts.append(f"[{role}]\n{content}")
        return "\n\n".join(parts)

    # -- event builders ----------------------------------------------------
    def _record(
        self,
        *,
        user_id: str,
        api_key_id: str | None,
        run_id: str,
        response: LLMResponse,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        request_json: str | None = None
        response_text = ""
        tool_calls_json = "[]"
        reasoning_text = ""
        log_truncated = False

        if self.settings.call_log_enabled:
            request_json = json.dumps(messages, ensure_ascii=False, default=str)
            response_text = response.content or ""
            reasoning_text = response.reasoning_content or ""
            tool_calls_json = json.dumps(
                [
                    {"id": call.id, "name": call.name, "arguments": call.arguments}
                    for call in response.tool_calls
                ],
                ensure_ascii=False,
            )
            limit = self.settings.call_log_max_chars
            if limit > 0:
                request_json, truncated_request = _limit_log(request_json, limit)
                response_text, truncated_response = _limit_log(response_text, limit)
                tool_calls_json, truncated_calls = _limit_log(tool_calls_json, limit)
                reasoning_text, truncated_reasoning = _limit_log(reasoning_text, limit)
                log_truncated = any(
                    (
                        truncated_request,
                        truncated_response,
                        truncated_calls,
                        truncated_reasoning,
                    )
                )

        return self.db.record_usage(
            user_id=user_id,
            api_key_id=api_key_id,
            run_id=run_id,
            model=response.model,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
            request_json=request_json,
            response_text=response_text,
            tool_calls_json=tool_calls_json,
            reasoning_text=reasoning_text,
            log_truncated=log_truncated,
        )

    @staticmethod
    def _usage_event(
        *,
        step: int,
        response: LLMResponse,
        run_usage: TokenUsage,
        summary: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "type": "usage",
            "step": step,
            "model": response.model,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
            "run_tokens": run_usage.total_tokens,
            "remaining_tokens": summary["remaining_tokens"],
        }

    @staticmethod
    def _final_event(
        *,
        run_id: str,
        output: str,
        steps: int,
        usage: TokenUsage,
        summary: dict[str, Any],
        stopped_reason: str,
        model: str,
    ) -> dict[str, Any]:
        return {
            "type": "final",
            "run_id": run_id,
            "model": model,
            "output": output,
            "steps": steps,
            "usage": usage.as_dict(),
            "remaining_tokens": summary["remaining_tokens"],
            "stopped_reason": stopped_reason,
        }
