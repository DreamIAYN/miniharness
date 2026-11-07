"""OpenAI-compatible chat-completions client.

Works with OpenAI, DeepSeek, vLLM, Ollama (OpenAI mode), LiteLLM and any other
provider exposing ``POST {base_url}/chat/completions`` with tool calling.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

import httpx

from .config import Settings
from .errors import LLMError, LLMNotConfigured
from .models import ModelCatalog, ModelSpec

logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    """Cheap, dependency-free token estimate.

    Counts CJK characters as one token each and everything else at ~4 chars per
    token.  Only used when a provider omits the ``usage`` block, so that quota
    accounting cannot be silently bypassed.
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return max(1, cjk + (other + 3) // 4)


def estimate_content_tokens(content: Any) -> int:
    if isinstance(content, str):
        return estimate_tokens(content)
    if isinstance(content, list):
        total = 0
        for part in content:
            if not isinstance(part, dict):
                total += estimate_tokens(str(part))
                continue
            kind = part.get("type")
            if kind == "text":
                total += estimate_tokens(str(part.get("text") or ""))
            elif kind in {"image_url", "input_image", "file"}:
                # DeepSeek caps image tokens at 1024 per image; use the cap for
                # conservative fallback accounting when the provider omits usage.
                total += 1024
            else:
                total += estimate_tokens(json.dumps(part, ensure_ascii=False))
        return total
    return estimate_tokens(str(content or ""))


def estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
    total = 0
    for message in messages:
        total += 4
        total += estimate_content_tokens(message.get("content"))
        for call in message.get("tool_calls") or []:
            fn = call.get("function") or {}
            total += estimate_tokens(str(fn.get("name") or ""))
            total += estimate_tokens(str(fn.get("arguments") or ""))
    return total


@dataclass(slots=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass(slots=True)
class LLMResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)
    model: str = ""
    finish_reason: str = ""
    reasoning_content: str = ""


class LLMClient(Protocol):
    async def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = "auto",
        temperature: float = 0.2,
        max_tokens: int = 393216,
        model: str | None = None,
        on_delta: Callable[[dict[str, Any]], None] | None = None,
    ) -> LLMResponse: ...

    async def close(self) -> None: ...


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text") or ""))
        return "".join(parts)
    return str(content)


class OpenAICompatLLM:
    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        thinking_mode: str | None = None,
    ) -> None:
        self.settings = settings
        self._thinking_mode = thinking_mode
        timeout = None if settings.llm_timeout_s <= 0 else httpx.Timeout(settings.llm_timeout_s)
        self._client = httpx.AsyncClient(
            base_url=settings.llm_base_url.rstrip("/"),
            timeout=timeout,
            transport=transport,
        )

    @property
    def configured(self) -> bool:
        return self.settings.llm_configured

    async def close(self) -> None:
        await self._client.aclose()

    async def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = "auto",
        temperature: float = 0.2,
        max_tokens: int = 393216,
        model: str | None = None,
        on_delta: Callable[[dict[str, Any]], None] | None = None,
    ) -> LLMResponse:
        if not self.configured:
            raise LLMNotConfigured()

        if on_delta is not None:
            return await self._complete_streaming(
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature,
                max_tokens=max_tokens,
                model=model,
                on_delta=on_delta,
            )

        payload: dict[str, Any] = {
            "model": model or self.settings.llm_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            if tool_choice:
                payload["tool_choice"] = tool_choice

        thinking_mode = self._thinking_mode
        if thinking_mode is None:
            thinking_mode = "enabled" if self.settings.llm_thinking else "omit"
        if thinking_mode == "enabled":
            payload["thinking"] = {"type": "enabled"}
            effort = (self.settings.llm_reasoning_effort or "").strip()
            if effort:
                payload["reasoning_effort"] = effort
        elif thinking_mode == "disabled":
            payload["thinking"] = {"type": "disabled"}

        headers = {
            "Authorization": f"Bearer {self.settings.llm_api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(self.settings.llm_max_retries + 1):
            try:
                response = await self._client.post(
                    "/chat/completions", json=payload, headers=headers
                )
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt < self.settings.llm_max_retries:
                    await asyncio.sleep(0.5 * (2**attempt))
                    continue
                raise LLMError(f"LLM request failed: {exc}") from exc

            if (
                response.status_code in {429, 500, 502, 503, 504}
                and attempt < self.settings.llm_max_retries
            ):
                logger.warning("LLM provider returned %s, retrying", response.status_code)
                await asyncio.sleep(0.5 * (2**attempt))
                continue

            if response.status_code >= 400:
                detail: Any
                try:
                    detail = response.json()
                except ValueError:
                    detail = response.text[:500]
                raise LLMError(
                    f"LLM provider returned HTTP {response.status_code}.",
                    details={"status": response.status_code, "provider_response": detail},
                )

            try:
                data = response.json()
            except ValueError as exc:
                raise LLMError("LLM provider returned non-JSON response.") from exc

            return self._parse_response(data, messages)

        raise LLMError(f"LLM request failed after retries: {last_error}")

    async def _complete_streaming(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: str | None,
        temperature: float,
        max_tokens: int,
        model: str | None,
        on_delta: Callable[[dict[str, Any]], None],
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": model or self.settings.llm_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            payload["tools"] = tools
            if tool_choice:
                payload["tool_choice"] = tool_choice

        thinking_mode = self._thinking_mode
        if thinking_mode is None:
            thinking_mode = "enabled" if self.settings.llm_thinking else "omit"
        if thinking_mode == "enabled":
            payload["thinking"] = {"type": "enabled"}
            effort = (self.settings.llm_reasoning_effort or "").strip()
            if effort:
                payload["reasoning_effort"] = effort
        elif thinking_mode == "disabled":
            payload["thinking"] = {"type": "disabled"}

        headers = {
            "Authorization": f"Bearer {self.settings.llm_api_key}",
            "Content-Type": "application/json",
        }

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_state: dict[int, dict[str, str]] = {}
        finish_reason = ""
        response_model = self.settings.llm_model
        usage_raw: dict[str, Any] = {}

        def emit(event: dict[str, Any]) -> None:
            try:
                on_delta(event)
            except Exception:  # pragma: no cover - live UI must never break a run
                logger.exception("on_delta callback failed")

        try:
            async with self._client.stream(
                "POST", "/chat/completions", json=payload, headers=headers
            ) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    try:
                        detail: Any = json.loads(body)
                    except (TypeError, ValueError):
                        detail = body.decode("utf-8", errors="replace")[:500]
                    raise LLMError(
                        f"LLM provider returned HTTP {response.status_code}.",
                        details={"status": response.status_code, "provider_response": detail},
                    )

                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if not raw:
                        continue
                    if raw == "[DONE]":
                        break
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    if data.get("model"):
                        response_model = str(data["model"])
                    if data.get("usage"):
                        usage_raw = data["usage"]

                    for choice in data.get("choices") or []:
                        if choice.get("finish_reason"):
                            finish_reason = str(choice["finish_reason"])
                        delta = choice.get("delta") or {}
                        text = _content_to_text(delta.get("content"))
                        if text:
                            content_parts.append(text)
                            emit({"type": "content", "text": text})
                        reasoning = _content_to_text(delta.get("reasoning_content"))
                        if reasoning:
                            reasoning_parts.append(reasoning)
                            emit({"type": "reasoning", "text": reasoning})
                        for raw_call in delta.get("tool_calls") or []:
                            index = raw_call.get("index")
                            if index is None:
                                index = len(tool_state)
                            state = tool_state.setdefault(
                                int(index), {"id": "", "name": "", "arguments": ""}
                            )
                            if raw_call.get("id"):
                                state["id"] = str(raw_call["id"])
                            function = raw_call.get("function") or {}
                            name = function.get("name")
                            if name:
                                if not state["name"]:
                                    state["name"] = str(name)
                                elif str(name) not in state["name"]:
                                    state["name"] += str(name)
                            arguments = function.get("arguments")
                            if arguments:
                                state["arguments"] += str(arguments)
        except httpx.HTTPError as exc:
            raise LLMError(f"LLM streaming request failed: {exc}") from exc

        content = "".join(content_parts)
        reasoning_content = "".join(reasoning_parts)
        tool_calls = [
            ToolCall(
                id=state["id"] or f"call_{index}",
                name=state["name"],
                arguments=state["arguments"] or "{}",
            )
            for index, state in sorted(tool_state.items())
        ]

        prompt_tokens = int(usage_raw.get("prompt_tokens") or 0)
        completion_tokens = int(usage_raw.get("completion_tokens") or 0)
        total_tokens = int(usage_raw.get("total_tokens") or 0)
        if total_tokens <= 0:
            prompt_tokens = prompt_tokens or estimate_messages_tokens(messages)
            completion_tokens = completion_tokens or estimate_tokens(
                content + "".join(call.name + call.arguments for call in tool_calls)
            )
            total_tokens = prompt_tokens + completion_tokens

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            ),
            model=response_model,
            finish_reason=finish_reason,
            reasoning_content=reasoning_content,
        )

    def _parse_response(self, data: dict[str, Any], messages: list[dict[str, Any]]) -> LLMResponse:
        choices = data.get("choices") or []
        if not choices:
            raise LLMError("LLM provider returned no choices.", details={"response": data})

        choice = choices[0]
        message = choice.get("message") or {}
        content = _content_to_text(message.get("content"))
        reasoning_content = _content_to_text(message.get("reasoning_content"))

        tool_calls: list[ToolCall] = []
        for index, raw_call in enumerate(message.get("tool_calls") or []):
            fn = raw_call.get("function") or {}
            arguments = fn.get("arguments", "{}")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False)
            tool_calls.append(
                ToolCall(
                    id=str(raw_call.get("id") or f"call_{index}"),
                    name=str(fn.get("name") or ""),
                    arguments=arguments,
                )
            )

        raw_usage = data.get("usage") or {}
        prompt_tokens = int(raw_usage.get("prompt_tokens") or 0)
        completion_tokens = int(raw_usage.get("completion_tokens") or 0)
        total_tokens = int(raw_usage.get("total_tokens") or 0)

        if total_tokens <= 0:
            prompt_tokens = prompt_tokens or estimate_messages_tokens(messages)
            completion_text = content + "".join(
                f"{call.name}{call.arguments}" for call in tool_calls
            )
            completion_tokens = completion_tokens or estimate_tokens(completion_text)
            total_tokens = prompt_tokens + completion_tokens

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            ),
            model=str(data.get("model") or self.settings.llm_model),
            finish_reason=str(choice.get("finish_reason") or ""),
            reasoning_content=reasoning_content,
        )


class ModelRouter:
    """Route a completion to the provider endpoint for the selected model."""

    def __init__(
        self,
        catalog: ModelCatalog,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.catalog = catalog
        self.settings = settings
        self._transport = transport
        self._clients: dict[str, OpenAICompatLLM] = {}

    def client_for(self, spec: ModelSpec) -> OpenAICompatLLM:
        client = self._clients.get(spec.alias)
        if client is None:
            client_settings = replace(
                self.settings,
                llm_base_url=spec.base_url,
                llm_api_key=spec.api_key,
                llm_model=spec.model_id,
            )
            client = OpenAICompatLLM(
                client_settings,
                transport=self._transport,
                thinking_mode=spec.thinking,
            )
            self._clients[spec.alias] = client
        return client

    async def complete(self, *, model: str | None = None, **kwargs: Any) -> LLMResponse:
        spec = self.catalog.resolve(model)
        return await self.client_for(spec).complete(**kwargs)

    async def close(self) -> None:
        if self._clients:
            await asyncio.gather(
                *(client.close() for client in self._clients.values()),
                return_exceptions=True,
            )
        self._clients.clear()
