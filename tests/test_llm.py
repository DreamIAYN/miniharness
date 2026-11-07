"""Unit tests for the OpenAI-compatible client."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from harness.config import Settings
from harness.errors import LLMError, LLMNotConfigured
from harness.llm import OpenAICompatLLM, estimate_messages_tokens, estimate_tokens


def make_llm(tmp_path, handler, **overrides) -> OpenAICompatLLM:
    settings = Settings(
        database_path=tmp_path / "db.sqlite",
        workspace_root=tmp_path / "ws",
        admin_token="test-admin",
        llm_api_key=overrides.pop("llm_api_key", "test-key"),
        llm_model="test-model",
        llm_base_url="http://provider.test/v1",
        llm_max_retries=0,
        **overrides,
    )
    return OpenAICompatLLM(settings, transport=httpx.MockTransport(handler))


def test_parse_tool_calls_and_usage(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"] == "test-model"
        assert body["tools"][0]["function"]["name"] == "run_python"
        assert request.headers["Authorization"] == "Bearer test-key"
        return httpx.Response(
            200,
            json={
                "model": "test-model",
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_9",
                                    "type": "function",
                                    "function": {
                                        "name": "run_python",
                                        "arguments": '{"code": "print(1)"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {
                    "prompt_tokens": 30,
                    "completion_tokens": 12,
                    "total_tokens": 42,
                },
            },
        )

    llm = make_llm(tmp_path, handler)
    try:
        response = asyncio.run(
            llm.complete(
                messages=[{"role": "user", "content": "hi"}],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "run_python",
                            "description": "run",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            )
        )
    finally:
        asyncio.run(llm.close())

    assert response.content == ""
    assert response.tool_calls[0].id == "call_9"
    assert response.tool_calls[0].name == "run_python"
    assert response.usage.total_tokens == 42


def test_usage_is_estimated_when_provider_omits_it(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "你好 hello"}, "finish_reason": "stop"}],
            },
        )

    llm = make_llm(tmp_path, handler)
    try:
        response = asyncio.run(llm.complete(messages=[{"role": "user", "content": "用中文回答"}]))
    finally:
        asyncio.run(llm.close())

    assert response.usage.total_tokens > 0
    assert response.usage.prompt_tokens > 0
    assert response.usage.completion_tokens > 0


def test_provider_error_becomes_llm_error(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "bad request"}})

    llm = make_llm(tmp_path, handler)
    try:
        with pytest.raises(LLMError) as exc_info:
            asyncio.run(llm.complete(messages=[{"role": "user", "content": "hi"}]))
    finally:
        asyncio.run(llm.close())
    assert exc_info.value.details["status"] == 400


def test_missing_choices_is_llm_error(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    llm = make_llm(tmp_path, handler)
    try:
        with pytest.raises(LLMError):
            asyncio.run(llm.complete(messages=[{"role": "user", "content": "hi"}]))
    finally:
        asyncio.run(llm.close())


def test_missing_api_key_raises_not_configured(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("should not call provider")

    llm = make_llm(tmp_path, handler, llm_api_key="")
    try:
        with pytest.raises(LLMNotConfigured):
            asyncio.run(llm.complete(messages=[{"role": "user", "content": "hi"}]))
    finally:
        asyncio.run(llm.close())


def test_estimate_tokens_counts_cjk():
    assert estimate_tokens("你好世界") == 4
    assert estimate_tokens("abcdefgh") == 2
    assert estimate_tokens("") == 0


def test_estimate_messages_tokens_counts_images():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hi"},
                {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
            ],
        }
    ]
    assert estimate_messages_tokens(messages) >= 1024


def test_streaming_completion_emits_deltas_and_usage(tmp_path):
    chunks = [
        {"model": "test-model", "choices": [{"delta": {"content": "he"}}]},
        {"choices": [{"delta": {"reasoning_content": "think"}}]},
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {"name": "run_python", "arguments": '{"code": '},
                            }
                        ]
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"arguments": '"print(1)"}'},
                            }
                        ]
                    }
                }
            ]
        },
        {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}},
    ]
    body = b"".join(f"data: {json.dumps(chunk)}\n\n".encode() for chunk in chunks)
    body += b"data: [DONE]\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["stream"] is True
        assert payload["stream_options"] == {"include_usage": True}
        return httpx.Response(
            200,
            content=body,
            headers={"content-type": "text/event-stream"},
        )

    llm = make_llm(tmp_path, handler)
    deltas: list[dict] = []
    try:
        response = asyncio.run(
            llm.complete(
                messages=[{"role": "user", "content": "run"}],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "run_python",
                            "description": "run",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
                on_delta=deltas.append,
                max_tokens=64,
            )
        )
    finally:
        asyncio.run(llm.close())

    assert response.content == "he"
    assert response.reasoning_content == "think"
    assert response.tool_calls[0].id == "call_1"
    assert response.tool_calls[0].name == "run_python"
    assert response.tool_calls[0].arguments == '{"code": "print(1)"}'
    assert response.usage.total_tokens == 7
    assert {"type": "content", "text": "he"} in deltas
    assert {"type": "reasoning", "text": "think"} in deltas
