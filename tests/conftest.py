"""Shared test fixtures and a scripted fake LLM."""

from __future__ import annotations

import contextlib
import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from harness.app import create_app
from harness.config import Settings
from harness.llm import LLMResponse, TokenUsage, ToolCall

ADMIN = {"X-Admin-Token": "test-admin"}


class FakeLLM:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def complete(self, **kwargs: Any) -> LLMResponse:
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("FakeLLM ran out of scripted responses")
        return self.responses.pop(0)

    async def close(self) -> None:
        return None


def usage(prompt: int = 10, completion: int = 5) -> TokenUsage:
    return TokenUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
    )


def final(text: str, *, prompt: int = 10, completion: int = 5) -> LLMResponse:
    return LLMResponse(
        content=text,
        usage=usage(prompt, completion),
        model="fake-model",
        finish_reason="stop",
    )


def call_tool(
    name: str,
    arguments: dict[str, Any],
    *,
    call_id: str = "call_1",
    prompt: int = 10,
    completion: int = 5,
) -> LLMResponse:
    return LLMResponse(
        content="",
        tool_calls=[
            ToolCall(id=call_id, name=name, arguments=json.dumps(arguments, ensure_ascii=False))
        ],
        usage=usage(prompt, completion),
        model="fake-model",
        finish_reason="tool_calls",
    )


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        database_path=tmp_path / "harness.db",
        workspace_root=tmp_path / "workspaces",
        admin_token="test-admin",
        sandbox_backend="local",
        llm_api_key="test-key",
        llm_model="fake-model",
        llm_base_url="http://fake.local/v1",
        default_quota_tokens=1000,
    )


@pytest.fixture
def client_factory(settings):
    stack = contextlib.ExitStack()

    def make(responses: list[LLMResponse]):
        llm = FakeLLM(responses)
        app = create_app(settings, llm=llm)
        client = stack.enter_context(TestClient(app))
        return client, llm

    yield make
    stack.close()


def create_user(
    client: TestClient,
    *,
    email: str = "alice@example.com",
    quota: int = 1000,
    name: str = "Alice",
) -> dict[str, Any]:
    response = client.post(
        "/admin/users",
        headers=ADMIN,
        json={"email": email, "name": name, "quota_tokens": quota, "key_name": "test"},
    )
    assert response.status_code == 201, response.text
    return response.json()


def auth(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}
