"""Model catalog, router and per-request model selection tests."""

from __future__ import annotations

import asyncio
import dataclasses
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from harness.app import create_app
from harness.config import Settings
from harness.errors import ModelNotFound
from harness.llm import ModelRouter
from harness.models import ModelCatalog

from .conftest import FakeLLM, auth, call_tool, create_user, final


def make_settings(tmp_path) -> Settings:
    return Settings(
        database_path=tmp_path / "db.sqlite",
        workspace_root=tmp_path / "ws",
        admin_token="test-admin",
        sandbox_backend="local",
        llm_base_url="https://direct.test/v1",
        llm_api_key="direct-key",
        llm_model="deepseek-flash",
        llm_default_model="deepseek-v4.1-flash",
        opencode_zen_api_key="zen-key",
        opencode_zen_base_url="https://zen.test/v1",
        opencode_zen_models=("deepseek-v4.1-flash", "glm-5.3-flash"),
    )


def test_catalog_lists_direct_and_zen_models(tmp_path):
    catalog = ModelCatalog.from_settings(make_settings(tmp_path))
    assert catalog.default_alias == "deepseek-v4.1-flash"
    assert set(catalog.aliases()) == {"deepseek-flash", "deepseek-v4.1-flash", "glm-5.3-flash"}

    glm = catalog.resolve("glm-5.3-flash")
    assert glm.provider == "opencode_zen"
    assert glm.context_window == 200_000
    assert glm.max_output_tokens == 131_072
    assert glm.supports_tools is True
    assert glm.supports_vision is True
    assert glm.thinking == "omit"

    with pytest.raises(ModelNotFound):
        catalog.resolve("does-not-exist")


def test_router_uses_provider_specific_endpoint_and_key(tmp_path):
    settings = make_settings(tmp_path)
    catalog = ModelCatalog.from_settings(settings)
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "model": "x",
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    router = ModelRouter(catalog, settings, transport=httpx.MockTransport(handler))
    try:
        asyncio.run(
            router.complete(
                model="glm-5.3-flash",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=100,
            )
        )
        asyncio.run(
            router.complete(
                model="deepseek-flash",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=100,
            )
        )
    finally:
        asyncio.run(router.close())

    zen_request, direct_request = seen
    assert str(zen_request.url) == "https://zen.test/v1/chat/completions"
    assert zen_request.headers["authorization"] == "Bearer zen-key"
    zen_payload = json.loads(zen_request.content)
    assert zen_payload["model"] == "glm-5.3-flash"
    assert "thinking" not in zen_payload

    assert str(direct_request.url) == "https://direct.test/v1/chat/completions"
    assert direct_request.headers["authorization"] == "Bearer direct-key"
    direct_payload = json.loads(direct_request.content)
    assert direct_payload["model"] == "deepseek-flash"
    assert direct_payload["thinking"] == {"type": "enabled"}


def test_models_endpoint_and_model_selection(settings):
    custom = dataclasses.replace(
        settings,
        opencode_zen_api_key="zen-key",
        opencode_zen_base_url="https://zen.test/v1",
        opencode_zen_models=("glm-5.3-flash",),
        llm_default_model="glm-5.3-flash",
    )
    llm = FakeLLM([final("ok")])
    with TestClient(create_app(custom, llm=llm)) as client:
        created = create_user(client, quota=-1)
        models = client.get("/v1/models", headers=auth(created["api_key"]))
        assert models.status_code == 200
        payload = models.json()
        assert payload["default"] == "glm-5.3-flash"
        ids = {item["id"] for item in payload["items"]}
        assert {"fake-model", "glm-5.3-flash"} <= ids
        glm = next(item for item in payload["items"] if item["id"] == "glm-5.3-flash")
        assert glm["provider"] == "opencode_zen"
        assert glm["max_output_tokens"] == 131_072
        assert glm["supports_vision"] is True

        response = client.post(
            "/v1/agent/run",
            headers=auth(created["api_key"]),
            json={"prompt": "hi", "model": "glm-5.3-flash", "max_tokens": 393_216},
        )
        assert response.status_code == 200, response.text
        assert response.json()["model"] == "glm-5.3-flash"
        assert llm.calls[0]["model"] == "glm-5.3-flash"
        assert llm.calls[0]["max_tokens"] == 131_072

        unknown = client.post(
            "/v1/agent/run",
            headers=auth(created["api_key"]),
            json={"prompt": "hi", "model": "does-not-exist"},
        )
        assert unknown.status_code == 400
        assert unknown.json()["error"]["code"] == "model_not_found"


def test_subagent_can_use_a_different_model(settings):
    custom = dataclasses.replace(
        settings,
        opencode_zen_api_key="zen-key",
        opencode_zen_base_url="https://zen.test/v1",
        opencode_zen_models=("glm-5.3-flash",),
        llm_default_model="glm-5.3-flash",
    )
    llm = FakeLLM(
        [
            call_tool("spawn_subagent", {"prompt": "subtask", "model": "glm-5.3-flash"}),
            final("child"),
            final("parent"),
        ]
    )
    with TestClient(create_app(custom, llm=llm)) as client:
        created = create_user(client, quota=-1)
        response = client.post(
            "/v1/agent/run",
            headers=auth(created["api_key"]),
            json={"prompt": "delegate", "model": "fake-model"},
        )
        assert response.status_code == 200, response.text

    assert llm.calls[0]["model"] == "fake-model"
    assert llm.calls[1]["model"] == "glm-5.3-flash"
    assert llm.calls[2]["model"] == "fake-model"
