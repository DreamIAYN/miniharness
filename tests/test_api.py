"""End-to-end API tests with a scripted LLM."""

from __future__ import annotations

import dataclasses
import json

from fastapi.testclient import TestClient

from harness.app import create_app
from harness.config import Settings

from .conftest import ADMIN, FakeLLM, auth, call_tool, create_user, final


def test_health(client_factory):
    client, _ = client_factory([])
    response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["sandbox"] == "local"


def test_admin_requires_token(client_factory):
    client, _ = client_factory([])
    assert client.get("/admin/users").status_code == 401
    assert client.get("/admin/users", headers={"X-Admin-Token": "wrong"}).status_code == 401


def test_admin_ui_served(client_factory):
    client, _ = client_factory([])
    response = client.get("/admin/ui")
    assert response.status_code == 200
    assert "mini-harness" in response.text


def test_user_lifecycle_and_me(client_factory):
    client, _ = client_factory([])
    created = create_user(client, quota=500)
    assert created["api_key"].startswith("hx_")
    user = created["user"]
    assert user["email"] == "alice@example.com"
    assert user["quota_tokens"] == 500
    assert user["remaining_tokens"] == 500
    assert user["active_keys"] == 1

    me = client.get("/v1/me", headers=auth(created["api_key"]))
    assert me.status_code == 200
    assert me.json()["id"] == user["id"]

    usage = client.get("/v1/usage", headers=auth(created["api_key"]))
    assert usage.status_code == 200
    assert usage.json()["summary"]["used_tokens"] == 0

    keys = client.get(f"/admin/users/{user['id']}/keys", headers=ADMIN)
    assert keys.status_code == 200
    assert len(keys.json()) == 1
    assert keys.json()[0]["prefix"].startswith("hx_")


def test_duplicate_email_conflict(client_factory):
    client, _ = client_factory([])
    create_user(client)
    response = client.post(
        "/admin/users",
        headers=ADMIN,
        json={"email": "alice@example.com", "quota_tokens": 10},
    )
    assert response.status_code == 409


def test_agent_run_with_tool(client_factory, settings):
    client, llm = client_factory(
        [
            call_tool(
                "run_python",
                {"code": "open('answer.txt','w').write('42')\nprint(2 + 2)"},
                prompt=10,
                completion=5,
            ),
            final("结果是 4", prompt=20, completion=3),
        ]
    )
    created = create_user(client, quota=1000)
    response = client.post(
        "/v1/agent/run",
        headers=auth(created["api_key"]),
        json={"prompt": "算一下 2+2 并写入文件"},
    )
    assert response.status_code == 200, response.text
    assert llm.calls[0]["max_tokens"] == 1000  # clamped to the user's remaining quota
    body = response.json()
    assert body["output"] == "结果是 4"
    assert body["steps"] == 2
    assert body["usage"] == {"prompt_tokens": 30, "completion_tokens": 8, "total_tokens": 38}
    assert body["remaining_tokens"] == 962
    assert body["stopped_reason"] == "completed"

    # The second model call sees the tool result.
    tool_messages = [m for m in llm.calls[1]["messages"] if m["role"] == "tool"]
    assert tool_messages and "exit_code: 0" in tool_messages[0]["content"]

    # Workspace is cleaned up by default.
    assert list(settings.workspace_root.iterdir()) == []


def test_quota_exceeded_mid_run(client_factory):
    client, _ = client_factory(
        [call_tool("run_python", {"code": "print(1)"}, prompt=10, completion=5)]
    )
    created = create_user(client, quota=5)
    response = client.post(
        "/v1/agent/run",
        headers=auth(created["api_key"]),
        json={"prompt": "run something"},
    )
    assert response.status_code == 402
    error = response.json()["error"]
    assert error["code"] == "quota_exceeded"
    assert error["details"]["quota_tokens"] == 5


def test_quota_blocks_next_run(client_factory):
    client, llm = client_factory([final("done", prompt=6, completion=4)])
    created = create_user(client, quota=10)
    first = client.post("/v1/agent/run", headers=auth(created["api_key"]), json={"prompt": "hello"})
    assert first.status_code == 200
    assert first.json()["remaining_tokens"] == 0

    second = client.post(
        "/v1/agent/run", headers=auth(created["api_key"]), json={"prompt": "again"}
    )
    assert second.status_code == 402
    assert len(llm.calls) == 1  # rejected before calling the model


def test_unlimited_quota(client_factory):
    client, _ = client_factory([final("ok", prompt=1_000_000, completion=1)])
    created = create_user(client, quota=-1)
    response = client.post("/v1/agent/run", headers=auth(created["api_key"]), json={"prompt": "hi"})
    assert response.status_code == 200
    assert response.json()["remaining_tokens"] is None


def test_reset_usage(client_factory):
    client, _ = client_factory([final("done", prompt=6, completion=4)])
    created = create_user(client, quota=100)
    client.post("/v1/agent/run", headers=auth(created["api_key"]), json={"prompt": "hi"})
    usage = client.get("/v1/usage", headers=auth(created["api_key"])).json()
    assert usage["summary"]["used_tokens"] == 10

    reset = client.post(f"/admin/users/{created['user']['id']}/usage/reset", headers=ADMIN)
    assert reset.status_code == 200
    assert reset.json()["used_tokens"] == 0

    usage = client.get("/v1/usage", headers=auth(created["api_key"])).json()
    assert usage["summary"]["used_tokens"] == 0
    # Audit trail is preserved.
    assert len(usage["events"]) == 1


def test_revoked_key_rejected(client_factory):
    client, _ = client_factory([])
    created = create_user(client)
    key_id = client.get(f"/admin/users/{created['user']['id']}/keys", headers=ADMIN).json()[0]["id"]
    revoked = client.delete(f"/admin/users/{created['user']['id']}/keys/{key_id}", headers=ADMIN)
    assert revoked.status_code == 200
    assert client.get("/v1/me", headers=auth(created["api_key"])).status_code == 401


def test_disabled_user_rejected(client_factory):
    client, _ = client_factory([])
    created = create_user(client)
    updated = client.patch(
        f"/admin/users/{created['user']['id']}", headers=ADMIN, json={"status": "disabled"}
    )
    assert updated.status_code == 200
    assert client.get("/v1/me", headers=auth(created["api_key"])).status_code == 403


def test_sse_stream(client_factory):
    client, _ = client_factory([final("streamed", prompt=4, completion=2)])
    created = create_user(client, quota=100)
    with client.stream(
        "POST",
        "/v1/agent/stream",
        headers=auth(created["api_key"]),
        json={"prompt": "hi"},
    ) as response:
        assert response.status_code == 200
        raw = "".join(response.iter_text())
    events = [
        json.loads(line[5:].strip())
        for line in raw.splitlines()
        if line.startswith("data: ") and line[5:].strip()
    ]
    types = [event["type"] for event in events]
    assert "start" in types
    assert "usage" in types
    assert "final" in types
    assert types[-1] == "done"
    assert events[types.index("final")]["output"] == "streamed"


def test_path_traversal_rejected(client_factory, settings):
    client, _ = client_factory(
        [
            call_tool("write_file", {"path": "../escape.txt", "content": "nope"}),
            final("blocked"),
        ]
    )
    created = create_user(client, quota=100)
    with client.stream(
        "POST",
        "/v1/agent/stream",
        headers=auth(created["api_key"]),
        json={"prompt": "escape"},
    ) as response:
        raw = "".join(response.iter_text())
    events = [
        json.loads(line[5:].strip()) for line in raw.splitlines() if line.startswith("data: ")
    ]
    tool_results = [event for event in events if event["type"] == "tool_result"]
    assert tool_results and tool_results[0]["is_error"] is True
    assert "escapes the run workspace" in tool_results[0]["output"]
    assert not tool_results[0]["output"].startswith("ERROR: ERROR:")
    assert not (settings.workspace_root / "escape.txt").exists()


def test_stream_emits_error_event_on_bad_key(client_factory):
    client, _ = client_factory([])
    with client.stream(
        "POST",
        "/v1/agent/stream",
        headers={"Authorization": "Bearer hx_invalid"},
        json={"prompt": "hi"},
    ) as response:
        # Authentication is checked before the stream starts.
        assert response.status_code == 401


def test_llm_not_configured(tmp_path):
    settings = Settings(
        database_path=tmp_path / "db.sqlite",
        workspace_root=tmp_path / "ws",
        admin_token="test-admin",
        sandbox_backend="local",
        llm_api_key="",
    )
    from harness.app import create_app

    with TestClient(create_app(settings)) as client:
        created = create_user(client)
        response = client.post(
            "/v1/agent/run",
            headers=auth(created["api_key"]),
            json={"prompt": "hi"},
        )
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "llm_not_configured"


def test_validation_rejects_empty_prompt(client_factory):
    client, _ = client_factory([])
    created = create_user(client)
    response = client.post("/v1/agent/run", headers=auth(created["api_key"]), json={"prompt": ""})
    assert response.status_code == 422


def test_default_max_tokens_is_deepseek_limit(client_factory):
    client, llm = client_factory([final("ok")])
    created = create_user(client, quota=-1)
    response = client.post("/v1/agent/run", headers=auth(created["api_key"]), json={"prompt": "hi"})
    assert response.status_code == 200, response.text
    assert llm.calls[0]["max_tokens"] == 393_216


def test_max_tokens_override_and_upper_bound(client_factory):
    client, llm = client_factory([final("ok")])
    created = create_user(client, quota=-1)

    accepted = client.post(
        "/v1/agent/run",
        headers=auth(created["api_key"]),
        json={"prompt": "hi", "max_tokens": 393_216},
    )
    assert accepted.status_code == 200
    assert llm.calls[0]["max_tokens"] == 393_216

    rejected = client.post(
        "/v1/agent/run",
        headers=auth(created["api_key"]),
        json={"prompt": "hi", "max_tokens": 393_217},
    )
    assert rejected.status_code == 422


def test_defaults_are_unlimited_and_deepseek_max(tmp_path):
    settings = Settings(
        database_path=tmp_path / "db.sqlite",
        workspace_root=tmp_path / "ws",
        admin_token="x",
    )
    assert settings.agent_run_timeout_s == 0
    assert settings.llm_timeout_s == 0
    assert settings.agent_max_output_tokens == 393_216
    assert settings.agent_context_window_tokens == 1_000_000
    assert settings.agent_max_subagent_depth == 2


def test_subagent_runs_and_usage_is_included(client_factory):
    client, llm = client_factory(
        [
            call_tool(
                "spawn_subagent",
                {"prompt": "子任务：计算 2+2", "max_steps": 3},
                prompt=10,
                completion=5,
            ),
            final("子代理结果：4", prompt=20, completion=5),
            final("父代理汇总：4", prompt=30, completion=5),
        ]
    )
    created = create_user(client, quota=1000)
    response = client.post(
        "/v1/agent/run",
        headers=auth(created["api_key"]),
        json={"prompt": "请派子代理计算 2+2", "max_steps": 4},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["output"] == "父代理汇总：4"
    assert body["steps"] == 2
    assert body["usage"] == {"prompt_tokens": 60, "completion_tokens": 15, "total_tokens": 75}
    assert body["remaining_tokens"] == 925

    # The child uses the subagent system prompt.
    assert "focused subagent" in llm.calls[1]["messages"][0]["content"]
    # The parent sees the child's final result as a tool message.
    tool_messages = [m for m in llm.calls[2]["messages"] if m["role"] == "tool"]
    assert tool_messages and "子代理结果：4" in tool_messages[-1]["content"]

    events = client.get("/v1/usage", headers=auth(created["api_key"])).json()["events"]
    assert len(events) == 3
    assert any(event["run_id"].startswith("sub_") for event in events)

    history = client.get("/admin/usage?limit=20", headers=ADMIN)
    assert history.status_code == 200
    payload = history.json()
    assert payload["total"] == 3
    assert len(payload["items"]) == 3
    assert all(item["email"] == "alice@example.com" for item in payload["items"])
    sub_calls = [item for item in payload["items"] if item["is_subagent"]]
    assert sub_calls and sub_calls[0]["run_id"].startswith("sub_")
    assert payload["items"][0]["key_prefix"].startswith("hx_")

    filtered = client.get(
        "/admin/usage", headers=ADMIN, params={"q": sub_calls[0]["run_id"]}
    ).json()
    assert filtered["total"] == 1
    assert filtered["items"][0]["run_id"] == sub_calls[0]["run_id"]

    assert all(item["has_body"] for item in payload["items"])
    parent_body = client.get(f"/admin/usage/{payload['items'][0]['id']}/body", headers=ADMIN).json()
    assert parent_body["response_text"] == "父代理汇总：4"
    assert parent_body["request_messages"][0]["role"] == "system"
    assert any(
        message.get("role") == "tool" and "子代理结果：4" in str(message.get("content"))
        for message in parent_body["request_messages"]
    )

    child_body = client.get(f"/admin/usage/{sub_calls[0]['id']}/body", headers=ADMIN).json()
    assert child_body["response_text"] == "子代理结果：4"
    assert "focused subagent" in child_body["request_messages"][0]["content"]


def test_subagent_depth_limit(settings):
    custom = dataclasses.replace(settings, agent_max_subagent_depth=0)
    llm = FakeLLM(
        [
            call_tool("spawn_subagent", {"prompt": "delegate"}),
            final("done"),
        ]
    )
    with TestClient(create_app(custom, llm=llm)) as client:
        created = create_user(client)
        with client.stream(
            "POST",
            "/v1/agent/stream",
            headers=auth(created["api_key"]),
            json={"prompt": "delegate"},
        ) as response:
            raw = "".join(response.iter_text())
        events = [
            json.loads(line[5:].strip()) for line in raw.splitlines() if line.startswith("data: ")
        ]
    tool_results = [event for event in events if event["type"] == "tool_result"]
    assert tool_results and tool_results[0]["is_error"] is True
    assert "Maximum subagent depth" in tool_results[0]["output"]
    assert events[-1]["type"] == "done"


def test_context_compaction(settings):
    custom = dataclasses.replace(
        settings,
        agent_context_window_tokens=200,
        agent_context_compress_ratio=0.5,
        agent_context_compress_input_tokens=2000,
        agent_context_summary_max_tokens=50,
    )
    llm = FakeLLM(
        [
            call_tool("run_python", {"code": "print(6 * 7)"}, prompt=80, completion=20),
            final("SUMMARY: 用户要计算 6*7", prompt=10, completion=5),
            final("答案是 42", prompt=100, completion=20),
        ]
    )
    with TestClient(create_app(custom, llm=llm)) as client:
        created = create_user(client, quota=1000)
        with client.stream(
            "POST",
            "/v1/agent/stream",
            headers=auth(created["api_key"]),
            json={
                "prompt": "计算 6*7。" + "请先分析需求，再使用工具，最后核对结果。" * 400,
                "max_steps": 4,
            },
        ) as response:
            raw = "".join(response.iter_text())
        events = [
            json.loads(line[5:].strip()) for line in raw.splitlines() if line.startswith("data: ")
        ]

    compactions = [event for event in events if event["type"] == "compaction"]
    assert compactions
    assert compactions[0]["summarized"] is True
    assert compactions[0]["dropped_messages"] == 1
    assert compactions[0]["after_tokens"] < compactions[0]["before_tokens"]
    assert compactions[0]["usage"]["total_tokens"] == 15

    final_event = next(event for event in events if event["type"] == "final")
    assert final_event["output"] == "答案是 42"
    assert final_event["usage"] == {
        "prompt_tokens": 190,
        "completion_tokens": 45,
        "total_tokens": 235,
    }
    assert final_event["remaining_tokens"] == 765

    # The second call is the summarizer (no tools); the third call sees the
    # compressed summary instead of the original user message.
    assert llm.calls[1]["tools"] is None
    assert any(
        "Compressed context" in str(message.get("content")) for message in llm.calls[2]["messages"]
    )


def test_multimodal_image_input(client_factory):
    client, llm = client_factory([final("红色", prompt=10, completion=2)])
    created = create_user(client, quota=-1)
    image = {"url": "https://example.com/red.png", "detail": "low"}
    response = client.post(
        "/v1/agent/run",
        headers=auth(created["api_key"]),
        json={"prompt": "这张图是什么颜色？", "images": [image]},
    )
    assert response.status_code == 200, response.text

    user_message = llm.calls[0]["messages"][-1]
    assert user_message["role"] == "user"
    content = user_message["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "这张图是什么颜色？"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"] == {"url": image["url"], "detail": "low"}

    item = client.get("/admin/usage?limit=1", headers=ADMIN).json()["items"][0]
    assert item["has_body"] is True
    body = client.get(f"/admin/usage/{item['id']}/body", headers=ADMIN).json()
    logged_content = body["request_messages"][-1]["content"]
    assert logged_content[1]["type"] == "image_url"
    assert logged_content[1]["image_url"]["url"] == image["url"]


def test_multimodal_rejects_non_image_url(client_factory):
    client, _ = client_factory([])
    created = create_user(client)
    response = client.post(
        "/v1/agent/run",
        headers=auth(created["api_key"]),
        json={"prompt": "image", "images": [{"url": "file:///etc/passwd"}]},
    )
    assert response.status_code == 422


def test_multimodal_accepts_data_url(client_factory):
    client, llm = client_factory([final("ok")])
    created = create_user(client, quota=-1)
    data_url = "data:image/png;base64,iVBORw0KGgo="
    response = client.post(
        "/v1/agent/run",
        headers=auth(created["api_key"]),
        json={"prompt": "describe", "images": [{"url": data_url}]},
    )
    assert response.status_code == 200, response.text
    image_part = llm.calls[0]["messages"][-1]["content"][1]
    assert image_part["image_url"]["url"] == data_url


def test_history_supports_content_blocks(client_factory):
    client, llm = client_factory([final("ok")])
    created = create_user(client, quota=-1)
    history = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "previous turn"},
                {"type": "image_url", "image_url": {"url": "https://example.com/prev.png"}},
            ],
        }
    ]
    response = client.post(
        "/v1/agent/run",
        headers=auth(created["api_key"]),
        json={"prompt": "continue", "history": history},
    )
    assert response.status_code == 200, response.text
    assert llm.calls[0]["messages"][1]["content"][1]["type"] == "image_url"
