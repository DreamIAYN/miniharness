"""Live-run registry and API tests."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from harness.app import create_app
from harness.live import LiveRunRegistry

from .conftest import ADMIN, FakeLLM, auth, create_user


def test_live_registry_tracks_partial_and_final_content():
    live = LiveRunRegistry(max_chars=10, ttl_seconds=60)
    live.create(
        run_id="run_live",
        user_id="usr_1",
        email="a@example.com",
        model="glm-5.3-flash",
        depth=0,
    )
    live.start_step("run_live", 1)
    live.add_delta("run_live", "content", "hello ")
    live.add_delta("run_live", "content", "world")
    live.add_delta("run_live", "reasoning", "thinking")

    snapshot = live.snapshot("run_live")
    assert snapshot is not None
    assert snapshot["status"] == "running"
    assert snapshot["content"].startswith("hello")
    assert snapshot["reasoning"] == "thinking"
    assert snapshot["truncated"] is True
    assert snapshot["chars"] <= 70

    live.finish_step(
        "run_live",
        step=1,
        content="hello world",
        reasoning="thinking",
        tool_calls=[],
        usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    )
    live.finish("run_live", status="completed", output="done")
    snapshot = live.snapshot("run_live")
    assert snapshot["status"] == "completed"
    assert snapshot["output"] == "done"
    assert snapshot["steps"][0]["usage"]["total_tokens"] == 3
    assert live.list(include_finished=False) == []
    assert live.list(include_finished=True)[0]["run_id"] == "run_live"


def test_live_run_api_is_scoped_to_owner(settings):
    live = LiveRunRegistry()
    with TestClient(create_app(settings, llm=FakeLLM([]), live=live)) as client:
        alice = create_user(client, email="alice@example.com")
        bob = create_user(client, email="bob@example.com")
        alice_id = alice["user"]["id"]
        live.create(
            run_id="run_live",
            user_id=alice_id,
            email="alice@example.com",
            model="glm-5.3-flash",
            depth=0,
        )
        live.start_step("run_live", 1)
        live.add_delta("run_live", "content", "partial answer")

        listing = client.get("/admin/runs/live", headers=ADMIN)
        assert listing.status_code == 200
        assert listing.json()["items"][0]["run_id"] == "run_live"
        assert listing.json()["items"][0]["content"] == "partial answer"

        admin_detail = client.get("/admin/runs/run_live/live", headers=ADMIN)
        assert admin_detail.status_code == 200
        assert admin_detail.json()["status"] == "running"

        own = client.get("/v1/runs/run_live/live", headers=auth(alice["api_key"]))
        assert own.status_code == 200
        assert own.json()["content"] == "partial answer"

        other = client.get("/v1/runs/run_live/live", headers=auth(bob["api_key"]))
        assert other.status_code == 404

        own_list = client.get("/v1/runs/live", headers=auth(alice["api_key"]))
        assert own_list.status_code == 200
        assert [item["run_id"] for item in own_list.json()["items"]] == ["run_live"]

        live.finish("run_live", status="completed", output="final answer")
        finished = client.get("/admin/runs/live?include_finished=true", headers=ADMIN)
        assert finished.status_code == 200
        assert finished.json()["items"][0]["status"] == "completed"
        assert finished.json()["items"][0]["output"] == "final answer"

        # Finished runs are hidden from the active list.
        assert client.get("/admin/runs/live", headers=ADMIN).json()["items"] == []


def test_live_sse_stream_sends_snapshot_and_done(settings):
    live = LiveRunRegistry()
    with TestClient(create_app(settings, llm=FakeLLM([]), live=live)) as client:
        live.create(
            run_id="run_live",
            user_id="usr_1",
            email="alice@example.com",
            model="glm-5.3-flash",
            depth=0,
        )
        live.start_step("run_live", 1)
        live.add_delta("run_live", "content", "partial answer")
        live.finish_step(
            "run_live",
            step=1,
            content="partial answer",
            reasoning="",
            tool_calls=[],
            usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        )
        live.finish("run_live", status="completed", output="partial answer")

        with client.stream("GET", "/admin/runs/run_live/live/stream", headers=ADMIN) as response:
            assert response.status_code == 200
            raw = "".join(response.iter_text())

    events = [
        json.loads(line[5:].strip())
        for line in raw.splitlines()
        if line.startswith("data: ") and line[5:].strip()
    ]
    assert events[0]["type"] == "snapshot"
    assert events[0]["status"] == "completed"
    assert events[0]["content_delta"] == "partial answer"
    assert events[-1]["type"] == "done"
