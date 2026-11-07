"""Sandbox and tool tests (local backend)."""

from __future__ import annotations

import asyncio
import shutil

import pytest

from harness.config import Settings
from harness.errors import SandboxError, ToolError
from harness.sandbox.base import SandboxSpec
from harness.sandbox.local import LocalSandbox
from harness.tools.base import ToolContext, build_default_registry
from harness.tools.builtin import _workspace_path


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        database_path=tmp_path / "db.sqlite",
        workspace_root=tmp_path / "ws",
        admin_token="test-admin",
        sandbox_backend="local",
        sandbox_max_output_bytes=100,
        sandbox_memory_mb=256,
        sandbox_cpu_seconds=5,
    )


@pytest.fixture
def sandbox(settings, tmp_path):
    ws = tmp_path / "ws" / "run_test"
    ws.mkdir(parents=True)
    instance = LocalSandbox(settings.workspace_root, settings)
    run(instance.start())
    return instance, ws


def test_shell_echo(sandbox):
    instance, ws = sandbox
    result = run(
        instance.run(SandboxSpec(command=["/bin/sh", "-c", "echo hello; echo err >&2"], cwd=ws))
    )
    assert result.ok
    assert result.stdout.strip() == "hello"
    assert result.stderr.strip() == "err"


def test_timeout_kills_process(sandbox):
    instance, ws = sandbox
    result = run(
        instance.run(SandboxSpec(command=["/bin/sh", "-c", "sleep 5"], cwd=ws, timeout_s=0.4))
    )
    assert result.timed_out
    assert result.duration_ms < 3000


def test_output_truncation(sandbox):
    instance, ws = sandbox
    result = run(
        instance.run(
            SandboxSpec(
                command=["python3", "-c", "print('x' * 10000)"],
                cwd=ws,
                timeout_s=10,
            )
        )
    )
    assert result.truncated
    assert len(result.stdout) == 100


def test_host_environment_is_not_inherited(sandbox, monkeypatch):
    instance, ws = sandbox
    monkeypatch.setenv("HOST_SECRET_VALUE", "leaked")
    result = run(
        instance.run(
            SandboxSpec(
                command=["/bin/sh", "-c", "printf '%s' \"${HOST_SECRET_VALUE:-missing}\""],
                cwd=ws,
            )
        )
    )
    assert result.stdout == "missing"


def test_cwd_outside_root_rejected(sandbox, tmp_path):
    instance, _ = sandbox
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(SandboxError):
        run(instance.run(SandboxSpec(command=["/bin/sh", "-c", "true"], cwd=outside)))


def test_python_roundtrip(sandbox):
    instance, ws = sandbox
    result = run(
        instance.run(
            SandboxSpec(
                command=instance.python_command("print(6 * 7)"),
                cwd=ws,
                timeout_s=10,
            )
        )
    )
    assert result.ok
    assert result.stdout.strip() == "42"


def test_tool_registry_write_read_list(settings, sandbox):
    instance, ws = sandbox
    registry = build_default_registry()
    ctx = ToolContext(workspace=ws, sandbox=instance, settings=settings, run_id="run_test")

    written = run(
        registry.execute("write_file", {"path": "sub/hello.txt", "content": "hi there"}, ctx)
    )
    assert not written.is_error
    read = run(registry.execute("read_file", {"path": "sub/hello.txt"}, ctx))
    assert not read.is_error
    assert "hi there" in read.content
    listed = run(registry.execute("list_dir", {"path": "sub"}, ctx))
    assert "hello.txt" in listed.content


def test_tool_path_escape_rejected(settings, sandbox):
    instance, ws = sandbox
    registry = build_default_registry()
    ctx = ToolContext(workspace=ws, sandbox=instance, settings=settings, run_id="run_test")
    result = run(registry.execute("read_file", {"path": "../../etc/passwd"}, ctx))
    assert result.is_error
    assert "escapes" in result.content


def test_absolute_path_rejected(settings, sandbox):
    _instance, ws = sandbox
    with pytest.raises(ToolError):
        _workspace_path(ws, "/etc/passwd")


def test_unknown_tool_returns_error(settings, sandbox):
    instance, ws = sandbox
    registry = build_default_registry()
    ctx = ToolContext(workspace=ws, sandbox=instance, settings=settings, run_id="run_test")
    result = run(registry.execute("does_not_exist", {}, ctx))
    assert result.is_error
    assert "unknown tool" in result.content


def docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        import subprocess

        return subprocess.run(["docker", "info"], capture_output=True, timeout=5).returncode == 0
    except Exception:
        return False


@pytest.mark.skipif(not docker_available(), reason="Docker daemon is not available")
def test_docker_backend_runs_isolated(settings, tmp_path):
    from harness.sandbox.docker import DockerSandbox

    ws = tmp_path / "ws" / "run_docker"
    ws.mkdir(parents=True)
    instance = DockerSandbox(settings.workspace_root, settings)
    run(instance.start())
    result = run(
        instance.run(SandboxSpec(command=["python", "-c", "print(1 + 1)"], cwd=ws, timeout_s=120))
    )
    assert result.ok
    assert result.stdout.strip() == "2"


def test_python_sympy_available(sandbox):
    instance, ws = sandbox
    result = run(
        instance.run(
            SandboxSpec(
                command=instance.python_command(
                    "import sympy as sp\n"
                    "print('sympy', sp.__version__)\n"
                    "x = sp.symbols('x')\n"
                    "print(sp.factor(x**2 - 1))\n"
                ),
                cwd=ws,
                timeout_s=30,
            )
        )
    )
    assert result.ok
    assert "sympy 1." in result.stdout
    assert "(x - 1)*(x + 1)" in result.stdout
