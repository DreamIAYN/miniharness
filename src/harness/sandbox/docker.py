"""Docker sandbox: one fresh, network-disabled container per tool call.

The per-run workspace is bind-mounted at ``/workspace``.  Containers run with a
read-only root filesystem, no Linux capabilities, a pids limit, a memory cap
and a CPU cap, and are removed after every call.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
import uuid
from pathlib import Path

from ..config import Settings
from ..errors import SandboxError
from .base import SandboxResult, SandboxSpec, decode_output, read_limited

logger = logging.getLogger(__name__)


class DockerSandbox:
    name = "docker"

    def __init__(self, root: Path, settings: Settings) -> None:
        self.root = Path(root).resolve()
        self.settings = settings
        self._started = False

    async def start(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker",
                "info",
                "--format",
                "{{.ServerVersion}}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, err = await asyncio.wait_for(proc.communicate(), timeout=15)
        except FileNotFoundError as exc:
            raise SandboxError(
                "Docker sandbox selected but the 'docker' CLI was not found. "
                "Install Docker or set HARNESS_SANDBOX_BACKEND=local."
            ) from exc
        except TimeoutError as exc:
            raise SandboxError("Timed out while contacting the Docker daemon.") from exc

        if proc.returncode != 0:
            raise SandboxError(
                "Docker sandbox selected but the daemon is unreachable: "
                + decode_output(err).strip()[:500]
            )
        self._started = True
        logger.info("Docker sandbox ready (image=%s)", self.settings.docker_image)

    async def close(self) -> None:
        return None

    def python_command(self, code: str) -> list[str]:
        return ["python", "-c", code]

    def shell_command(self, command: str) -> list[str]:
        return ["sh", "-lc", command]

    def _validate_cwd(self, cwd: Path) -> str:
        resolved = Path(cwd).resolve()
        if not resolved.is_relative_to(self.root):
            raise SandboxError("Sandbox working directory is outside the workspace root.")
        if not resolved.is_dir():
            raise SandboxError(f"Sandbox working directory does not exist: {resolved}")
        rel = resolved.relative_to(self.root)
        return "/workspace" if str(rel) == "." else f"/workspace/{rel.as_posix()}"

    async def _stop_container(self, name: str) -> None:
        with contextlib.suppress(Exception):  # best-effort cleanup
            proc = await asyncio.create_subprocess_exec(
                "docker",
                "rm",
                "-f",
                name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.wait(), timeout=10)

    async def run(self, spec: SandboxSpec) -> SandboxResult:
        if not self._started:
            await self.start()

        container = f"harness-sbx-{uuid.uuid4().hex[:12]}"
        workdir = self._validate_cwd(spec.cwd)
        timeout = max(0.1, float(spec.timeout_s))
        memory = max(64, self.settings.sandbox_memory_mb)

        args: list[str] = [
            "docker",
            "run",
            "--rm",
            "--name",
            container,
            "--network",
            "bridge" if self.settings.sandbox_allow_network else "none",
            "--memory",
            f"{memory}m",
            "--memory-swap",
            f"{memory}m",
            "--cpus",
            "1",
            "--pids-limit",
            "128",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=64m",
            "-v",
            f"{self.root}:/workspace:rw",
            "-w",
            workdir,
        ]
        if hasattr(os, "getuid"):
            args += ["--user", f"{os.getuid()}:{os.getgid()}"]
        if self.settings.docker_pull:
            args.append("--pull=missing")
        if spec.stdin is not None:
            args.append("-i")

        env = {
            "HOME": "/workspace",
            "TMPDIR": "/tmp",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            **{str(k): str(v) for k, v in spec.env.items()},
        }
        for key, value in env.items():
            args += ["-e", f"{key}={value}"]

        args += [self.settings.docker_image, *spec.command]

        started = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdin=asyncio.subprocess.PIPE
                if spec.stdin is not None
                else asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise SandboxError("Docker CLI not found.") from exc

        assert proc.stdout is not None and proc.stderr is not None
        readers = [
            asyncio.create_task(read_limited(proc.stdout, self.settings.sandbox_max_output_bytes)),
            asyncio.create_task(read_limited(proc.stderr, self.settings.sandbox_max_output_bytes)),
        ]

        stdin_task: asyncio.Task | None = None
        if spec.stdin is not None:
            assert proc.stdin is not None

            async def feed() -> None:
                try:
                    proc.stdin.write((spec.stdin or "").encode("utf-8"))
                    await proc.stdin.drain()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    with contextlib.suppress(Exception):
                        proc.stdin.close()

            stdin_task = asyncio.create_task(feed())

        timed_out = False
        try:
            await asyncio.wait_for(proc.wait(), timeout=timeout)
        except TimeoutError:
            timed_out = True
            await self._stop_container(container)
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except TimeoutError:
                logger.error("Docker CLI %s did not exit after kill", proc.pid)

        if stdin_task is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(stdin_task, timeout=2)

        results = await asyncio.gather(*readers, return_exceptions=True)
        stdout_bytes, stderr_bytes = b"", b""
        truncated = False
        if isinstance(results[0], tuple):
            stdout_bytes, t1 = results[0]
            truncated = truncated or t1
        if isinstance(results[1], tuple):
            stderr_bytes, t2 = results[1]
            truncated = truncated or t2

        return SandboxResult(
            exit_code=proc.returncode if proc.returncode is not None else -1,
            stdout=decode_output(stdout_bytes),
            stderr=decode_output(stderr_bytes),
            timed_out=timed_out,
            duration_ms=int((time.monotonic() - started) * 1000),
            truncated=truncated,
        )
