"""Local subprocess sandbox.

This backend provides *resource* isolation (wall-clock timeout, CPU, address
space, file size, output limits, sanitised environment, isolated working
directory).  It is **not** a security boundary: the child runs as the same OS
user and can read the host filesystem and use the network.  Use the Docker
backend when running untrusted code.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import sys
import time
from pathlib import Path

from ..config import Settings
from ..errors import SandboxError
from .base import SandboxResult, SandboxSpec, decode_output, read_limited

logger = logging.getLogger(__name__)


class LocalSandbox:
    name = "local"

    def __init__(self, root: Path, settings: Settings) -> None:
        self.root = Path(root).resolve()
        self.settings = settings

    async def start(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    async def close(self) -> None:
        return None

    def python_command(self, code: str) -> list[str]:
        return [sys.executable, "-c", code]

    def shell_command(self, command: str) -> list[str]:
        return ["/bin/sh", "-lc", command]

    def _validate_cwd(self, cwd: Path) -> Path:
        resolved = Path(cwd).resolve()
        if not resolved.is_relative_to(self.root):
            raise SandboxError("Sandbox working directory is outside the workspace root.")
        if not resolved.is_dir():
            raise SandboxError(f"Sandbox working directory does not exist: {resolved}")
        return resolved

    def _build_env(self, cwd: Path, extra: dict[str, str]) -> dict[str, str]:
        tmp = cwd / ".tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        env = {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HOME": str(cwd),
            "TMPDIR": str(tmp),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        for key, value in extra.items():
            if key.replace("_", "").isalnum() and not key[0].isdigit():
                env[key] = str(value)
        return env

    def _preexec(self) -> None:
        """Apply rlimits in the child.  Never raises out of the child."""
        try:
            import resource

            cpu = max(1, int(self.settings.sandbox_cpu_seconds))
            memory = max(32, self.settings.sandbox_memory_mb) * 1024 * 1024
            file_size = max(1024, self.settings.sandbox_max_file_bytes)
            resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
            resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
            resource.setrlimit(resource.RLIMIT_FSIZE, (file_size, file_size))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        except Exception:  # pragma: no cover - best effort in restricted hosts
            pass

    async def run(self, spec: SandboxSpec) -> SandboxResult:
        cwd = self._validate_cwd(spec.cwd)
        env = self._build_env(cwd, spec.env)
        timeout = max(0.1, float(spec.timeout_s))
        started = time.monotonic()

        try:
            proc = await asyncio.create_subprocess_exec(
                *spec.command,
                cwd=str(cwd),
                env=env,
                stdin=asyncio.subprocess.PIPE
                if spec.stdin is not None
                else asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
                preexec_fn=self._preexec,
            )
        except (FileNotFoundError, PermissionError) as exc:
            raise SandboxError(f"Failed to start sandbox process: {exc}") from exc

        assert proc.stdout is not None and proc.stderr is not None
        readers = [
            asyncio.create_task(read_limited(proc.stdout, self.settings.sandbox_max_output_bytes)),
            asyncio.create_task(read_limited(proc.stderr, self.settings.sandbox_max_output_bytes)),
        ]

        async def feed_stdin() -> None:
            assert proc.stdin is not None
            try:
                proc.stdin.write((spec.stdin or "").encode("utf-8"))
                await proc.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with contextlib.suppress(Exception):
                    proc.stdin.close()

        stdin_task = asyncio.create_task(feed_stdin()) if spec.stdin is not None else None
        timed_out = False

        try:
            await asyncio.wait_for(proc.wait(), timeout=timeout)
        except TimeoutError:
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except TimeoutError:
                logger.error("Sandbox process %s did not exit after SIGKILL", proc.pid)

        if stdin_task is not None:
            try:
                await asyncio.wait_for(stdin_task, timeout=2)
            except Exception:
                stdin_task.cancel()

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
