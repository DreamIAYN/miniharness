"""Sandbox interface shared by the local and Docker backends."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass(slots=True)
class SandboxSpec:
    command: list[str]
    cwd: Path
    timeout_s: float = 15.0
    env: dict[str, str] = field(default_factory=dict)
    stdin: str | None = None


@dataclass(slots=True)
class SandboxResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    duration_ms: int
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


class Sandbox(Protocol):
    name: str

    async def start(self) -> None: ...

    async def close(self) -> None: ...

    async def run(self, spec: SandboxSpec) -> SandboxResult: ...

    def python_command(self, code: str) -> list[str]: ...

    def shell_command(self, command: str) -> list[str]: ...


async def read_limited(stream: asyncio.StreamReader, limit: int) -> tuple[bytes, bool]:
    """Drain a stream while retaining at most ``limit`` bytes."""
    chunks: list[bytes] = []
    size = 0
    truncated = False
    while True:
        chunk = await stream.read(8192)
        if not chunk:
            break
        remaining = limit - size
        if remaining > 0:
            take = chunk[:remaining]
            chunks.append(take)
            size += len(take)
            if len(take) < len(chunk):
                truncated = True
        else:
            truncated = True
    return b"".join(chunks), truncated


def decode_output(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")
