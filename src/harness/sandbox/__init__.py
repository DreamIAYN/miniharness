"""Sandbox backend selection."""

from __future__ import annotations

from pathlib import Path

from ..config import Settings
from ..errors import SandboxError
from .base import Sandbox
from .docker import DockerSandbox
from .local import LocalSandbox


def create_sandbox(settings: Settings) -> Sandbox:
    root = Path(settings.workspace_root)
    if settings.sandbox_backend == "local":
        return LocalSandbox(root, settings)
    if settings.sandbox_backend == "docker":
        return DockerSandbox(root, settings)
    raise SandboxError(f"Unknown sandbox backend: {settings.sandbox_backend!r}")
