"""Tool abstractions and the default registry."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Settings
from ..errors import HarnessError, ToolError
from ..sandbox.base import Sandbox

logger = logging.getLogger(__name__)

SpawnSubagent = Callable[..., AsyncIterator[dict[str, Any]]]


@dataclass(slots=True)
class ToolContext:
    workspace: Path
    sandbox: Sandbox
    settings: Settings
    run_id: str
    depth: int = 0
    spawn_subagent: SpawnSubagent | None = None


@dataclass(slots=True)
class ToolResult:
    content: str
    is_error: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


class Tool(ABC):
    name: str
    description: str
    parameters: dict[str, Any]

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @abstractmethod
    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        raise NotImplementedError


def _parse_args(raw: str | dict[str, Any] | None) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError as exc:
        raise ToolError(f"Invalid JSON arguments: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ToolError("Tool arguments must be a JSON object.")
    return parsed


class ToolRegistry:
    def __init__(self, tools: list[Tool]) -> None:
        self._tools = {tool.name: tool for tool in tools}

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    async def execute(
        self, name: str, raw_arguments: str | dict[str, Any] | None, ctx: ToolContext
    ) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(f"ERROR: unknown tool {name!r}.", is_error=True)
        try:
            args = _parse_args(raw_arguments)
            return await tool.execute(args, ctx)
        except HarnessError as exc:
            return ToolResult(f"ERROR: {exc.message}", is_error=True, meta={"code": exc.code})
        except Exception as exc:  # defensive: a broken tool must not kill the run
            logger.exception("Tool %s failed", name)
            return ToolResult(f"ERROR: tool {name!r} failed: {exc}", is_error=True)


def build_default_registry() -> ToolRegistry:
    from .builtin import (
        ListDirTool,
        ReadFileTool,
        RunPythonTool,
        RunShellTool,
        SpawnAgentTool,
        WriteFileTool,
    )

    return ToolRegistry(
        [
            RunShellTool(),
            RunPythonTool(),
            ReadFileTool(),
            WriteFileTool(),
            ListDirTool(),
            SpawnAgentTool(),
        ]
    )
