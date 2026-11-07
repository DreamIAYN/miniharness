"""Built-in agent tools.

Shell and Python execute inside the configured sandbox.  File tools operate on
the host side but are strictly confined to the run workspace, so they behave
identically for both sandbox backends.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from ..errors import HarnessError, SandboxError, ToolError
from ..llm import TokenUsage
from ..sandbox.base import SandboxResult, SandboxSpec
from .base import Tool, ToolContext, ToolResult


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… [truncated, {len(text) - limit} chars omitted]"


def _format_result(result: SandboxResult, limit: int) -> str:
    stdout = _clip(result.stdout, limit)
    stderr = _clip(result.stderr, limit)
    parts = [
        f"exit_code: {result.exit_code}",
        f"timed_out: {'true' if result.timed_out else 'false'}",
        f"duration_ms: {result.duration_ms}",
    ]
    if result.truncated:
        parts.append("output_truncated: true")
    parts.append("stdout:\n" + (stdout if stdout else "(empty)"))
    if stderr:
        parts.append("stderr:\n" + stderr)
    return "\n".join(parts)


def _workspace_path(root: Path, raw: str, *, must_exist: bool = False) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ToolError("A non-empty 'path' is required.")
    path = Path(raw)
    if path.is_absolute():
        raise ToolError("Absolute paths are not allowed; use a workspace-relative path.")
    root_resolved = root.resolve()
    candidate = root_resolved / path
    resolved = candidate.resolve(strict=False)
    if not resolved.is_relative_to(root_resolved):
        raise ToolError("Path escapes the run workspace.")
    if must_exist and not resolved.exists():
        raise ToolError(f"Path does not exist: {raw}")
    return resolved


class RunShellTool(Tool):
    name = "run_shell"
    description = (
        "Run a shell command inside the sandbox, with the run workspace as the "
        "working directory. Returns exit code, stdout and stderr. The sandbox "
        "has a wall-clock timeout and output limit."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to execute."},
            "timeout_seconds": {
                "type": "integer",
                "minimum": 1,
                "description": "Optional timeout; capped by the server.",
            },
        },
        "required": ["command"],
        "additionalProperties": False,
    }

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        command = str(args.get("command") or "")
        if not command.strip():
            raise ToolError("'command' must not be empty.")
        if len(command) > 16_000:
            raise ToolError("'command' is too long (max 16000 characters).")
        timeout = float(args.get("timeout_seconds") or ctx.settings.sandbox_default_timeout_s)
        timeout = max(1.0, min(timeout, ctx.settings.sandbox_max_timeout_s))
        try:
            result = await ctx.sandbox.run(
                SandboxSpec(
                    command=ctx.sandbox.shell_command(command),
                    cwd=ctx.workspace,
                    timeout_s=timeout,
                )
            )
        except SandboxError as exc:
            return ToolResult(f"ERROR: {exc.message}", is_error=True)
        return ToolResult(
            content=_format_result(result, ctx.settings.tool_output_chars),
            is_error=not result.ok,
            meta={
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
                "duration_ms": result.duration_ms,
                "truncated": result.truncated,
            },
        )


class RunPythonTool(Tool):
    name = "run_python"
    description = (
        "Execute Python code inside the sandbox with the run workspace as the "
        "working directory. Use this for calculations, data processing and "
        "scripted file work."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python source code to execute."},
            "timeout_seconds": {
                "type": "integer",
                "minimum": 1,
                "description": "Optional timeout; capped by the server.",
            },
        },
        "required": ["code"],
        "additionalProperties": False,
    }

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        code = str(args.get("code") or "")
        if not code.strip():
            raise ToolError("'code' must not be empty.")
        if len(code) > 40_000:
            raise ToolError("'code' is too long (max 40000 characters).")
        timeout = float(args.get("timeout_seconds") or ctx.settings.sandbox_default_timeout_s)
        timeout = max(1.0, min(timeout, ctx.settings.sandbox_max_timeout_s))
        try:
            result = await ctx.sandbox.run(
                SandboxSpec(
                    command=ctx.sandbox.python_command(code),
                    cwd=ctx.workspace,
                    timeout_s=timeout,
                )
            )
        except SandboxError as exc:
            return ToolResult(f"ERROR: {exc.message}", is_error=True)
        return ToolResult(
            content=_format_result(result, ctx.settings.tool_output_chars),
            is_error=not result.ok,
            meta={
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
                "duration_ms": result.duration_ms,
                "truncated": result.truncated,
            },
        )


class ReadFileTool(Tool):
    name = "read_file"
    description = (
        "Read a UTF-8 text file from the run workspace. Returns at most the "
        "configured read limit; use run_shell for binary inspection."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Workspace-relative file path."}},
        "required": ["path"],
        "additionalProperties": False,
    }

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = _workspace_path(ctx.workspace, str(args.get("path") or ""), must_exist=True)
        if not path.is_file():
            raise ToolError(f"Not a regular file: {args.get('path')}")
        limit = ctx.settings.file_read_max_bytes
        with path.open("rb") as handle:
            data = handle.read(limit + 1)
        truncated = len(data) > limit
        text = data[:limit].decode("utf-8", errors="replace")
        suffix = "\n… [truncated by read limit]" if truncated else ""
        return ToolResult(
            content=f"path: {path.relative_to(ctx.workspace.resolve()).as_posix()}\n"
            f"size: {path.stat().st_size} bytes\n---\n{text}{suffix}",
            meta={"bytes": path.stat().st_size, "truncated": truncated},
        )


class WriteFileTool(Tool):
    name = "write_file"
    description = (
        "Create or overwrite a UTF-8 text file inside the run workspace. "
        "Parent directories are created automatically."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Workspace-relative file path."},
            "content": {"type": "string", "description": "Full file content."},
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    }

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        raw_path = str(args.get("path") or "")
        content = args.get("content")
        if not isinstance(content, str):
            raise ToolError("'content' must be a string.")
        data = content.encode("utf-8")
        if len(data) > ctx.settings.file_write_max_bytes:
            raise ToolError(
                f"Refusing to write {len(data)} bytes; limit is "
                f"{ctx.settings.file_write_max_bytes} bytes."
            )
        path = _workspace_path(ctx.workspace, raw_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Re-check after creating parents (guards against symlinked directories).
        if not path.parent.resolve().is_relative_to(ctx.workspace.resolve()):
            raise ToolError("Path escapes the run workspace.")
        path.write_text(content, encoding="utf-8")
        return ToolResult(
            content=f"Wrote {len(data)} bytes to {path.relative_to(ctx.workspace.resolve()).as_posix()}.",
            meta={"bytes": len(data)},
        )


class ListDirTool(Tool):
    name = "list_dir"
    description = "List files and directories inside the run workspace."
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Workspace-relative directory path (default '.').",
                "default": ".",
            }
        },
        "additionalProperties": False,
    }

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = _workspace_path(ctx.workspace, str(args.get("path") or "."), must_exist=True)
        if not path.is_dir():
            raise ToolError(f"Not a directory: {args.get('path')}")
        entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        limit = ctx.settings.file_list_max_entries
        lines = []
        for entry in entries[:limit]:
            try:
                stat = entry.stat()
            except OSError:
                lines.append(f"? {entry.name} (unreadable)")
                continue
            if entry.is_dir():
                lines.append(f"d {entry.name}/")
            elif entry.is_symlink():
                lines.append(f"l {entry.name} -> {entry.readlink()}")
            else:
                lines.append(f"f {entry.name} ({stat.st_size} bytes)")
        if len(entries) > limit:
            lines.append(f"… {len(entries) - limit} more entries")
        return ToolResult(
            content="\n".join(lines) if lines else "(empty directory)",
            meta={"entries": len(entries)},
        )


class SpawnAgentTool(Tool):
    name = "spawn_subagent"
    description = (
        "Delegate a focused, self-contained subtask to a child agent. The child "
        "shares this workspace and the same user quota, runs its own tool loop, "
        "and returns its final result. Use this for genuinely decomposable work; "
        "avoid trivial tasks and repeated delegation. Subagents run sequentially."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Clear, self-contained task for the subagent.",
            },
            "system": {
                "type": "string",
                "description": "Optional system prompt overriding the subagent default.",
            },
            "model": {
                "type": "string",
                "description": "Optional model alias for this subagent; defaults to the parent model.",
            },
            "max_steps": {
                "type": "integer",
                "minimum": 1,
                "description": "Optional step limit; capped by the server.",
            },
        },
        "required": ["prompt"],
        "additionalProperties": False,
    }

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        if ctx.spawn_subagent is None:
            raise ToolError("Subagents are not available in this context.")
        if ctx.depth >= ctx.settings.agent_max_subagent_depth:
            raise ToolError(
                f"Maximum subagent depth ({ctx.settings.agent_max_subagent_depth}) reached; "
                "complete this task directly instead of delegating further."
            )

        prompt = str(args.get("prompt") or "").strip()
        if not prompt:
            raise ToolError("'prompt' must not be empty.")
        if len(prompt) > 32_000:
            raise ToolError("'prompt' is too long (max 32000 characters).")

        system = args.get("system")
        if system is not None:
            system = str(system).strip() or None
            if system is not None and len(system) > 16_000:
                raise ToolError("'system' is too long (max 16000 characters).")

        model = str(args.get("model") or "").strip() or None
        if model is not None and len(model) > 100:
            raise ToolError("'model' is too long (max 100 characters).")

        max_steps = ctx.settings.agent_default_max_steps
        if args.get("max_steps") is not None:
            try:
                max_steps = int(args["max_steps"])
            except (TypeError, ValueError) as exc:
                raise ToolError("'max_steps' must be an integer.") from exc
        max_steps = max(1, min(max_steps, ctx.settings.agent_max_steps_limit))

        child_usage = TokenUsage()
        child_run_id: str | None = None
        final_event: dict[str, Any] | None = None
        try:
            async for event in ctx.spawn_subagent(
                prompt=prompt, system=system, max_steps=max_steps, model=model
            ):
                kind = event.get("type")
                if kind == "start":
                    child_run_id = event.get("run_id")
                elif kind == "usage":
                    child_usage = child_usage + TokenUsage(
                        prompt_tokens=int(event.get("prompt_tokens") or 0),
                        completion_tokens=int(event.get("completion_tokens") or 0),
                        total_tokens=int(event.get("total_tokens") or 0),
                    )
                elif kind == "compaction":
                    usage = event.get("usage") or {}
                    child_usage = child_usage + TokenUsage(
                        prompt_tokens=int(usage.get("prompt_tokens") or 0),
                        completion_tokens=int(usage.get("completion_tokens") or 0),
                        total_tokens=int(usage.get("total_tokens") or 0),
                    )
                elif kind == "final":
                    final_event = event
        except HarnessError as exc:
            return ToolResult(
                f"Subagent failed: {exc.message}",
                is_error=True,
                meta={
                    "subagent_run_id": child_run_id,
                    "subagent_usage": child_usage.as_dict(),
                    "subagent_error": exc.code,
                },
            )

        if final_event is None:
            return ToolResult(
                "Subagent ended without a final answer.",
                is_error=True,
                meta={
                    "subagent_run_id": child_run_id,
                    "subagent_usage": child_usage.as_dict(),
                },
            )

        usage = final_event.get("usage") or child_usage.as_dict()
        output = (
            str(final_event.get("output") or "").strip() or "(subagent returned an empty answer)"
        )
        result_run_id = final_event.get("run_id") or child_run_id
        content = (
            f"subagent_run_id: {result_run_id}\n"
            f"steps: {final_event.get('steps')}\n"
            f"usage: {usage}\n"
            f"stopped_reason: {final_event.get('stopped_reason')}\n\n"
            f"result:\n{_clip(output, ctx.settings.tool_output_chars)}"
        )
        return ToolResult(
            content=content,
            meta={
                "subagent_run_id": result_run_id,
                "subagent_steps": final_event.get("steps"),
                "subagent_usage": usage,
            },
        )
