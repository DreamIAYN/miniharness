"""Runtime configuration, loaded from environment variables.

Everything has a development-friendly default except the LLM API key and the
admin token.  The admin token is generated on first start when missing so the
service is usable out of the box; set ``HARNESS_ADMIN_TOKEN`` in production.
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from pathlib import Path

DEFAULT_OPENCODE_ZEN_MODELS = (
    "deepseek-v4.1-flash",
    "deepseek-v4-pro",
    "glm-5.3-flash",
    "glm-5.3",
    "minimax-m3",
)


def _load_dotenv(path: str = ".env") -> None:
    """Tiny .env loader so the project needs no extra dependency."""
    p = Path(path)
    if not p.is_file():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    return int(raw) if raw else default


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    return float(raw) if raw else default


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_path(name: str, default: str) -> Path:
    return Path(_env(name, default)).expanduser()


@dataclass(slots=True)
class Settings:
    # Server
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: tuple[str, ...] = ("*",)
    log_level: str = "INFO"

    # Storage
    database_path: Path = Path("data/harness.db")
    workspace_root: Path = Path("data/workspaces")
    keep_workspaces: bool = False

    # Admin
    admin_token: str = ""
    admin_token_generated: bool = False

    # LLM (any OpenAI-compatible chat-completions endpoint)
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_api_key: str = ""
    llm_model: str = "deepseek-flash"
    # Request-facing default model alias. Empty means llm_model.
    llm_default_model: str = ""
    llm_timeout_s: float = 0.0  # 0 = no per-request timeout
    llm_max_retries: int = 2
    llm_thinking: bool = True
    llm_reasoning_effort: str = "high"

    # OpenCode Zen (OpenAI-compatible chat endpoint).
    opencode_zen_api_key: str = ""
    opencode_zen_base_url: str = "https://opencode.ai/inference/openai/v1"
    opencode_zen_models: tuple[str, ...] = DEFAULT_OPENCODE_ZEN_MODELS

    # Agent
    agent_default_max_steps: int = 8
    agent_max_steps_limit: int = 20
    agent_run_timeout_s: float = 0.0  # 0 = no wall-clock limit for a run
    # DeepSeek's maximum output is 384K tokens (393216); lower it for other providers.
    agent_max_output_tokens: int = 393216
    agent_temperature: float = 0.2
    agent_max_subagent_depth: int = 2

    # Context compaction. 0 window disables it.
    agent_context_window_tokens: int = 1_000_000
    agent_context_compress_ratio: float = 0.8
    agent_context_compress_input_tokens: int = 32_000
    agent_context_summary_max_tokens: int = 1024

    # Call logs store prompts/responses separately from the usage ledger.
    call_log_enabled: bool = True
    call_log_retention_days: int = 7
    call_log_max_chars: int = 0  # 0 = unlimited per field

    # Tools
    tool_output_chars: int = 8000
    file_read_max_bytes: int = 128 * 1024
    file_write_max_bytes: int = 256 * 1024
    file_list_max_entries: int = 200

    # Sandbox, backend independent
    sandbox_backend: str = "local"
    sandbox_default_timeout_s: float = 15.0
    sandbox_max_timeout_s: float = 60.0
    sandbox_memory_mb: int = 256
    sandbox_cpu_seconds: int = 10
    sandbox_max_output_bytes: int = 64 * 1024
    sandbox_max_file_bytes: int = 8 * 1024 * 1024

    # Sandbox, docker backend
    docker_image: str = "python:3.12-slim"
    docker_pull: bool = True
    sandbox_allow_network: bool = False

    # Defaults applied to newly created users (quota is reset per user).
    default_quota_tokens: int = 100_000

    def __post_init__(self) -> None:
        self.sandbox_backend = self.sandbox_backend.lower()
        if self.sandbox_backend not in {"local", "docker"}:
            raise ValueError("HARNESS_SANDBOX_BACKEND must be 'local' or 'docker'")
        if self.agent_max_steps_limit < 1:
            raise ValueError("agent_max_steps_limit must be >= 1")
        if self.agent_max_subagent_depth < 0:
            raise ValueError("agent_max_subagent_depth must be >= 0")
        if not 0 < self.agent_context_compress_ratio <= 1:
            raise ValueError("agent_context_compress_ratio must be in (0, 1]")
        if self.call_log_retention_days < 0:
            raise ValueError("call_log_retention_days must be >= 0")
        if self.call_log_max_chars < 0:
            raise ValueError("call_log_max_chars must be >= 0")

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_api_key and self.llm_model and self.llm_base_url)

    @property
    def admin_auth_enabled(self) -> bool:
        return bool(self.admin_token)

    @classmethod
    def from_env(cls) -> Settings:
        _load_dotenv()

        admin_token = _env("HARNESS_ADMIN_TOKEN")
        generated = False
        if not admin_token:
            admin_token = secrets.token_urlsafe(32)
            generated = True

        origins_raw = _env("HARNESS_CORS_ORIGINS", "*")
        origins = tuple(o.strip() for o in origins_raw.split(",") if o.strip())

        settings = cls(
            host=_env("HARNESS_HOST", "127.0.0.1"),
            port=_env_int("HARNESS_PORT", 8000),
            cors_origins=origins or ("*",),
            log_level=_env("HARNESS_LOG_LEVEL", "INFO").upper(),
            database_path=_env_path("HARNESS_DB_PATH", "data/harness.db"),
            workspace_root=_env_path("HARNESS_WORKSPACE_ROOT", "data/workspaces"),
            keep_workspaces=_env_bool("HARNESS_KEEP_WORKSPACES", False),
            admin_token=admin_token,
            admin_token_generated=generated,
            llm_base_url=_env("LLM_BASE_URL", "https://api.deepseek.com/v1"),
            llm_api_key=_env("LLM_API_KEY"),
            llm_model=_env("LLM_MODEL", "deepseek-flash"),
            llm_default_model=_env("LLM_DEFAULT_MODEL"),
            llm_timeout_s=_env_float("LLM_TIMEOUT_SECONDS", 0.0),
            llm_max_retries=_env_int("LLM_MAX_RETRIES", 2),
            llm_thinking=_env_bool("LLM_THINKING", True),
            llm_reasoning_effort=_env("LLM_REASONING_EFFORT", "high"),
            opencode_zen_api_key=_env("OPENCODE_ZEN_API_KEY"),
            opencode_zen_base_url=_env(
                "OPENCODE_ZEN_BASE_URL", "https://opencode.ai/inference/openai/v1"
            ),
            opencode_zen_models=(
                tuple(
                    model.strip()
                    for model in _env("OPENCODE_ZEN_MODELS").split(",")
                    if model.strip()
                )
                or DEFAULT_OPENCODE_ZEN_MODELS
            ),
            agent_default_max_steps=_env_int("AGENT_DEFAULT_MAX_STEPS", 8),
            agent_max_steps_limit=_env_int("AGENT_MAX_STEPS_LIMIT", 20),
            agent_run_timeout_s=_env_float("AGENT_RUN_TIMEOUT_SECONDS", 0.0),
            agent_max_output_tokens=_env_int("AGENT_MAX_OUTPUT_TOKENS", 393216),
            agent_max_subagent_depth=_env_int("AGENT_MAX_SUBAGENT_DEPTH", 2),
            agent_context_window_tokens=_env_int("AGENT_CONTEXT_WINDOW_TOKENS", 1_000_000),
            agent_context_compress_ratio=_env_float("AGENT_CONTEXT_COMPRESS_RATIO", 0.8),
            agent_context_compress_input_tokens=_env_int(
                "AGENT_CONTEXT_COMPRESS_INPUT_TOKENS", 32_000
            ),
            agent_context_summary_max_tokens=_env_int("AGENT_CONTEXT_SUMMARY_MAX_TOKENS", 1024),
            agent_temperature=_env_float("AGENT_TEMPERATURE", 0.2),
            call_log_enabled=_env_bool("CALL_LOG_ENABLED", True),
            call_log_retention_days=_env_int("CALL_LOG_RETENTION_DAYS", 7),
            call_log_max_chars=_env_int("CALL_LOG_MAX_CHARS", 0),
            tool_output_chars=_env_int("TOOL_OUTPUT_CHARS", 8000),
            file_read_max_bytes=_env_int("FILE_READ_MAX_BYTES", 128 * 1024),
            file_write_max_bytes=_env_int("FILE_WRITE_MAX_BYTES", 256 * 1024),
            file_list_max_entries=_env_int("FILE_LIST_MAX_ENTRIES", 200),
            sandbox_backend=_env("HARNESS_SANDBOX_BACKEND", "local"),
            sandbox_default_timeout_s=_env_float("SANDBOX_DEFAULT_TIMEOUT_SECONDS", 15.0),
            sandbox_max_timeout_s=_env_float("SANDBOX_MAX_TIMEOUT_SECONDS", 60.0),
            sandbox_memory_mb=_env_int("SANDBOX_MEMORY_MB", 256),
            sandbox_cpu_seconds=_env_int("SANDBOX_CPU_SECONDS", 10),
            sandbox_max_output_bytes=_env_int("SANDBOX_MAX_OUTPUT_BYTES", 64 * 1024),
            sandbox_max_file_bytes=_env_int("SANDBOX_MAX_FILE_BYTES", 8 * 1024 * 1024),
            docker_image=_env("SANDBOX_DOCKER_IMAGE", "python:3.12-slim"),
            docker_pull=_env_bool("SANDBOX_DOCKER_PULL", True),
            sandbox_allow_network=_env_bool("SANDBOX_ALLOW_NETWORK", False),
            default_quota_tokens=_env_int("DEFAULT_QUOTA_TOKENS", 100_000),
        )
        return settings
