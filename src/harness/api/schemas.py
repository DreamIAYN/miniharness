"""Request/response models for the HTTP API."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

_IMAGE_DATA_URL = re.compile(r"^data:image/(?:jpeg|png|gif|webp);base64,", re.IGNORECASE)


# -- shared ---------------------------------------------------------------
class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str | list[dict[str, Any]] = Field(
        description="Text, or OpenAI-style content blocks (images only in user messages)."
    )

    @field_validator("content")
    @classmethod
    def _check_content(cls, value: str | list[dict[str, Any]]) -> str | list[dict[str, Any]]:
        if isinstance(value, str):
            if not value.strip():
                raise ValueError("history content must not be empty")
            if len(value) > 16_000:
                raise ValueError("history text content is too long (max 16000 characters)")
            return value
        if not value:
            raise ValueError("history content blocks must not be empty")
        return value


class ImageInput(BaseModel):
    url: str = Field(
        min_length=1,
        max_length=45_000_000,
        description="data:image/(jpeg|png|gif|webp);base64,... or a public http(s) URL",
    )
    detail: Literal["low", "high", "original", "auto"] = "auto"

    @field_validator("url")
    @classmethod
    def _validate_url(cls, value: str) -> str:
        value = value.strip()
        if _IMAGE_DATA_URL.match(value):
            return value
        if value.startswith(("http://", "https://")):
            if len(value) > 8192:
                raise ValueError("external image URL must be at most 8192 characters")
            return value
        raise ValueError("image url must be a data:image/...;base64 URL or a public http(s) URL")


class RunRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=32_000)
    system: str | None = Field(default=None, max_length=16_000)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=100)
    images: list[ImageInput] = Field(default_factory=list, max_length=10)
    model: str | None = Field(default=None, max_length=100)
    max_steps: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=16, le=393_216)


class TokenUsageOut(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class RunResponse(BaseModel):
    run_id: str
    model: str | None = None
    output: str
    steps: int
    usage: TokenUsageOut
    remaining_tokens: int | None
    stopped_reason: str


class ModelOut(BaseModel):
    id: str
    provider: str
    model: str
    context_window: int
    max_output_tokens: int
    supports_tools: bool
    supports_vision: bool
    label: str


class ModelListOut(BaseModel):
    default: str | None
    items: list[ModelOut]


class LiveRunOut(BaseModel):
    run_id: str
    user_id: str
    email: str
    model: str
    depth: int
    is_subagent: bool
    status: str
    current_step: int
    content: str
    reasoning: str
    tool_calls: list[dict[str, Any]]
    steps: list[dict[str, Any]]
    output: str
    error: str
    truncated: bool
    started_at: str
    updated_at: str
    elapsed_ms: int
    chars: int


class LiveRunListOut(BaseModel):
    items: list[LiveRunOut]


# -- users ----------------------------------------------------------------
class UserCreate(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    name: str = Field(default="", max_length=100)
    quota_tokens: int | None = Field(default=None, ge=-1)
    issue_key: bool = True
    key_name: str = Field(default="default", max_length=60)

    @field_validator("email")
    @classmethod
    def _check_email(cls, value: str) -> str:
        value = value.strip().lower()
        if "@" not in value or " " in value or value.startswith("@") or value.endswith("@"):
            raise ValueError("invalid email address")
        return value


class UserUpdate(BaseModel):
    name: str | None = Field(default=None, max_length=100)
    quota_tokens: int | None = Field(default=None, ge=-1)
    status: Literal["active", "disabled"] | None = None


class KeyCreate(BaseModel):
    name: str = Field(default="default", max_length=60)


class QuotaSummary(BaseModel):
    user_id: str
    quota_tokens: int
    used_tokens: int
    remaining_tokens: int | None
    unlimited: bool
    period_start: str


class UserOut(BaseModel):
    id: str
    email: str
    name: str
    status: str
    quota_tokens: int
    used_tokens: int
    remaining_tokens: int | None
    unlimited: bool
    active_keys: int = 0
    created_at: str
    updated_at: str
    quota_period_start: str

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> UserOut:
        quota = int(record["quota_tokens"])
        used = int(record.get("used_tokens") or 0)
        unlimited = quota < 0
        return cls(
            id=record["id"],
            email=record["email"],
            name=record["name"],
            status=record["status"],
            quota_tokens=quota,
            used_tokens=used,
            remaining_tokens=None if unlimited else max(0, quota - used),
            unlimited=unlimited,
            active_keys=int(record.get("active_keys") or 0),
            created_at=record["created_at"],
            updated_at=record["updated_at"],
            quota_period_start=record["quota_period_start"],
        )


class UserCreatedOut(BaseModel):
    user: UserOut
    api_key: str | None = None


class UserListOut(BaseModel):
    items: list[UserOut]
    total: int


class KeyOut(BaseModel):
    id: str
    prefix: str
    name: str
    created_at: str
    last_used_at: str | None = None
    revoked_at: str | None = None


class KeyCreatedOut(BaseModel):
    api_key: str
    key: KeyOut


class UsageEventOut(BaseModel):
    id: int
    run_id: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    created_at: str


class UsageOut(BaseModel):
    summary: QuotaSummary
    events: list[UsageEventOut]


class CallEventOut(BaseModel):
    id: int
    run_id: str
    user_id: str
    email: str
    name: str
    key_prefix: str | None = None
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    created_at: str
    is_subagent: bool = False
    has_body: bool = False


class CallBodyOut(BaseModel):
    usage_event_id: int
    run_id: str
    email: str
    name: str
    model: str
    created_at: str
    request_messages: list[dict[str, Any]]
    request_text: str | None = None
    response_text: str
    tool_calls: list[dict[str, Any]]
    reasoning_text: str
    truncated: bool


class CallHistoryOut(BaseModel):
    items: list[CallEventOut]
    total: int
    limit: int
    offset: int


class StatsOut(BaseModel):
    users: int
    active_users: int
    active_keys: int
    total_tokens: int
    tokens_today: int
    runs: int
