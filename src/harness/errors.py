"""Domain errors with stable machine-readable codes and HTTP mappings."""

from __future__ import annotations

from typing import Any


class HarnessError(Exception):
    status_code = 400
    code = "bad_request"

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        status_code: int | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}
        if status_code is not None:
            self.status_code = status_code
        if code is not None:
            self.code = code

    def to_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details:
            body["details"] = self.details
        return {"error": body}


class AuthError(HarnessError):
    status_code = 401
    code = "unauthorized"


class ForbiddenError(HarnessError):
    status_code = 403
    code = "forbidden"


class NotFoundError(HarnessError):
    status_code = 404
    code = "not_found"


class ConflictError(HarnessError):
    status_code = 409
    code = "conflict"


class QuotaExceeded(HarnessError):
    status_code = 402
    code = "quota_exceeded"

    def __init__(self, *, used: int, quota: int) -> None:
        super().__init__(
            "Token quota exhausted for this user.",
            details={"used_tokens": used, "quota_tokens": quota},
        )
        self.used = used
        self.quota = quota


class LLMError(HarnessError):
    status_code = 502
    code = "llm_error"


class LLMNotConfigured(LLMError):
    status_code = 503
    code = "llm_not_configured"

    def __init__(self) -> None:
        super().__init__(
            "LLM is not configured. Set LLM_API_KEY or OPENCODE_ZEN_API_KEY "
            "(and optionally LLM_BASE_URL / LLM_MODEL)."
        )


class ModelNotFound(HarnessError):
    status_code = 400
    code = "model_not_found"

    def __init__(self, alias: str, available: list[str]) -> None:
        super().__init__(f"Unknown model {alias!r}.", details={"available": available})


class SandboxError(HarnessError):
    status_code = 500
    code = "sandbox_error"


class ToolError(HarnessError):
    status_code = 400
    code = "tool_error"


class RunTimeout(HarnessError):
    status_code = 504
    code = "run_timeout"
