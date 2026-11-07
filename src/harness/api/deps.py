"""FastAPI dependencies: authentication and app-state access."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import Header, Request

from ..db import Database
from ..errors import AuthError, ForbiddenError
from ..security import hash_api_key, verify_admin_token


@dataclass(slots=True)
class Principal:
    user: dict[str, Any]
    api_key_id: str


def get_db(request: Request) -> Database:
    return request.app.state.db


def get_principal(
    request: Request,
    authorization: str | None = Header(default=None),
) -> Principal:
    if not authorization:
        raise AuthError("Missing Authorization header. Use 'Bearer <api-key>'.")
    scheme, _, raw = authorization.partition(" ")
    if scheme.lower() != "bearer" or not raw.strip():
        raise AuthError("Invalid Authorization header. Use 'Bearer <api-key>'.")

    record = get_db(request).find_key(hash_api_key(raw.strip()))
    if record is None:
        raise AuthError("Invalid API key.")
    if record.get("revoked_at"):
        raise AuthError("This API key has been revoked.")
    if record.get("status") != "active":
        raise ForbiddenError("This user account is disabled.")

    get_db(request).touch_key(record["key_id"])
    user = {
        k: v
        for k, v in record.items()
        if k not in {"key_hash", "key_id", "prefix", "revoked_at", "key_created_at"}
    }
    return Principal(user=user, api_key_id=record["key_id"])


def require_admin(
    request: Request,
    x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
) -> None:
    settings = request.app.state.settings
    if not verify_admin_token(x_admin_token, settings.admin_token):
        raise AuthError("Invalid or missing X-Admin-Token.")
