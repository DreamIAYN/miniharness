"""API-key and admin-token helpers.

API keys are shown once and only their SHA-256 digest is persisted.  Admin
authentication uses a single bearer-style shared secret from the environment;
this is deliberately simple for a small internal deployment.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets

API_KEY_PREFIX = "hx_"


def generate_api_key() -> str:
    return API_KEY_PREFIX + secrets.token_urlsafe(32)


def hash_api_key(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def key_prefix(raw: str) -> str:
    return raw[:12]


def verify_admin_token(provided: str | None, expected: str) -> bool:
    if not provided or not expected:
        return False
    return hmac.compare_digest(provided, expected)
