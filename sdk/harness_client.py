"""Tiny dependency-free Python client for mini-harness.

Copy this file into another project, or add the ``sdk/`` directory to
``PYTHONPATH``.  Only ``httpx`` is required.

    from harness_client import HarnessClient

    with HarnessClient("http://127.0.0.1:8000", "hx_...") as client:
        result = client.run("Inspect the workspace and compute 2+2")
        print(result["output"])
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx


class HarnessError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status = status
        self.code = code
        self.details = details or {}


def _error_from_response(response: httpx.Response) -> HarnessError:
    message = f"HTTP {response.status_code}"
    code = None
    details: dict[str, Any] = {}
    try:
        data = response.json()
    except ValueError:
        data = None
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict):
            message = error.get("message") or message
            code = error.get("code")
            details = error.get("details") or {}
        elif data.get("detail"):
            message = str(data["detail"])
    return HarnessError(message, status=response.status_code, code=code, details=details)


class HarnessClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout: float = 300.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout),
            headers={"Authorization": f"Bearer {api_key}"},
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> HarnessClient:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- user API ----------------------------------------------------------
    def me(self) -> dict[str, Any]:
        response = self._client.get("/v1/me")
        if response.status_code >= 400:
            raise _error_from_response(response)
        return response.json()

    def usage(self, limit: int = 50) -> dict[str, Any]:
        response = self._client.get("/v1/usage", params={"limit": limit})
        if response.status_code >= 400:
            raise _error_from_response(response)
        return response.json()

    def models(self) -> dict[str, Any]:
        response = self._client.get("/v1/models")
        if response.status_code >= 400:
            raise _error_from_response(response)
        return response.json()

    def run(
        self,
        prompt: str,
        *,
        system: str | None = None,
        history: list[dict[str, Any]] | None = None,
        images: list[str | dict[str, Any]] | None = None,
        model: str | None = None,
        max_steps: int | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"prompt": prompt}
        if system is not None:
            payload["system"] = system
        if history:
            payload["history"] = history
        if images:
            payload["images"] = [
                {"url": image} if isinstance(image, str) else image for image in images
            ]
        if model is not None:
            payload["model"] = model
        if max_steps is not None:
            payload["max_steps"] = max_steps
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        response = self._client.post("/v1/agent/run", json=payload)
        if response.status_code >= 400:
            raise _error_from_response(response)
        return response.json()

    def stream(
        self,
        prompt: str,
        *,
        system: str | None = None,
        history: list[dict[str, Any]] | None = None,
        images: list[str | dict[str, Any]] | None = None,
        model: str | None = None,
        max_steps: int | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        payload: dict[str, Any] = {"prompt": prompt}
        if system is not None:
            payload["system"] = system
        if history:
            payload["history"] = history
        if images:
            payload["images"] = [
                {"url": image} if isinstance(image, str) else image for image in images
            ]
        if model is not None:
            payload["model"] = model
        if max_steps is not None:
            payload["max_steps"] = max_steps
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        with self._client.stream("POST", "/v1/agent/stream", json=payload) as response:
            if response.status_code >= 400:
                response.read()
                raise _error_from_response(response)
            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if not raw:
                    continue
                event = json.loads(raw)
                if event.get("type") == "error":
                    raise HarnessError(
                        event.get("message") or "agent error",
                        code=event.get("code"),
                        details=event.get("details") or {},
                    )
                yield event
                if event.get("type") == "done":
                    break


class AdminClient:
    """Programmatic access to the admin API (user + quota management)."""

    def __init__(
        self,
        base_url: str,
        admin_token: str,
        *,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout),
            headers={"X-Admin-Token": admin_token},
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> AdminClient:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        response = self._client.request(method, path, **kwargs)
        if response.status_code >= 400:
            raise _error_from_response(response)
        if response.status_code == 204:
            return None
        return response.json()

    def stats(self) -> dict[str, Any]:
        return self._request("GET", "/admin/stats")

    def list_users(self, *, query: str = "", limit: int = 50, offset: int = 0) -> dict[str, Any]:
        return self._request(
            "GET", "/admin/users", params={"q": query, "limit": limit, "offset": offset}
        )

    def get_user(self, user_id: str) -> dict[str, Any]:
        return self._request("GET", f"/admin/users/{user_id}")

    def create_user(
        self,
        email: str,
        *,
        name: str = "",
        quota_tokens: int | None = None,
        issue_key: bool = True,
        key_name: str = "default",
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "email": email,
            "name": name,
            "issue_key": issue_key,
            "key_name": key_name,
        }
        if quota_tokens is not None:
            payload["quota_tokens"] = quota_tokens
        return self._request("POST", "/admin/users", json=payload)

    def update_user(
        self,
        user_id: str,
        *,
        name: str | None = None,
        quota_tokens: int | None = None,
        status: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if name is not None:
            payload["name"] = name
        if quota_tokens is not None:
            payload["quota_tokens"] = quota_tokens
        if status is not None:
            payload["status"] = status
        return self._request("PATCH", f"/admin/users/{user_id}", json=payload)

    def reset_usage(self, user_id: str) -> dict[str, Any]:
        return self._request("POST", f"/admin/users/{user_id}/usage/reset")

    def usage(self, user_id: str, *, limit: int = 50) -> dict[str, Any]:
        return self._request("GET", f"/admin/users/{user_id}/usage", params={"limit": limit})

    def create_key(self, user_id: str, *, name: str = "default") -> dict[str, Any]:
        return self._request("POST", f"/admin/users/{user_id}/keys", json={"name": name})

    def list_keys(self, user_id: str) -> list[dict[str, Any]]:
        return self._request("GET", f"/admin/users/{user_id}/keys")

    def revoke_key(self, user_id: str, key_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/admin/users/{user_id}/keys/{key_id}")
