"""SQLite persistence for users, API keys and the token-usage ledger.

The usage ledger is append-only.  ``users.quota_period_start`` is the reset
watermark: usage is the sum of events created at or after it.  Resetting a
user's quota therefore keeps the audit trail intact.
"""

from __future__ import annotations

import sqlite3
import threading
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .errors import ConflictError, NotFoundError
from .security import generate_api_key, hash_api_key, key_prefix


def utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id                 TEXT PRIMARY KEY,
    email              TEXT NOT NULL UNIQUE,
    name               TEXT NOT NULL DEFAULT '',
    status             TEXT NOT NULL DEFAULT 'active'
                       CHECK (status IN ('active', 'disabled')),
    quota_tokens       INTEGER NOT NULL DEFAULT 100000,
    quota_period_start TEXT NOT NULL,
    created_at         TEXT NOT NULL,
    updated_at         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS api_keys (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    key_hash    TEXT NOT NULL UNIQUE,
    prefix      TEXT NOT NULL,
    name        TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL,
    last_used_at TEXT,
    revoked_at  TEXT
);

CREATE TABLE IF NOT EXISTS usage_events (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id           TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    api_key_id        TEXT,
    run_id            TEXT NOT NULL,
    model             TEXT NOT NULL,
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens      INTEGER NOT NULL DEFAULT 0,
    created_at        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_usage_user_time
    ON usage_events (user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_usage_run
    ON usage_events (run_id);
CREATE INDEX IF NOT EXISTS idx_keys_user
    ON api_keys (user_id);

CREATE TABLE IF NOT EXISTS call_logs (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    usage_event_id INTEGER NOT NULL UNIQUE REFERENCES usage_events(id) ON DELETE CASCADE,
    user_id        TEXT NOT NULL,
    api_key_id     TEXT,
    run_id         TEXT NOT NULL,
    request_json   TEXT NOT NULL,
    response_text  TEXT NOT NULL DEFAULT '',
    tool_calls_json TEXT NOT NULL DEFAULT '[]',
    reasoning_text TEXT NOT NULL DEFAULT '',
    truncated      INTEGER NOT NULL DEFAULT 0,
    created_at     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_call_logs_time
    ON call_logs (created_at);
"""

# Columns an administrator may change.  Kept explicit to avoid SQL injection.
_UPDATABLE_USER_FIELDS = {"name", "status", "quota_tokens"}


class Database:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA busy_timeout = 5000")

    # -- lifecycle ---------------------------------------------------------
    def init(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript(SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- users -------------------------------------------------------------
    def create_user(
        self,
        *,
        email: str,
        name: str = "",
        quota_tokens: int = 100_000,
    ) -> dict[str, Any]:
        now = utcnow()
        user_id = _new_id("usr")
        try:
            with self._lock, self._conn:
                self._conn.execute(
                    """INSERT INTO users
                       (id, email, name, status, quota_tokens, quota_period_start,
                        created_at, updated_at)
                       VALUES (?, ?, ?, 'active', ?, ?, ?, ?)""",
                    (user_id, email.lower().strip(), name.strip(), quota_tokens, now, now, now),
                )
        except sqlite3.IntegrityError as exc:
            raise ConflictError(f"A user with email {email!r} already exists.") from exc
        user = self.get_user(user_id)
        assert user is not None
        return user

    def get_user(self, user_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None

    def get_user_by_email(self, email: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM users WHERE email = ?", (email.lower().strip(),)
            ).fetchone()
        return dict(row) if row else None

    def update_user(self, user_id: str, **fields: Any) -> dict[str, Any]:
        updates = {k: v for k, v in fields.items() if k in _UPDATABLE_USER_FIELDS and v is not None}
        if not updates:
            raise ValueError("no updatable fields supplied")
        updates["updated_at"] = utcnow()
        assignments = ", ".join(f"{key} = ?" for key in updates)
        values = [*updates.values(), user_id]
        with self._lock, self._conn:
            cur = self._conn.execute(f"UPDATE users SET {assignments} WHERE id = ?", values)
            if cur.rowcount == 0:
                raise NotFoundError(f"User {user_id!r} not found.")
        user = self.get_user(user_id)
        assert user is not None
        return user

    def get_user_with_usage(self, user_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT u.*,
                       COALESCE((
                           SELECT SUM(e.total_tokens) FROM usage_events e
                           WHERE e.user_id = u.id AND e.created_at >= u.quota_period_start
                       ), 0) AS used_tokens,
                       (SELECT COUNT(*) FROM api_keys k
                        WHERE k.user_id = u.id AND k.revoked_at IS NULL) AS active_keys
                FROM users u WHERE u.id = ?
                """,
                (user_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_users(
        self, *, limit: int = 50, offset: int = 0, query: str = ""
    ) -> tuple[list[dict[str, Any]], int]:
        like = f"%{query.strip()}%"
        where = "WHERE (? = '' OR u.email LIKE ? OR u.name LIKE ?)"
        params = (query.strip(), like, like)
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT u.*,
                       COALESCE((
                           SELECT SUM(e.total_tokens) FROM usage_events e
                           WHERE e.user_id = u.id AND e.created_at >= u.quota_period_start
                       ), 0) AS used_tokens,
                       (SELECT COUNT(*) FROM api_keys k
                        WHERE k.user_id = u.id AND k.revoked_at IS NULL) AS active_keys
                FROM users u {where}
                ORDER BY u.created_at DESC
                LIMIT ? OFFSET ?
                """,
                (*params, limit, offset),
            ).fetchall()
            total = self._conn.execute(
                f"SELECT COUNT(*) AS n FROM users u {where}", params
            ).fetchone()["n"]
        return [dict(r) for r in rows], int(total)

    # -- quota -------------------------------------------------------------
    def quota_summary(self, user_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT u.id AS user_id, u.quota_tokens, u.quota_period_start,
                       COALESCE((
                           SELECT SUM(e.total_tokens) FROM usage_events e
                           WHERE e.user_id = u.id AND e.created_at >= u.quota_period_start
                       ), 0) AS used_tokens
                FROM users u WHERE u.id = ?
                """,
                (user_id,),
            ).fetchone()
        return self._summary(row) if row else None

    @staticmethod
    def _summary(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
        quota = int(row["quota_tokens"])
        used = int(row["used_tokens"])
        unlimited = quota < 0
        return {
            "user_id": row["user_id"],
            "quota_tokens": quota,
            "used_tokens": used,
            "remaining_tokens": None if unlimited else max(0, quota - used),
            "unlimited": unlimited,
            "period_start": row["quota_period_start"],
        }

    def record_usage(
        self,
        *,
        user_id: str,
        api_key_id: str | None,
        run_id: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        request_json: str | None = None,
        response_text: str = "",
        tool_calls_json: str = "[]",
        reasoning_text: str = "",
        log_truncated: bool = False,
    ) -> dict[str, Any]:
        now = utcnow()
        with self._lock, self._conn:
            cursor = self._conn.execute(
                """INSERT INTO usage_events
                   (user_id, api_key_id, run_id, model, prompt_tokens,
                    completion_tokens, total_tokens, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    user_id,
                    api_key_id,
                    run_id,
                    model,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    now,
                ),
            )
            if request_json is not None:
                self._conn.execute(
                    """INSERT INTO call_logs
                       (usage_event_id, user_id, api_key_id, run_id, request_json,
                        response_text, tool_calls_json, reasoning_text, truncated, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        cursor.lastrowid,
                        user_id,
                        api_key_id,
                        run_id,
                        request_json,
                        response_text,
                        tool_calls_json,
                        reasoning_text,
                        1 if log_truncated else 0,
                        now,
                    ),
                )
        summary = self.quota_summary(user_id)
        if summary is None:
            raise NotFoundError(f"User {user_id!r} not found.")
        return summary

    def get_call_log(self, usage_event_id: int) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT l.usage_event_id, l.request_json, l.response_text,
                       l.tool_calls_json, l.reasoning_text, l.truncated,
                       e.run_id, e.model, e.prompt_tokens, e.completion_tokens,
                       e.total_tokens, e.created_at,
                       u.email, u.name
                FROM call_logs l
                JOIN usage_events e ON e.id = l.usage_event_id
                JOIN users u ON u.id = l.user_id
                WHERE l.usage_event_id = ?
                """,
                (usage_event_id,),
            ).fetchone()
        return dict(row) if row else None

    def purge_call_logs(self, retention_days: int) -> int:
        """Delete call bodies older than ``retention_days``; usage ledger is untouched."""
        if retention_days <= 0:
            return 0
        cutoff = (datetime.now(UTC) - timedelta(days=retention_days)).isoformat(
            timespec="microseconds"
        )
        with self._lock, self._conn:
            cursor = self._conn.execute("DELETE FROM call_logs WHERE created_at < ?", (cutoff,))
        return int(cursor.rowcount)

    def reset_usage(self, user_id: str) -> dict[str, Any]:
        now = utcnow()
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE users SET quota_period_start = ?, updated_at = ? WHERE id = ?",
                (now, now, user_id),
            )
            if cur.rowcount == 0:
                raise NotFoundError(f"User {user_id!r} not found.")
        summary = self.quota_summary(user_id)
        assert summary is not None
        return summary

    def list_usage_events(self, user_id: str, *, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT id, run_id, model, prompt_tokens, completion_tokens,
                          total_tokens, created_at
                   FROM usage_events WHERE user_id = ?
                   ORDER BY id DESC LIMIT ?""",
                (user_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def list_all_usage(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        query: str = "",
        user_id: str | None = None,
        run_id: str | None = None,
        since: str | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """Global call history (metadata only) joined with user and key info."""
        conditions: list[str] = []
        params: list[Any] = []
        if user_id:
            conditions.append("e.user_id = ?")
            params.append(user_id)
        if run_id:
            conditions.append("e.run_id = ?")
            params.append(run_id)
        if query.strip():
            needle = query.strip().lower()
            conditions.append(
                "(instr(lower(u.email), ?) > 0 OR instr(lower(u.name), ?) > 0 "
                "OR instr(lower(e.run_id), ?) > 0)"
            )
            params.extend([needle, needle, needle])
        if since:
            conditions.append("e.created_at >= ?")
            params.append(since)
        where_sql = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT e.id, e.run_id, e.user_id, e.model,
                       e.prompt_tokens, e.completion_tokens, e.total_tokens,
                       e.created_at, u.email, u.name,
                       k.prefix AS key_prefix,
                       EXISTS(
                           SELECT 1 FROM call_logs cl WHERE cl.usage_event_id = e.id
                       ) AS has_body
                FROM usage_events e
                JOIN users u ON u.id = e.user_id
                LEFT JOIN api_keys k ON k.id = e.api_key_id
                {where_sql}
                ORDER BY e.id DESC
                LIMIT ? OFFSET ?
                """,
                (*params, limit, offset),
            ).fetchall()
            total = self._conn.execute(
                f"""
                SELECT COUNT(*) AS n
                FROM usage_events e
                JOIN users u ON u.id = e.user_id
                {where_sql}
                """,
                params,
            ).fetchone()["n"]
        result = [dict(row) for row in rows]
        for row in result:
            row["has_body"] = bool(row["has_body"])
        return result, int(total)

    # -- api keys ----------------------------------------------------------
    def create_api_key(self, user_id: str, *, name: str = "default") -> tuple[str, dict[str, Any]]:
        raw = generate_api_key()
        now = utcnow()
        key_id = _new_id("key")
        with self._lock, self._conn:
            if not self._conn.execute("SELECT 1 FROM users WHERE id = ?", (user_id,)).fetchone():
                raise NotFoundError(f"User {user_id!r} not found.")
            self._conn.execute(
                """INSERT INTO api_keys
                   (id, user_id, key_hash, prefix, name, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    key_id,
                    user_id,
                    hash_api_key(raw),
                    key_prefix(raw),
                    name.strip() or "default",
                    now,
                ),
            )
        record = {
            "id": key_id,
            "user_id": user_id,
            "prefix": key_prefix(raw),
            "name": name.strip() or "default",
            "created_at": now,
            "last_used_at": None,
            "revoked_at": None,
        }
        return raw, record

    def find_key(self, key_hash: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                """SELECT k.id AS key_id, k.key_hash, k.prefix, k.revoked_at,
                          k.created_at AS key_created_at, u.*
                   FROM api_keys k JOIN users u ON u.id = k.user_id
                   WHERE k.key_hash = ?""",
                (key_hash,),
            ).fetchone()
        return dict(row) if row else None

    def touch_key(self, key_id: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE api_keys SET last_used_at = ? WHERE id = ?", (utcnow(), key_id)
            )

    def list_keys(self, user_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT id, prefix, name, created_at, last_used_at, revoked_at
                   FROM api_keys WHERE user_id = ? ORDER BY created_at DESC""",
                (user_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def revoke_key(self, user_id: str, key_id: str) -> bool:
        with self._lock, self._conn:
            cur = self._conn.execute(
                """UPDATE api_keys SET revoked_at = ?
                   WHERE id = ? AND user_id = ? AND revoked_at IS NULL""",
                (utcnow(), key_id, user_id),
            )
            return cur.rowcount > 0

    # -- aggregate stats ---------------------------------------------------
    def stats(self) -> dict[str, Any]:
        today = datetime.now(UTC).date().isoformat() + "T00:00:00"
        with self._lock:
            users = self._conn.execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"]
            active = self._conn.execute(
                "SELECT COUNT(*) AS n FROM users WHERE status = 'active'"
            ).fetchone()["n"]
            keys = self._conn.execute(
                "SELECT COUNT(*) AS n FROM api_keys WHERE revoked_at IS NULL"
            ).fetchone()["n"]
            total_tokens = self._conn.execute(
                "SELECT COALESCE(SUM(total_tokens), 0) AS n FROM usage_events"
            ).fetchone()["n"]
            tokens_today = self._conn.execute(
                "SELECT COALESCE(SUM(total_tokens), 0) AS n FROM usage_events WHERE created_at >= ?",
                (today,),
            ).fetchone()["n"]
            runs = self._conn.execute(
                "SELECT COUNT(DISTINCT run_id) AS n FROM usage_events"
            ).fetchone()["n"]
        return {
            "users": int(users),
            "active_users": int(active),
            "active_keys": int(keys),
            "total_tokens": int(total_tokens),
            "tokens_today": int(tokens_today),
            "runs": int(runs),
        }
