"""Call-log retention tests: bodies expire, usage ledger does not."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from harness.db import Database


def test_purge_call_logs_keeps_usage_ledger(tmp_path):
    db = Database(tmp_path / "db.sqlite")
    db.init()
    try:
        user = db.create_user(email="logs@example.com", quota_tokens=1000)
        summary = db.record_usage(
            user_id=user["id"],
            api_key_id=None,
            run_id="run_logs",
            model="fake-model",
            prompt_tokens=1,
            completion_tokens=2,
            total_tokens=3,
            request_json='[{"role": "user", "content": "hi"}]',
            response_text="hello",
            tool_calls_json="[]",
            reasoning_text="thinking",
        )
        assert summary["used_tokens"] == 3

        with db._lock, db._conn:
            event_id = db._conn.execute("SELECT id FROM usage_events").fetchone()["id"]
            old = (datetime.now(UTC) - timedelta(days=8)).isoformat(timespec="microseconds")
            db._conn.execute(
                "UPDATE call_logs SET created_at = ? WHERE usage_event_id = ?",
                (old, event_id),
            )

        log = db.get_call_log(event_id)
        assert log is not None
        assert log["response_text"] == "hello"
        assert log["reasoning_text"] == "thinking"

        rows, total = db.list_all_usage(limit=10)
        assert total == 1
        assert rows[0]["has_body"] is True

        removed = db.purge_call_logs(7)
        assert removed == 1
        assert db.get_call_log(event_id) is None

        rows, total = db.list_all_usage(limit=10)
        assert total == 1
        assert rows[0]["has_body"] is False
        # The usage ledger (and therefore quota accounting) is untouched.
        assert db.quota_summary(user["id"])["used_tokens"] == 3
    finally:
        db.close()
