"""Example: use the harness as the agent backend of another service.

This simulates a code-review bot.  The outer service owns its own business
logic, prompts and permissions; the harness owns the agent loop, tools,
sandbox and token quota.  Swap the dummy diff for your real input.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sdk"))

from harness_client import HarnessClient

SYSTEM = """You are a careful code reviewer. Use the tools only when you need to
inspect the workspace. Report concrete problems with file names and line
references. Keep the final review under 200 words."""

DUMMY_DIFF = """
diff --git a/app.py b/app.py
@@
-def divide(a, b):
-    return a / b
+def divide(a, b):
+    return a / b if b else 0
"""


def review_diff(diff: str) -> dict:
    with HarnessClient(
        os.environ.get("HARNESS_URL", "http://127.0.0.1:8000"),
        os.environ["HARNESS_API_KEY"],
    ) as client:
        return client.run(
            prompt=(
                "Review this diff. Consider correctness, edge cases and style.\n\n"
                f"```diff\n{diff}\n```"
            ),
            system=SYSTEM,
            max_steps=5,
            max_tokens=800,
        )


if __name__ == "__main__":
    result = review_diff(DUMMY_DIFF)
    print(result["output"])
    print(f"\n[tokens used: {result['usage']['total_tokens']}]")
