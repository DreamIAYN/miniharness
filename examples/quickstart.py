"""Run one agent task through the harness.

    HARNESS_URL=http://127.0.0.1:8000 \
    HARNESS_API_KEY=hx_... \
    python examples/quickstart.py "用 Python 算一下 1 到 100 的和"

The example intentionally contains no LLM SDK: the calling project only needs
an HTTP client, so it can be written in any language.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sdk"))

from harness_client import HarnessClient, HarnessError


def main() -> int:
    prompt = " ".join(sys.argv[1:]) or "用 Python 计算 1 到 100 的和，并给出结果。"
    base_url = os.environ.get("HARNESS_URL", "http://127.0.0.1:8000")
    api_key = os.environ.get("HARNESS_API_KEY")
    if not api_key:
        print("请先设置 HARNESS_API_KEY（在 /admin/ui 中创建用户后获得）", file=sys.stderr)
        return 2

    try:
        with HarnessClient(base_url, api_key) as client:
            print(f"→ {prompt}\n")
            print(f"user: {client.me()['email']}")
            result = client.run(prompt, max_steps=6)

            print("\n=== final answer ===")
            print(result["output"])
            print("\n=== run info ===")
            print(
                f"run_id={result['run_id']} steps={result['steps']} "
                f"tokens={result['usage']['total_tokens']} "
                f"remaining={result['remaining_tokens']}"
            )
            return 0
    except HarnessError as exc:
        print(f"harness error [{exc.status}/{exc.code}]: {exc.message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
