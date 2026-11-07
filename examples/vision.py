"""Send a local image to the harness.

    HARNESS_URL=http://127.0.0.1:8000 \
    HARNESS_API_KEY=hx_... \
    python examples/vision.py chart.png "分析这张图表的趋势"

Requires a vision-capable model (DeepSeek: deepseek-flash).
"""

from __future__ import annotations

import base64
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sdk"))

from harness_client import HarnessClient, HarnessError

MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python examples/vision.py <image> [prompt]", file=sys.stderr)
        return 2

    image_path = Path(sys.argv[1]).expanduser()
    prompt = " ".join(sys.argv[2:]) or "描述这张图片，并指出图中值得注意的信息。"
    mime = MIME.get(image_path.suffix.lower())
    if mime is None:
        print("supported image formats: jpg, jpeg, png, gif, webp", file=sys.stderr)
        return 2

    api_key = os.environ.get("HARNESS_API_KEY")
    if not api_key:
        print("set HARNESS_API_KEY first", file=sys.stderr)
        return 2

    data_url = f"data:{mime};base64," + base64.b64encode(image_path.read_bytes()).decode()
    try:
        with HarnessClient(
            os.environ.get("HARNESS_URL", "http://127.0.0.1:8000"), api_key, timeout=600
        ) as client:
            result = client.run(
                prompt,
                images=[{"url": data_url, "detail": "high"}],
                max_steps=4,
            )
            print(result["output"])
            print(f"\n[tokens: {result['usage']['total_tokens']}]")
            return 0
    except HarnessError as exc:
        print(f"harness error [{exc.status}/{exc.code}]: {exc.message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
