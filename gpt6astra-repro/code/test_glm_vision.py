"""测试智谱端点：glm-4.6 / glm-5.3 是否接受 image_url 输入（视觉能力探测）。"""
import base64
import io
import json
import os
import sys

import httpx
from PIL import Image

KEY = os.environ["ZHIPUAI_API_KEY"]
BASE = "https://open.bigmodel.cn/api/paas/v4"

img = Image.new("RGB", (64, 64), (200, 30, 30))
buf = io.BytesIO()
img.save(buf, format="PNG")
data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def probe(model: str) -> None:
    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What color is this image? Answer in one word."},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "max_tokens": 1024,
    }
    r = httpx.post(f"{BASE}/chat/completions", headers={"Authorization": f"Bearer {KEY}"},
                   json=body, timeout=60)
    if r.status_code != 200:
        print(f"{model}: HTTP {r.status_code} — {r.text[:200]}")
        return
    d = r.json()
    msg = d["choices"][0]["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    usage = d.get("usage", {})
    print(f"{model}: OK content={content.strip()[:60]!r} "
          f"reasoning_len={len(reasoning)} usage={usage.get('prompt_tokens')}+{usage.get('completion_tokens')}")


for m in sys.argv[1:] or ["glm-4.6", "glm-5.3"]:
    try:
        probe(m)
    except Exception as e:
        print(f"{m}: EXC {e}")
