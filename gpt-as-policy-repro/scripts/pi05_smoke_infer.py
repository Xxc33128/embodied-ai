#!/usr/bin/env python3
"""π0.5 推理服务冒烟测试：合成观测 → 50×14 动作。

仅验证链路（图像编码/预处理/NPU 前向/后处理/形状与数值健康），
数值验收（W7 对齐原 JAX 流）另行执行。
"""
import base64
import json
import sys
import time
import urllib.request

import numpy as np

HOST = sys.argv[1] if len(sys.argv) > 1 else "localhost"
PORT = sys.argv[2] if len(sys.argv) > 2 else "8642"

rng = np.random.default_rng(0)
images = {k: rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
          for k in ("cam_high", "cam_left_wrist", "cam_right_wrist")}
state = [0.0] * 14
req = {
    "images": {k: base64.b64encode(v.tobytes()).decode() for k, v in images.items()},
    "shapes": {k: list(v.shape) for k, v in images.items()},
    "state": state,
    "prompt": "organize the table",
}
t0 = time.time()
r = urllib.request.urlopen(urllib.request.Request(
    f"http://{HOST}:{PORT}/infer",
    data=json.dumps(req).encode(),
    headers={"Content-Type": "application/json"}), timeout=300)
out = json.load(r)
dt = (time.time() - t0) * 1000
a = np.array(out["actions"])
assert a.shape == (50, 14), f"bad shape {a.shape}"
assert np.isfinite(a).all(), "non-finite actions"
print(f"SMOKE_OK shape={a.shape} service_ms={out.get('ms')} wall_ms={dt:.0f} "
      f"absmax={np.abs(a).max():.4f} mean={a.mean():.6f}")
