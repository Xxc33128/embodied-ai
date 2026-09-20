#!/usr/bin/env python3
"""W6/W7 参考推理基线：同事转换权重的确定性前向 → 固定输出指纹。

在 gap-repro 容器内运行：
  python3 scripts/w6_reference_inference.py [host port]
通过 HTTP 调用 pi05_service（与正式推理同链路），固定种子合成观测，
记录 (50,14) 输出的 sha256 与逐维统计，作为 W7 NPU 数值验收的基准。
"""

import base64
import hashlib
import json
import sys
import urllib.request

import numpy as np

HOST = sys.argv[1] if len(sys.argv) > 1 else "localhost"
PORT = sys.argv[2] if len(sys.argv) > 2 else "8642"

rng = np.random.default_rng(20260917)
images = {k: rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
          for k in ("cam_high", "cam_left_wrist", "cam_right_wrist")}
state = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7,
         -0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7]
req = {"images": {k: base64.b64encode(v.tobytes()).decode() for k, v in images.items()},
       "shapes": {k: list(v.shape) for k, v in images.items()},
       "state": state, "prompt": "organize the table"}
r = urllib.request.urlopen(urllib.request.Request(
    f"http://{HOST}:{PORT}/infer", data=json.dumps(req).encode(),
    headers={"Content-Type": "application/json"}), timeout=300)
out = json.load(r)
a = np.array(out["actions"], dtype=np.float64)
assert a.shape == (50, 14) and np.isfinite(a).all()
digest = hashlib.sha256(a.tobytes()).hexdigest()
record = {
    "date": "2026-09-18",
    "weights": "robodojo_pi05_pt（同事适配 v1，lock.checkpoint.colleague_adapted_version）",
    "service": f"{HOST}:{PORT} (pi05_service.py npu)",
    "input": {"state": state, "prompt": "organize the table",
              "images": "fixed-seed rng(20260917) 480x640x3 uint8 x3"},
    "actions_sha256": digest,
    "absmax": float(np.abs(a).max()), "mean": float(a.mean()),
    "per_dim_mean": a.mean(axis=0).tolist(),
    "per_dim_std": a.std(axis=0).tolist(),
    "row0": a[0].tolist(), "service_ms": out.get("ms"),
}
print(json.dumps(record, indent=1))
with open("/workspace/data/w7_npu_baseline.json", "w") as f:
    json.dump(record, f, indent=1)
print("BASELINE_SAVED /workspace/data/w7_npu_baseline.json")
