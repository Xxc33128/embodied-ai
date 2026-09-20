#!/usr/bin/env python3
"""LP-A0 正式运行（validation.json）：8 draws × 三腿（jax_cpu/torch_cpu/torch_npu）。

- 噪声按 W7 规则由 JAX 生成（PRNGKey(1000+draw)），同数组注入三腿。
- draws 0-1 = 合成观测（seed0/1）；draws 2-7 = LIBERO Goal 真实帧（lp_a0_frames.npz）。
- 逐 draw 记录三腿动作 sha256、成对差值与阈值判定（validation.json）。
在 gap-repro 容器执行（PYTHONPATH 含 /data/pi05_hybrid/openpi/src；NPU 卡 1）。
"""
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, "/data/pi05_hybrid/openpi/src")

import numpy as np  # noqa: E402
import jax  # noqa: E402
from jax import random  # noqa: E402

from openpi.policies import policy_config  # noqa: E402
from openpi.training import config as _config  # noqa: E402
from openpi.training import checkpoints as ocp_checkpoints  # noqa: E402
import dataclasses  # noqa: E402
import pathlib  # noqa: E402

CFG_NAME = "pi05_libero"
JAX_CKPT = "/workspace/data/checkpoints/pi05_libero"
PT_CKPT = "/workspace/data/checkpoints/pi05_libero_pt"
NORM_BASE = PT_CKPT + "/assets/physical-intelligence"
ASSET_ID = "libero"
TH = {"per_draw_max_abs": 0.03, "per_draw_mean_abs": 0.005}

cfg = _config.get_config(CFG_NAME)
cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, pytorch_compile_mode=None))
norm_stats = ocp_checkpoints.load_norm_stats(pathlib.Path(NORM_BASE), ASSET_ID)
print("norm_stats loaded", flush=True)


def build(device, ckpt):
    return policy_config.create_trained_policy(
        cfg, ckpt, norm_stats=norm_stats, pytorch_device=device)


def sha(a):
    return hashlib.sha256(np.asarray(a, dtype=np.float32).tobytes()).hexdigest()[:16]


def cmp(a, b):
    d = np.abs(a - b)
    return {"max_abs": float(d.max()), "mean_abs": float(d.mean())}


results = []
p_jax = p_cpu = p_npu = None

t0 = time.time()
for draw in range(8):
    rng = np.random.default_rng(draw) if draw < 2 else None
    if draw < 2:
        inputs = {
            "observation/state": rng.uniform(-1, 1, 8).astype(np.float32),
            "observation/image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
            "observation/wrist_image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
            "prompt": "open the middle drawer of the cabinet",
        }
    else:
        frames = np.load("/workspace/data/lp_a0_frames.npz")
        i = draw - 2
        inputs = {
            "observation/state": frames["state"][i],
            "observation/image": frames["agentview"][i],
            "observation/wrist_image": frames["wrist"][i],
            "prompt": "open the middle drawer of the cabinet",
        }
    key = random.PRNGKey(1000 + draw)
    noise = np.asarray(random.normal(key, (10, 32), dtype=np.float32))

    rec = {"draw": draw, "noise_sha": sha(noise), "input_sha": sha(inputs["observation/image"])}
    t = time.time()
    if p_jax is None:
        p_jax = build(None, JAX_CKPT); print("jax built", time.time() - t0, flush=True)
    a_jax = np.asarray(p_jax.infer(dict(inputs), noise=noise)["actions"], dtype=np.float32)
    rec["jax_sha"], rec["jax_s"] = sha(a_jax), round(time.time() - t, 1)

    t = time.time()
    if p_cpu is None:
        p_cpu = build("cpu", PT_CKPT); print("cpu built", time.time() - t0, flush=True)
    a_cpu = np.asarray(p_cpu.infer(dict(inputs), noise=noise)["actions"], dtype=np.float32)
    rec["cpu_sha"], rec["cpu_s"] = sha(a_cpu), round(time.time() - t, 1)

    t = time.time()
    if p_npu is None:
        p_npu = build("npu", PT_CKPT); print("npu built", time.time() - t0, flush=True)
    a_npu = np.asarray(p_npu.infer(dict(inputs), noise=noise)["actions"], dtype=np.float32)
    rec["npu_sha"], rec["npu_s"] = sha(a_npu), round(time.time() - t, 1)

    assert a_jax.shape == a_cpu.shape == a_npu.shape, (a_jax.shape, a_cpu.shape, a_npu.shape)
    rec["npu_vs_cpu"] = cmp(a_npu, a_cpu)
    rec["jax_vs_cpu"] = cmp(a_jax, a_cpu)
    rec["gate"] = (rec["npu_vs_cpu"]["max_abs"] <= TH["per_draw_max_abs"]
                   and rec["npu_vs_cpu"]["mean_abs"] <= TH["per_draw_mean_abs"])
    results.append(rec)
    print(f"DRAW {draw} gate={rec['gate']} npu_vs_cpu={rec['npu_vs_cpu']} "
          f"jax_vs_cpu={rec['jax_vs_cpu']}", flush=True)

passes = sum(r["gate"] for r in results)
out = {"schema": "gap_repro.lp_a0_formal.v1", "date": "2026-09-18",
       "thresholds": TH, "draws": results, "passes": passes,
       "note": "jax leg = 官方 JAX ckpt 原始前向；torch legs = 转换权重（LP2 逐位保真）"}
with open("/workspace/data/lp_a0_formal.json", "w") as f:
    json.dump(out, f, indent=1)
print(f"FORMAL_DONE passes={passes}/8", flush=True)
