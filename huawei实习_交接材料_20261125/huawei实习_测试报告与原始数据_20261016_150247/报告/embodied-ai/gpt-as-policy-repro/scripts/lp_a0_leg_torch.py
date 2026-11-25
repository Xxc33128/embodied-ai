#!/usr/bin/env python3
"""LP-A0 三腿分进程：leg 2/3 = Torch CPU / Torch NPU（转换权重，无 JAX 导入）。

用法: lp_a0_leg_torch.py <cpu|npu>
输入与噪声来自 JAX leg 产物（/workspace/data/lp_a0_formal/{noise,frames}.npz）。
输出 torch_<device>.npz。
"""
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, "/data/pi05_hybrid/openpi/src")

import numpy as np
import torch

from openpi.policies import policy_config
from openpi.training import config as _config
from openpi.training import checkpoints as ocp_checkpoints
import dataclasses
import pathlib

CFG_NAME = "pi05_libero"
PT_CKPT = "/workspace/data/checkpoints/pi05_libero_pt"
NORM_BASE = PT_CKPT + "/assets/physical-intelligence"
ASSET_ID = "libero"
OUT = "/workspace/data/lp_a0_formal"
DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cpu"


def make_inputs(draw, frames):
    rng = np.random.default_rng(draw) if draw < 2 else None
    if draw < 2:
        return {
            "observation/state": rng.uniform(-1, 1, 8).astype(np.float32),
            "observation/image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
            "observation/wrist_image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
            "prompt": "open the middle drawer of the cabinet",
        }
    i = draw - 2
    return {
        "observation/state": frames["state"][i],
        "observation/image": frames["agentview"][i],
        "observation/wrist_image": frames["wrist"][i],
        "prompt": "open the middle drawer of the cabinet",
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    frames = np.load("/workspace/data/lp_a0_frames.npz")
    noises = np.load(OUT + "/noise.npz")["noise"]
    cfg = _config.get_config(CFG_NAME)
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(
        cfg.model, pytorch_compile_mode=None))
    norm_stats = ocp_checkpoints.load_norm_stats(pathlib.Path(NORM_BASE), ASSET_ID)
    policy = policy_config.create_trained_policy(
        cfg, PT_CKPT, norm_stats=norm_stats, pytorch_device=DEVICE)
    print(f"torch {DEVICE} policy built", flush=True)

    outs, metas = [], []
    for draw in range(8):
        inputs = make_inputs(draw, frames)
        noise = noises[draw]
        t = time.time()
        with torch.inference_mode():
            a = np.asarray(policy.infer(dict(inputs), noise=noise)["actions"],
                           dtype=np.float32)
        metas.append({"draw": draw, "s": round(time.time() - t, 1),
                      "sha": hashlib.sha256(a.tobytes()).hexdigest()[:16]})
        outs.append(a)
        print(f"{DEVICE} draw {draw} {metas[-1]['s']}s sha={metas[-1]['sha']}", flush=True)
    np.savez(OUT + f"/torch_{DEVICE}.npz", actions=np.stack(outs))
    json.dump(metas, open(OUT + f"/torch_{DEVICE}_meta.json", "w"), indent=1)
    print(f"TORCH_{DEVICE}_LEG_DONE", flush=True)


if __name__ == "__main__":
    main()
