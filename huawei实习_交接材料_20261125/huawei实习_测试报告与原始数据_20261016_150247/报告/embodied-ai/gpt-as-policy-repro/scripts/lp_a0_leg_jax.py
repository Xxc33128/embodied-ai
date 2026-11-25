#!/usr/bin/env python3
"""LP-A0 三腿分进程架构：leg 1 = JAX CPU（官方 JAX ckpt 原始前向）+ 噪声生成。

输出 /workspace/data/lp_a0_formal/{jax.npz, noise.npz}。
"""
import json
import os
import sys
import time

sys.path.insert(0, "/data/pi05_hybrid/openpi/src")

import numpy as np
import jax
from jax import random

from openpi.policies import policy_config
from openpi.training import config as _config
from openpi.training import checkpoints as ocp_checkpoints
import dataclasses
import pathlib

CFG_NAME = "pi05_libero"
JAX_CKPT = "/workspace/data/checkpoints/pi05_libero"
PT_CKPT = "/workspace/data/checkpoints/pi05_libero_pt"
NORM_BASE = PT_CKPT + "/assets/physical-intelligence"
ASSET_ID = "libero"
OUT = "/workspace/data/lp_a0_formal"


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
    frames = np.load(OUT + "/../lp_a0_frames.npz")
    cfg = _config.get_config(CFG_NAME)
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(
        cfg.model, pytorch_compile_mode=None))
    norm_stats = ocp_checkpoints.load_norm_stats(pathlib.Path(NORM_BASE), ASSET_ID)
    policy = policy_config.create_trained_policy(
        cfg, JAX_CKPT, norm_stats=norm_stats)  # 无 pytorch_device → JAX
    print("jax policy built", flush=True)

    jax_out, noises, metas = [], [], []
    for draw in range(8):
        inputs = make_inputs(draw, frames)
        noise = np.asarray(random.normal(
            random.PRNGKey(1000 + draw), (10, 32), dtype=np.float32))
        t = time.time()
        a = np.asarray(policy.infer(dict(inputs), noise=noise)["actions"], dtype=np.float32)
        jax_out.append(a)
        noises.append(noise)
        metas.append({"draw": draw, "jax_s": round(time.time() - t, 1),
                      "jax_sha": hashlib_sha(a)})
        print(f"jax draw {draw} {metas[-1]['jax_s']}s sha={metas[-1]['jax_sha']}", flush=True)
    np.savez(OUT + "/jax.npz", actions=np.stack(jax_out))
    np.savez(OUT + "/noise.npz", noise=np.stack(noises))
    json.dump(metas, open(OUT + "/jax_meta.json", "w"), indent=1)
    print("JAX_LEG_DONE", flush=True)


def hashlib_sha(a):
    import hashlib
    return hashlib.sha256(np.asarray(a, dtype=np.float32).tobytes()).hexdigest()[:16]


if __name__ == "__main__":
    main()
