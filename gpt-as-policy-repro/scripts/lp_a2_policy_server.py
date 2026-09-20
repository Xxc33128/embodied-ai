#!/usr/bin/env python3
"""LP-A2 策略服务（gap-repro 容器）：pi05_libero_pt 在 torch cpu/npu 上推理。

协议（LP-A0 分容器架构沿用）：轮询 /workspace/data/lp_a2/req/*.npz，
infer(obs, noise=...) 后写 /workspace/data/lp_a2/resp/<同名>.npz 并删除请求。
请求键：agentview/wrist（已按官方客户端旋转 180°、uint8 224）、state(8,)、
prompt(str)、noise(10,32)（pi05 内部动作维 32）、meta(标签字符串)。响应：actions(10,7) float32（输出解码到 7）+ meta。
用法：lp_a2_policy_server.py <cpu|npu>；进程隔离由外层 docker exec 保证。
"""
import dataclasses
import hashlib
import json
import os
import pathlib
import sys
import time
import traceback

sys.path.insert(0, "/data/pi05_hybrid/openpi/src")

import numpy as np
import torch

from openpi.policies import policy_config
from openpi.training import config as _config
from openpi.training import checkpoints as ocp_checkpoints

CFG_NAME = "pi05_libero"
PT_CKPT = "/workspace/data/checkpoints/pi05_libero_pt"
NORM_BASE = PT_CKPT + "/assets/physical-intelligence"
ASSET_ID = "libero"
# 交换目录（L3 集成修正）：默认沿用 LP-A2 目录（向后兼容）；正式 campaign
# 经环境变量指向独立目录（如 /workspace/data/lp5）
BASE = pathlib.Path(os.environ.get("LP_POLICY_EXCHANGE_DIR",
                                   "/workspace/data/lp_a2"))
DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cpu"
(BASE / "req").mkdir(parents=True, exist_ok=True)
(BASE / "resp").mkdir(parents=True, exist_ok=True)


def serve_one(policy, req_path: pathlib.Path) -> None:
    d = np.load(req_path, allow_pickle=False)
    inputs = {
        "observation/image": d["agentview"],
        "observation/wrist_image": d["wrist"],
        "observation/state": d["state"],
        "prompt": str(d["prompt"]),
    }
    # 噪声契约：请求含 noise → 显式传入（LP-A2 设备对照的验证机制）；
    # 不含 → infer(noise=None) → openpi 内部采样（正式协议，官方客户端
    # 行为逐字一致）。meta 记录模式供审计对账。
    noise_mode = "explicit" if "noise" in d.files else "internal"
    t = time.time()
    with torch.inference_mode():
        out = policy.infer(dict(inputs),
                           noise=d["noise"] if noise_mode == "explicit" else None)
    actions = np.asarray(out["actions"], dtype=np.float32)
    meta = {
        "device": DEVICE, "infer_s": round(time.time() - t, 2),
        "actions_sha": hashlib.sha256(actions.tobytes()).hexdigest()[:16],
        "noise_mode": noise_mode,
    }
    tmp = req_path.with_name(req_path.stem + ".tmp.npz")  # 以 .npz 结尾防 savez 追加后缀
    np.savez(tmp, actions=actions, meta=np.array(json.dumps(meta)))
    tmp.rename(BASE / "resp" / req_path.name)  # 原子落位
    req_path.unlink()
    print(f"[{DEVICE}] served {req_path.name} {meta['infer_s']}s "
          f"sha={meta['actions_sha']}", flush=True)


def main():
    cfg = _config.get_config(CFG_NAME)
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(
        cfg.model, pytorch_compile_mode=None))
    norm_stats = ocp_checkpoints.load_norm_stats(
        pathlib.Path(NORM_BASE), ASSET_ID)
    policy = policy_config.create_trained_policy(
        cfg, PT_CKPT, norm_stats=norm_stats, pytorch_device=DEVICE)
    print(f"LP_A2_SERVER_READY device={DEVICE}", flush=True)
    (BASE / f"server_ready_{DEVICE}").touch()
    while True:
        reqs = sorted((BASE / "req").glob("*.npz"),
                      key=lambda p: p.stat().st_mtime)
        for r in reqs:
            try:
                serve_one(policy, r)
            except Exception:
                traceback.print_exc()
                err = BASE / "resp" / (r.stem + ".err")
                err.write_text(traceback.format_exc())
                r.unlink()
        time.sleep(0.05)


if __name__ == "__main__":
    main()
