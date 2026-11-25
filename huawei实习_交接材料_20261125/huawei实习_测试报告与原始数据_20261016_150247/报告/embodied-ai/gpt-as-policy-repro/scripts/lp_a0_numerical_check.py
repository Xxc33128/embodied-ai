#!/usr/bin/env python3
"""LP-A0 数值验收（第一步）：转换后 pi05_libero 在 Torch CPU 与 Torch NPU 上，
固定观测 + 固定种子的动作对比。

- 权重级等价已由 LP2 审查证明（811 张量逐位）；本脚本验证运行时等价。
- 阈值说明：本脚本产出实测差值作为 LP-A0 证据；正式 W6/W7 阈值仍按主计划
  预注册程序另行确定，不以本脚本输出反推。
在 gap-repro 容器执行：
  PYTHONPATH=/data/pi05_hybrid/openpi/src python3 scripts/lp_a0_numerical_check.py
"""
import sys
import time

sys.path.insert(0, "/data/pi05_hybrid/openpi/src")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from openpi.policies import policy_config  # noqa: E402
from openpi.training import config as _config  # noqa: E402

CFG_NAME = "pi05_libero"
CKPT = "/workspace/data/checkpoints/pi05_libero_pt"
SEED = 1234

cfg = _config.get_config(CFG_NAME)
# 平台适配：NPU 上 torch.compile(inductor) 需要 torch_mlir（未装）；同事服务同款
# 处理是显式关闭编译。eager 对数值对比也更干净，两设备统一关闭。
import dataclasses
cfg = dataclasses.replace(
    cfg, model=dataclasses.replace(cfg.model, pytorch_compile_mode=None))


def build(device):
    return policy_config.create_trained_policy(cfg, CKPT, pytorch_device=device)


rng = np.random.default_rng(0)
inputs = {
    "observation/state": rng.uniform(-1, 1, 8).astype(np.float32),
    "observation/image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
    "observation/wrist_image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
    "prompt": "open the middle drawer of the cabinet",
}

t0 = time.time()
# W7 规则：噪声生成一次、显式注入两栈（生成式策略对噪声敏感，分开播种必然分歧）
noise = np.random.default_rng(42).standard_normal((10, 32)).astype(np.float32)

pc = build("cpu")
with torch.inference_mode():
    a_cpu1 = np.asarray(pc.infer(dict(inputs), noise=noise)["actions"])
    a_cpu2 = np.asarray(pc.infer(dict(inputs), noise=noise)["actions"])
t_cpu = time.time() - t0

t1 = time.time()
pn = build("npu")
with torch.inference_mode():
    a_npu = np.asarray(pn.infer(dict(inputs), noise=noise)["actions"])
t_npu = time.time() - t1

determin = np.abs(a_cpu1 - a_cpu2).max()
diff = np.abs(a_cpu1 - a_npu)
denom = np.maximum(np.abs(a_cpu1), 1e-6)
rel = diff / denom
print(f"A0 shape={a_cpu1.shape} cpu_time={t_cpu:.1f}s npu_time={t_npu:.1f}s")
print(f"A0 cpu_determinism_max_abs={determin:.3e}")
print(f"A0 npu_vs_cpu max_abs={diff.max():.3e} mean_abs={diff.mean():.3e} "
      f"mean_rel={rel.mean():.3e} p99_abs={np.quantile(diff, 0.99):.3e}")
print(f"A0 cpu_sample[0,:5]={a_cpu1[0, :5].round(6).tolist()}")
print(f"A0 npu_sample[0,:5]={a_npu[0, :5].round(6).tolist()}")
print("A0_DONE")
