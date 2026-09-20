#!/usr/bin/env python3
"""W6:JAX 侧参考前向(原 checkpoint, 只读共享盘)。"""
import sys, json, hashlib
import numpy as np
sys.path.insert(0, "/data/pi05_hybrid/openpi/src")
sys.path.insert(0, "/data/pi05_hybrid/scripts")
import pi05_service

# 与 pi05_service 同构, 但: (1) checkpoint 指向只读共享盘原 ckpt, (2) JAX 设备
pi05_service.CKPT_DIR = "/data_shared/pi05_hybrid/robodojo_ckpt/ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/59999"
policy = pi05_service.build_policy.__wrapped__() if hasattr(pi05_service.build_policy, "__wrapped__") else None
import inspect
src = inspect.getsource(pi05_service.build_policy)
# build_policy 内部 create_trained_policy(..., pytorch_device=DEVICE) → DEVICE="npu";
# JAX 侧需 pytorch_device=None。直接内联等价构建:
from openpi.training import config as train_config_mod
from openpi.training import checkpoints as ocp_checkpoints
from openpi.policies import policy_config
from openpi.models import pi0_config
from openpi import transforms
CKPT = "/data_shared/pi05_hybrid/robodojo_ckpt/ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/59999"
ASSETS = CKPT + "/assets"
data_cfg = train_config_mod.LeRobotAlohaDataConfig(
    repo_id="arx_x5_sim",
    assets=train_config_mod.AssetsConfig(assets_dir=ASSETS, asset_id="arx_x5_sim"),
    base_config=train_config_mod.DataConfig(prompt_from_task=True))
train_cfg = train_config_mod.TrainConfig(
    name="w6_jax_ref", model=pi0_config.Pi0Config(pi05=True), data=data_cfg)
norm = ocp_checkpoints.load_norm_stats(ASSETS + "/arx_x5_sim", "arx_x5_sim") if False else None
import pathlib
norm = ocp_checkpoints.load_norm_stats(pathlib.Path(ASSETS), "arx_x5_sim")
repack = transforms.Group(inputs=[transforms.RepackTransform({
    "images": {k: f"images/{k}" for k in pi05_service.CAM_KEYS},
    "state": "state", "prompt": "prompt"})])
policy = policy_config.create_trained_policy(
    train_cfg, "/data_shared/pi05_hybrid/robodojo_ckpt/ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/59999",
    repack_transforms=repack, norm_stats=norm)

rng = np.random.default_rng(20260917)
images = {k: rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
          for k in ("cam_high", "cam_left_wrist", "cam_right_wrist")}
state = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7,
         -0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7]
result = policy.infer({"images": images, "state": state, "prompt": "organize the table"})
a = np.asarray(result["actions"], dtype=np.float64)
digest = hashlib.sha256(a.tobytes()).hexdigest()
rec = {"actions_sha256": digest, "shape": list(a.shape),
       "absmax": float(np.abs(a).max()), "mean": float(a.mean()),
       "per_dim_mean": a.mean(axis=0).tolist(), "per_dim_std": a.std(axis=0).tolist(),
       "row0": a[0].tolist()}
print(json.dumps(rec, indent=1))
open("/workspace/data/w6_jax_ref.json", "w").write(json.dumps(rec, indent=1))
print("JAX_REF_SAVED")
