#!/usr/bin/env python3
"""W6 固定噪声三方对比：CPU-torch / NPU-torch（同事转换权重）/ JAX（原 ckpt）。

同输入 + 同一 noise 数组（policy.infer 的 noise 参数），隔离设备与转换差异。
产出 docs/acceptance/w6-conversion-verification.json（重写结论）。
在 gap-repro 容器内运行（需 NPU 设备与共享盘只读挂载）。
"""

import json
import sys

import numpy as np

sys.path.insert(0, "/data/pi05_hybrid/openpi/src")
sys.path.insert(0, "/data/pi05_hybrid/scripts")

import pi05_service  # noqa: E402  提供 CAM_KEYS 与同构 build_policy 组件

NOISE_SEED = 20260918
# 原 S26 警告：noise 用模型内部动作维度（pi05=32），非解码后的 14 维。
# action_dim 在 main() 里从 torch model config 动态读取。


def build_torch_policy(device: str):
    """与 pi05_service.build_policy 同构，但设备/权重路径参数化。"""
    from openpi.training import config as train_config_mod
    from openpi.training import checkpoints as ocp_checkpoints
    from openpi.policies import policy_config
    from openpi.models import pi0_config
    from openpi import transforms
    import pathlib

    ckpt = pi05_service.CKPT_DIR
    assets = pi05_service.ASSETS_DIR
    data_cfg = train_config_mod.LeRobotAlohaDataConfig(
        repo_id="arx_x5_sim",
        assets=train_config_mod.AssetsConfig(assets_dir=assets, asset_id="arx_x5_sim"),
        base_config=train_config_mod.DataConfig(prompt_from_task=True))
    train_cfg = train_config_mod.TrainConfig(
        name=f"gap_repro_{device}",
        model=pi0_config.Pi0Config(pi05=True, pytorch_compile_mode=None),
        data=data_cfg)
    norm = ocp_checkpoints.load_norm_stats(pathlib.Path(assets), "arx_x5_sim")
    repack = transforms.Group(inputs=[transforms.RepackTransform({
        "images": {k: f"images/{k}" for k in pi05_service.CAM_KEYS},
        "state": "state", "prompt": "prompt"})])
    return policy_config.create_trained_policy(
        train_cfg, ckpt, repack_transforms=repack, norm_stats=norm,
        pytorch_device=device)


def build_jax_policy():
    """原 JAX checkpoint（只读共享盘）→ JAX Policy（与 w6_jax_ref.py 一致）。"""
    from openpi.training import config as train_config_mod
    from openpi.training import checkpoints as ocp_checkpoints
    from openpi.policies import policy_config
    from openpi.models import pi0_config
    from openpi import transforms
    import pathlib

    ckpt = ("/data_shared/pi05_hybrid/robodojo_ckpt/ckpt/RoboDojo/"
            "Pi_05/RoboDojo-sim-arx_x5-joint-0/59999")
    assets = ckpt + "/assets"
    data_cfg = train_config_mod.LeRobotAlohaDataConfig(
        repo_id="arx_x5_sim",
        assets=train_config_mod.AssetsConfig(assets_dir=assets, asset_id="arx_x5_sim"),
        base_config=train_config_mod.DataConfig(prompt_from_task=True))
    train_cfg = train_config_mod.TrainConfig(
        name="w6_jax_fixed", model=pi0_config.Pi0Config(pi05=True), data=data_cfg)
    norm = ocp_checkpoints.load_norm_stats(pathlib.Path(assets), "arx_x5_sim")
    repack = transforms.Group(inputs=[transforms.RepackTransform({
        "images": {k: f"images/{k}" for k in pi05_service.CAM_KEYS},
        "state": "state", "prompt": "prompt"})])
    return policy_config.create_trained_policy(
        train_cfg, ckpt, repack_transforms=repack, norm_stats=norm)


def fixed_obs():
    rng = np.random.default_rng(20260917)
    images = {k: rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
              for k in pi05_service.CAM_KEYS}
    return {"images": images,
            "state": [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7,
                      -0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7],
            "prompt": "organize the table"}


def main():
    probe = build_torch_policy("cpu")
    action_dim = int(probe._model.config.action_dim)  # pi05 内部维度（=32）
    horizon = int(probe._model.config.action_horizon)  # =50
    del probe
    print(f"internal action_dim={action_dim} horizon={horizon}", flush=True)
    noise = np.random.default_rng(NOISE_SEED).normal(
        0.0, 1.0, size=(horizon, action_dim)).astype(np.float32)
    obs = fixed_obs()
    out = {}

    import hashlib

    def sha(a):
        return hashlib.sha256(a.tobytes()).hexdigest()[:16]

    def run(tag, policy):
        r = policy.infer(dict(obs), noise=noise.copy())
        assert np.asarray(r["actions"]).shape[-1] == 14, "输出应为解码后的 14 维"
        a = np.asarray(r["actions"], dtype=np.float64)
        assert a.shape[-1] == 14, a.shape
        out[tag] = a
        print(f"{tag}: absmax={np.abs(a).max():.4f} sha={sha(a)}", flush=True)

    print("building torch-cpu policy...", flush=True)
    run("torch_cpu", build_torch_policy("cpu"))
    print("building torch-npu policy...", flush=True)
    run("torch_npu", build_torch_policy("npu"))
    print("building jax policy (original ckpt)...", flush=True)
    run("jax_original", build_jax_policy())

    def d(x, y):
        diff = np.abs(out[x] - out[y])
        return {"mean_absdiff": float(diff.mean()), "max_absdiff": float(diff.max()),
                "per_dim_mean_absdiff": diff.mean(axis=0).tolist()}

    res = {
        "noise_seed": NOISE_SEED, "noise_shape": list(noise.shape),
        "input": "fixed_obs()（与 w7 基线同源）",
        "pairwise": {"torch_cpu_vs_npu": d("torch_cpu", "torch_npu"),
                     "jax_vs_torch_cpu": d("jax_original", "torch_cpu"),
                     "jax_vs_torch_npu": d("jax_original", "torch_npu")},
        "sha": {k: sha(v) for k, v in out.items()},
        "note": "固定 noise 后的差值 = 设备/转换效应，不含采样噪声（红1 修复）",
    }
    print(json.dumps(res["pairwise"], indent=1))
    open("/workspace/data/w6-conversion-verification.json", "w").write(
        json.dumps(res, indent=1))
    print("W6_VERIFY_SAVED")
    # 判定门槛（固定噪声下）：设备对 |Δ|mean ≤ 0.01 且 转换对 |Δ|mean ≤ 0.05 → 带内
    dp = res["pairwise"]
    verdict = {
        "device_effect_within": dp["torch_cpu_vs_npu"]["mean_absdiff"] <= 0.01,
        "conversion_within": dp["jax_vs_torch_cpu"]["mean_absdiff"] <= 0.05,
    }
    print("VERDICT", json.dumps(verdict))
    open("/workspace/data/w6-verdict.json", "w").write(json.dumps(verdict))
    np.save("/workspace/data/w6_actions.npz",
            {k: v for k, v in out.items()})  # 三方原始动作落盘，避免重算


if __name__ == "__main__":
    main()
