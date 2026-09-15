"""后端核验（新口径 §4.1）：robosuite Lift/Panda 是否满足
"已有机器人控制器 + 可用物理抓放示例 + 相机/本体观测"。

独立脚本仅验环境，不参与正式模型决策。运行：
  .venv-robosuite/bin/python verify_robosuite_backend.py
"""
import json
import time

import numpy as np
import robosuite
from robosuite.controllers import load_composite_controller_config


def make_env(seed=0):
    _cc = load_composite_controller_config(controller="BASIC")
    _cc["body_parts"]["right"]["input_ref_frame"] = "world"  # EEF delta 用世界系
    return robosuite.make(
        "Lift",
        robots="Panda",
        controller_configs=_cc,
        seed=seed,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["frontview", "robot0_eye_in_hand"],
        camera_heights=224,
        camera_widths=224,
        control_freq=20,
        reward_shaping=False,
    )


def scripted_lift_episode(env, max_steps=350, verbose=False):
    """独立验证脚本：OSC 末端目标脚本抓方块并抬升（仅验环境）。"""
    obs = env.reset()
    kp = 4.0
    phase, phase_steps = "approach", 0
    success_seen = False
    for t in range(max_steps):
        eef = obs["robot0_eef_pos"]
        cube = obs["cube_pos"]
        gq = obs["robot0_gripper_qpos"]
        if phase == "approach":
            target = cube + np.array([0.0, 0.0, 0.12])
            gripper = -1.0  # robosuite Panda: -1=张开（E0 标定）
            if np.linalg.norm(eef - target) < 0.02:
                phase, phase_steps = "descend", 0
        elif phase == "descend":
            target = cube + np.array([0.0, 0.0, 0.005])
            gripper = -1.0
            if np.linalg.norm(eef - target) < 0.015 or phase_steps > 60:
                phase, phase_steps = "close", 0
        elif phase == "close":
            target = eef.copy()
            gripper = 1.0  # +1=闭合
            if np.all(np.abs(gq) < 0.02) or phase_steps > 50:
                phase, phase_steps = "lift", 0
        else:  # lift
            target = eef + np.array([0.0, 0.0, 0.12])
            gripper = 1.0
        d = np.clip(kp * (target - eef), -0.2, 0.2)
        action = np.concatenate([d, [0.0, 0.0, 0.0], [gripper]])
        obs, reward, done, info = env.step(action)
        phase_steps += 1
        ok = env._check_success()
        success_seen = success_seen or ok
        if verbose and t % 50 == 0:
            print(f"    t={t} phase={phase} eef_z={eef[2]:.3f} cube_z={cube[2]:.3f} ok={ok}")
        if success_seen and phase == "lift" and phase_steps > 30:
            break
        if done:
            break
    return success_seen, t + 1


def main():
    env = make_env()
    print("=== 接口清单 ===")
    lo, hi = env.action_spec[0], env.action_spec[1]
    print("action dim:", lo.shape, "low:", np.round(lo, 3), "high:", np.round(hi, 3))
    obs = env.reset()
    state_keys = {k: (v.shape if hasattr(v, "shape") else type(v).__name__) for k, v in obs.items()}
    for k in sorted(state_keys):
        print(f"  obs[{k}]: {state_keys[k]}")
    print("control_freq:", env.control_freq, " model nq:", env.sim.model.nq)

    print("\n=== 独立脚本抓放验证（10 episodes，仅验环境）===")
    results = []
    env.close()
    for ep in range(10):
        env = make_env(seed=1000 + ep)
        t0 = time.time()
        ok, steps = scripted_lift_episode(env)
        dt = time.time() - t0
        results.append(ok)
        print(f"  ep{ep}: success={ok} steps={steps} wall={dt:.1f}s")
        env.close()
    print(f"\n脚本验证成功率: {sum(results)}/10")
    with open("verify_robosuite_result.json", "w") as f:
        json.dump({"success": int(sum(results)), "episodes": 10}, f)


if __name__ == "__main__":
    main()
