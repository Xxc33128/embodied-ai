#!/usr/bin/env python3
"""
run_isaac_onnx.py — Humanoid Lab dance_9 直接 ONNX runner（Isaac 侧，本周替代链）

R1 修订（2026-08-27，依据 2026-08-27 修订计划 §R1）：
  - CLI 增加 --mode as_shipped|aligned、--start-frame、--initial-state、--history-init、--obs-source
  - --motion 真实控制 env 的 motion_file（不再硬编码仓库路径）
  - canonical 契约写入 runtime：action term use_default_offset=False + offset=metadata default q；
    创建后从 runtime 回读 default q / offset / scale / Kp / Kd
  - 每周期记录 actual processed joint position target，并与 recorded q_des 对齐
  - aligned 模式用 deploy MimicObsBuilder 构造观测（与 MuJoCo 侧同一实现），history repeat-first
  - --initial-state 写入共同初态并回读；--make-initial-state 从 reset 后 readback 生成 canonical npz
  - 失败判定后立即停止（不再多执行一个 control interval）
  - world_to_init_motion_quaternion 保存完整 4 维
  - 记录 14-body reference/actual 状态（body 主任务指标用）
  - 显式 seed；manifest 含模式/哈希/commit/回读参数
  - 非 0/20ms 延迟拒绝；20ms 仅当原型已验证（默认拒绝非零，见下）

正式物理对比固定： --mode aligned --start-frame 0 --history-init repeat-first --obs-source deploy-builder

用法：
  E:\\isaaclab_env51\\Scripts\\python.exe scripts\\experiments\\run_isaac_onnx.py \
    --task Tracking-Flat-L7_29Dof-v0 \
    --policy deploy\\policy\\dance_9\\motion_anchor_obs_model_23500.onnx \
    --motion  deploy\\policy\\dance_9\\dance_9.npz \
    --mode aligned --obs-source deploy-builder --history-init repeat-first \
    --duration 5 --seed 42 --output <run_dir> [--record-video] [--save-interface-state-count 104] \
    --headless --/renderer/activeApi=d3d12
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import onnxruntime as ort
import yaml

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "yes")

REPO = Path(__file__).resolve().parents[2]  # E:\humanoid-lab
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_mujoco_onnx import (  # noqa: E402
    ADAPTER_BASE_ANG_B,
    ADAPTER_BASE_LIN,
    ADAPTER_STATES,
    MimicObsBuilder,
    MimicOnnxPolicy,
    load_contract,
    map_by_name,
    quat_apply,
    quat_apply_inverse,
    quat_inv,
    quat_mul,
    yaw_quaternion,
)

# ------------------------------------------------ argparse + AppLauncher（顺序必须先于 isaaclab import）
ap = argparse.ArgumentParser(description="Isaac direct ONNX runner (Lab dance_9, R1)")
ap.add_argument("--task", default="Tracking-Flat-L7_29Dof-v0")
ap.add_argument("--policy", required=True)
ap.add_argument("--motion", required=True)
ap.add_argument("--duration", type=float, default=5.0)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--fixed-delay-ms", type=float, default=0.0)
ap.add_argument("--gain-scale", type=float, default=1.0)
ap.add_argument("--mode", choices=["as_shipped", "aligned"], default="aligned")
ap.add_argument("--start-frame", type=int, default=0)
ap.add_argument("--initial-state", default=None, help="canonical_initial_state.npz 写入并回读")
ap.add_argument("--make-initial-state", default=None, help="输出 reset 后 readback 的 canonical npz 后退出")
ap.add_argument("--adapter-test", default=None, help="G0b：4 组静态注入状态 dump canonical readback 到 npz（主循环前）")
ap.add_argument("--history-init", choices=["zero", "repeat-first"], default="repeat-first")
ap.add_argument("--obs-source", choices=["native", "deploy-builder"], default="deploy-builder")
ap.add_argument("--output", required=True)
ap.add_argument("--record-video", action="store_true")
ap.add_argument("--save-interface-state-count", type=int, default=0)
from isaaclab.app import AppLauncher  # noqa: E402
AppLauncher.add_app_launcher_args(ap)
_args, _unknown = ap.parse_known_args()
_args.kit_args = (_args.kit_args or "") + " --/renderer/activeApi=d3d12"
_args.headless = True
_args.enable_cameras = bool(_args.record_video)
launcher = AppLauncher(_args)
from isaacsim.simulation_app import SimulationApp  # noqa: E402

# ------------------------------------------------ runtime imports（Kit 启动后）
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from era_okcc_humanoid_lab.tasks.mimic.robots.l7_29dof.l7_29dof_tracking_env_cfg import L7_29DofTrackingEnvCfg  # noqa: E402

if _args.record_video:
    from isaaclab.sensors import CameraCfg  # noqa: E402
    from isaaclab.sim import PinholeCameraCfg  # noqa: E402

import re  # noqa: E402


def sha256_bytes(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def repo_stamp(repo):
    try:
        head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True, timeout=10)
        return {"commit": head.stdout.strip(), "dirty": bool(dirty.stdout.strip()), "dirty_files": len(dirty.stdout.splitlines())}
    except Exception as e:
        return {"commit": "unavailable", "dirty": None, "error": str(e)}


def main():
    out_dir = Path(_args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "stdout.log"
    logf = open(log_path, "w", encoding="utf-8")
    def log(msg=""):
        print(msg, flush=True)
        logf.write(str(msg) + "\n")
        logf.flush()

    t0 = time.time()
    if _args.fixed_delay_ms != 0.0:
        raise SystemExit(
            f"--fixed-delay-ms={_args.fixed_delay_ms} 被拒绝：N2 起两侧在 R7 前统一禁止非零延迟"
            f"（MuJoCo 需 --delay-self-test 通过+启动前历史定义统一后才能启用 20ms）。"
        )
    if _args.mode == "as_shipped" and _args.obs_source == "deploy-builder":
        log("WARN as_shipped 模式强制 --obs-source native（保留官方语义）")
        _args.obs_source = "native"
    if _args.mode == "as_shipped" and _args.history_init != "zero":
        log("WARN as_shipped 模式强制 --history-init zero（官方前置补零语义）")
        _args.history_init = "zero"

    policy = MimicOnnxPolicy(_args.policy, log=log)  # 29 joints；与部署侧同源解析
    motion = dict(np.load(_args.motion))
    motion_len = motion["joint_pos"].shape[0]
    n_cycles = int(round(_args.duration / 0.02))
    delay_steps = max(0, int(round(_args.fixed_delay_ms / 5.0)))  # isaac physics dt=5ms
    log(f"policy joints={len(policy.joint_names)} motion frames={motion_len} cycles={n_cycles} "
        f"delay_steps={delay_steps} mode={_args.mode} obs_source={_args.obs_source} "
        f"history_init={_args.history_init} start_frame={_args.start_frame}")

    # ---------------- env cfg 覆写
    env_cfg = L7_29DofTrackingEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.dt = 0.005
    env_cfg.decimation = 4
    env_cfg.episode_length_s = float(_args.duration) + 2.0
    env_cfg.commands.motion.motion_file = str(Path(_args.motion).resolve())  # R1：CLI 控制 motion
    env_cfg.observations.policy.enable_corruption = False
    for name in list(vars(env_cfg.events).keys()):
        if not name.startswith("__"):
            setattr(env_cfg.events, name, None)
    for name in list(vars(env_cfg.terminations).keys()):
        if not name.startswith("__"):
            setattr(env_cfg.terminations, name, None)
    env_cfg.commands.motion.pose_range = {k: (0.0, 0.0) for k in ["x", "y", "z", "roll", "pitch", "yaw"]}
    env_cfg.commands.motion.velocity_range = {k: (0.0, 0.0) for k in ["x", "y", "z", "roll", "pitch", "yaw"]}
    env_cfg.commands.motion.joint_position_range = (0.0, 0.0)
    # R1：canonical contract 写入 action term —— scale + offset（use_default_offset=False）
    env_cfg.actions.joint_pos.scale = {n: float(s) for n, s in zip(policy.joint_names, policy.action_scale)}
    env_cfg.actions.joint_pos.use_default_offset = False
    env_cfg.actions.joint_pos.offset = {n: float(q) for n, q in zip(policy.joint_names, policy.default_q)}
    # actuator delay 固定 + 增益 = canonical
    name2kp = dict(zip(policy.joint_names, (policy.kp * _args.gain_scale).tolist()))
    name2kd = dict(zip(policy.joint_names, (policy.kd * _args.gain_scale).tolist()))
    for gname, g in env_cfg.scene.robot.actuators.items():
        g.min_delay = g.max_delay = delay_steps
        matched = []
        for expr in g.joint_names_expr:
            for j in policy.joint_names:
                if re.fullmatch(expr, j) and j not in matched:
                    matched.append(j)
        if matched:
            g.stiffness = {j: name2kp[j] for j in matched}
            g.damping = {j: name2kd[j] for j in matched}
            log(f"actuator[{gname}] joints={len(matched)} delay={delay_steps}")
    if _args.record_video:
        env_cfg.scene.camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera",
            update_period=0.0,
            height=720, width=1280,
            data_types=["rgb"],
            spawn=PinholeCameraCfg(clipping_range=(0.1, 30.0)),
        )

    # ---------------- 创建环境
    env = ManagerBasedRLEnv(cfg=env_cfg)
    art = env.scene.articulations["robot"]
    cmd = env.command_manager.get_term("motion")

    def _fixed_sampling(self, env_ids):
        self.time_steps[env_ids] = _args.start_frame
    cmd._adaptive_sampling = types.MethodType(_fixed_sampling, cmd)

    env.reset()
    log(f"env ready  t={time.time()-t0:.1f}s, dofs={len(art.joint_names)}")

    # 运行时关节序核对（契约要求：isaac order == onnx metadata order）
    rt_names = list(art.joint_names)
    assert len(rt_names) == 29, f"runtime dofs {len(rt_names)} != 29"
    order_ok = rt_names == policy.joint_names
    if not order_ok:
        miss = [n for n in policy.joint_names if n not in rt_names]
        extra = [n for n in rt_names if n not in policy.joint_names]
        raise SystemExit(f"runtime joint order != ONNX metadata（必须显式兼容表修复后才能跑 ALIGNED）: "
                         f"缺{miss} 多{extra}")
    log(f"runtime joint order == onnx metadata order: {order_ok}")

    # runtime 回读（R1）：default q / action offset / scale / Kp / Kd —— 全部 runtime 序
    jp = env.action_manager.get_term("joint_pos")
    default_rt = art.data.default_joint_pos[0].cpu().numpy().astype(np.float32)
    offset_rt = jp._offset[0].cpu().numpy().astype(np.float32) if isinstance(jp._offset, torch.Tensor) else float(jp._offset)
    scale_rt = jp._scale[0].cpu().numpy().astype(np.float32) if isinstance(jp._scale, torch.Tensor) else float(jp._scale)
    if isinstance(offset_rt, (float, int)):
        offset_rt = np.full(29, float(offset_rt), dtype=np.float32)
    if isinstance(scale_rt, (float, int)):
        scale_rt = np.full(29, float(scale_rt), dtype=np.float32)
    offset_pol = np.array([offset_rt[rt_names.index(n)] for n in policy.joint_names], dtype=np.float32)
    default_pol = np.array([default_rt[rt_names.index(n)] for n in policy.joint_names], dtype=np.float32)
    off_err = float(np.max(np.abs(offset_pol - policy.default_q)))
    default_err = float(np.max(np.abs(default_pol - policy.default_q)))
    log(f"runtime offset vs metadata default_q max_err={off_err:.2e} (default_q runtime max_err={default_err:.2e})")
    # P0-5/N2 审定：requested 值与 runtime readback 分离（不允许用 metadata 重构冒充 readback）
    kp_req_rt = np.array([policy.kp[policy.joint_names.index(n)] for n in rt_names], dtype=np.float32)
    kd_req_rt = np.array([policy.kd[policy.joint_names.index(n)] for n in rt_names], dtype=np.float32)
    scale_rt_named = np.array([policy.action_scale[policy.joint_names.index(n)] for n in rt_names], dtype=np.float32)

    def _rt_readback(name):
        """N3.1-P1.1：逐字段独立 readback；shape 必须 (29,) 且全 finite，否则返回 None。"""
        try:
            v = np.asarray(getattr(art.data, name)[0].cpu().numpy(), dtype=np.float32)
            if v.shape != (29,) or not np.isfinite(v).all():
                return None
            return v
        except Exception:
            return None

    rb_defs = [("kp", "joint_stiffness"), ("kd", "joint_damping"), ("armature", "joint_armature"),
               ("effort_limit", "joint_effort_limits"), ("velocity_limit", "joint_vel_limits")]
    readbacks = {}
    for tag, attr in rb_defs:
        v = _rt_readback(attr)
        readbacks[tag] = v
        log(f"runtime readback [{tag}]: "
            + (f"OK {v.round(4).tolist()[:4]}" if v is not None else "UNAVAILABLE"))

    # 失败谓词辅助
    EE_BODIES = ["left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_roll_link", "right_wrist_roll_link"]
    cmd_body_name_index = {n: i for i, n in enumerate(cmd.cfg.body_names)}
    assert all(n in cmd_body_name_index for n in EE_BODIES), "EE bodies missing in cfg.body_names"

    # ---------------- 初态处理：生成 / 写入 / 回读 ----------------
    def curr_state_policy():
        q_rt = art.data.joint_pos[0].cpu().numpy().astype(np.float32)
        dq_rt = art.data.joint_vel[0].cpu().numpy().astype(np.float32)
        return (np.array([q_rt[rt_names.index(n)] for n in policy.joint_names], dtype=np.float32),
                np.array([dq_rt[rt_names.index(n)] for n in policy.joint_names], dtype=np.float32))

    def apply_canonical_state(q_pol_np, dq_pol_np, root_pos_np, root_quat_np, lin_w_np, ang_b_np):
        """把 canonical（policy 序）写进 runtime，写入后 art.data 即时反映。"""
        q_pol = torch.as_tensor(np.asarray(q_pol_np, dtype=np.float32), device=env.device).reshape(1, -1)
        dq_pol = torch.as_tensor(np.asarray(dq_pol_np, dtype=np.float32), device=env.device).reshape(1, -1)
        q_rt_t = torch.cat([q_pol[:, policy.joint_names.index(n):policy.joint_names.index(n) + 1]
                            for n in rt_names], dim=1)
        dq_rt_t = torch.cat([dq_pol[:, policy.joint_names.index(n):policy.joint_names.index(n) + 1]
                             for n in rt_names], dim=1)
        root_pos = torch.as_tensor(np.asarray(root_pos_np, dtype=np.float64), device=env.device).reshape(1, 3)
        root_quat = torch.as_tensor(np.asarray(root_quat_np, dtype=np.float64), device=env.device).reshape(1, 4)
        root_lin_vel_w = torch.as_tensor(np.asarray(lin_w_np, dtype=np.float64), device=env.device).reshape(1, 3)
        root_ang_vel_b = torch.as_tensor(np.asarray(ang_b_np, dtype=np.float64), device=env.device).reshape(1, 3)
        art.write_joint_state_to_sim(q_rt_t, dq_rt_t)
        ang_w_np = quat_apply(root_quat[0].cpu().numpy(), root_ang_vel_b[0].cpu().numpy())  # body->world (isaac 语义)
        ang_w = torch.as_tensor(ang_w_np, dtype=torch.float64, device=env.device).reshape(1, 3)
        art.write_root_pose_to_sim(torch.cat([root_pos, root_quat], dim=-1))
        art.write_root_velocity_to_sim(torch.cat([root_lin_vel_w, ang_w], dim=-1))

    if _args.make_initial_state:
        q_pol, dq_pol = curr_state_policy()
        root_pos = art.data.root_pos_w[0].cpu().numpy().astype(np.float64)
        root_quat = art.data.root_quat_w[0].cpu().numpy().astype(np.float64)
        root_lin_vel_w = art.data.root_lin_vel_w[0].cpu().numpy().astype(np.float64)
        root_ang_vel_b = art.data.root_ang_vel_b[0].cpu().numpy().astype(np.float64)
        np.savez(_args.make_initial_state,
                 joint_names=np.array(policy.joint_names),
                 joint_pos=q_pol, joint_vel=dq_pol,
                 root_pos=root_pos, root_quat=root_quat,
                 root_lin_vel_w=root_lin_vel_w, root_ang_vel_b=root_ang_vel_b,
                 source="isaac_reset_readback", start_frame=_args.start_frame,
                 task=_args.task)
        log(f"canonical initial state written: {_args.make_initial_state}")
        log(f"  q[:6]={q_pol[:6]} root_z={root_pos[2]:.4f} root_quat={root_quat}")
        logf.close()
        os._exit(0)

    init_from = "motion_frame0"
    init_readback = None
    if _args.initial_state:
        init = np.load(_args.initial_state)
        assert list(init["joint_names"]) == list(policy.joint_names), "canonical npz joint order != policy"
        apply_canonical_state(init["joint_pos"], init["joint_vel"], init["root_pos"], init["root_quat"],
                              init["root_lin_vel_w"], init["root_ang_vel_b"])
        rq, rdq = curr_state_policy()
        q_err = float(np.max(np.abs(rq - np.asarray(init["joint_pos"], dtype=np.float32))))
        dq_err = float(np.max(np.abs(rdq - np.asarray(init["joint_vel"], dtype=np.float32))))
        init_from = "canonical_npz"
        init_readback = {
            "joint_pos_policy": rq.tolist(), "joint_vel_policy": rdq.tolist(),
            "root_pos": art.data.root_pos_w[0].cpu().numpy().tolist(),
            "root_quat": art.data.root_quat_w[0].cpu().numpy().tolist(),
            "root_lin_vel_w": art.data.root_lin_vel_w[0].cpu().numpy().tolist(),
            "root_ang_vel_b": art.data.root_ang_vel_b[0].cpu().numpy().tolist(),
            "q_max_err_vs_requested": q_err, "dq_max_err_vs_requested": dq_err,
            "npz_sha256": sha256_bytes(_args.initial_state),
        }
        log(f"initial state applied: q_max_err={q_err:.2e} dq_max_err={dq_err:.2e}")

    # ---- G0b adapter test（注入 4 组静态状态读回；结束后恢复 canonical 初态并继续主循环）----
    if _args.adapter_test:
        base_q_pol, base_dq_pol = curr_state_policy()
        base_root_pos = art.data.root_pos_w[0].cpu().numpy().copy()
        npz_keys = {}
        for nm, quat in ADAPTER_STATES.items():
            apply_canonical_state(base_q_pol, base_dq_pol, base_root_pos, quat,
                                  ADAPTER_BASE_LIN, ADAPTER_BASE_ANG_B)
            rq_pol, rdq_pol = curr_state_policy()
            vals = {"joint_pos": rq_pol, "joint_vel": rdq_pol,
                    "root_pos": art.data.root_pos_w[0].cpu().numpy().copy(),
                    "root_quat": art.data.root_quat_w[0].cpu().numpy().copy(),
                    "root_lin_vel_w": art.data.root_lin_vel_w[0].cpu().numpy().copy(),
                    "root_ang_vel_b": art.data.root_ang_vel_b[0].cpu().numpy().copy()}
            for fld, arr in vals.items():
                npz_keys[f"state_{nm}_{fld}"] = np.asarray(arr, dtype=np.float32)
        np.savez(_args.adapter_test, **npz_keys)
        log(f"G0b adapter dump written: {_args.adapter_test} ({len(npz_keys)} keys)")
        # 恢复 canonical 初态（若提供），保证主循环从同一初态开始
        if _args.initial_state:
            init = np.load(_args.initial_state)
            apply_canonical_state(init["joint_pos"], init["joint_vel"], init["root_pos"], init["root_quat"],
                                  init["root_lin_vel_w"], init["root_ang_vel_b"])
            log("adapter done -> canonical initial state restored")

    # ---------------- aligned 观测构造器（deploy-builder） ----------------
    obs_builder = None
    if _args.obs_source == "deploy-builder":
        contract_cfg = REPO / "deploy/src/era_rl_controller/configs/mimic_dance_9.yaml"
        contract = load_contract(str(contract_cfg))
        assert contract["obs_total"] == 770
        obs_builder = MimicObsBuilder(policy, motion, contract["obs_groups"])
        obs_builder.time_step = int(_args.start_frame)
        if _args.history_init == "repeat-first":
            rq, rdq = curr_state_policy()
            root_quat0 = art.data.root_quat_w[0].cpu().numpy().astype(np.float32)
            root_ang0 = art.data.root_ang_vel_b[0].cpu().numpy().astype(np.float32)
            obs_builder.prefill_repeat_first(root_quat0, root_ang0, rq, rdq)
            log("obs prefill: repeat-first (5 槽同第 0 帧 terms)")

    # 相机
    camera = None
    if _args.record_video:
        camera = env.scene.sensors["camera"]
        camera.set_world_poses(np.array([[0.0, -2.9, 1.2]], dtype=np.float64),
                               np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64))
        try:
            env.sim.render()
            camera.update(0.001)
            img0 = camera.data.output["rgb"][0].cpu().numpy()
            log(f"camera smoke: shape={img0.shape} max={int(img0.max())} black={'YES' if img0.max() <= 5 else 'no'}")
        except Exception as e:
            log(f"WARN camera smoke failed: {e}")

    def look_at(pos, target):
        fwd = target - pos
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        R = np.stack([right, up, -fwd], axis=1)
        tr = np.trace(R)
        qw = math.sqrt(max(0.0, 1.0 + tr)) / 2.0
        qx = (R[2, 1] - R[1, 2]) / (4 * qw) if qw > 1e-8 else 0.0
        qy = (R[0, 2] - R[2, 0]) / (4 * qw) if qw > 1e-8 else 0.0
        qz = (R[1, 0] - R[0, 1]) / (4 * qw) if qw > 1e-8 else 0.0
        return np.array([qw, qx, qy, qz], dtype=np.float64)

    # ---------------- 主循环 ----------------
    rec = {k: [] for k in ["policy_step", "t", "time_step", "obs(770)", "action(29)", "q_des(29)",
                           "q(29)", "dq(29)", "root_pos(3)", "root_quat(4)",
                           "root_lin_vel(3)", "root_ang_vel(3)", "actual_processed_q_target(29)",
                           "reference_body_pos_w(14,3)", "reference_body_quat_w(14,4)",
                           "reference_body_lin_vel_w(14,3)", "reference_body_ang_vel_w(14,3)",
                           "actual_body_pos_w(14,3)", "actual_body_quat_w(14,4)",
                           "actual_body_lin_vel_w(14,3)", "actual_body_ang_vel_w(14,3)",
                           "done", "fall_reason"]}
    iface = {k: [] for k in ["policy_step", "reference_time", "root_quaternion_wxyz", "root_angular_velocity_body",
                             "joint_position_in_policy_order", "joint_velocity_in_policy_order",
                             "previous_action", "full_isaac_observation_770", "history_valid_length",
                             "reference_joint_position", "reference_joint_velocity",
                             "reference_anchor_quaternion", "world_to_init_motion_quaternion",
                             "obs_source"]} \
        if _args.save_interface_state_count > 0 else None

    fall_reason = None
    video_frames = []
    last_vframe = -1e9
    reset_count = 0

    np.random.seed(_args.seed)
    torch.manual_seed(_args.seed)

    for cyc in range(n_cycles):
        t = cyc * 0.02
        # state_t（runtime 序 -> policy 序）
        root_pos = art.data.root_pos_w[0].cpu().numpy().astype(np.float32)
        root_quat = art.data.root_quat_w[0].cpu().numpy().astype(np.float32)          # wxyz
        root_ang_vel = art.data.root_ang_vel_b[0].cpu().numpy().astype(np.float32)    # body
        root_lin_vel = art.data.root_lin_vel_w[0].cpu().numpy().astype(np.float32)
        q_rt = art.data.joint_pos[0].cpu().numpy().astype(np.float32)
        dq_rt = art.data.joint_vel[0].cpu().numpy().astype(np.float32)
        ref_t = int(cmd.time_steps[0].item())
        q_pol = np.array([q_rt[rt_names.index(n)] for n in policy.joint_names], dtype=np.float32)
        dq_pol = np.array([dq_rt[rt_names.index(n)] for n in policy.joint_names], dtype=np.float32)
        # obs_t
        if obs_builder is not None:
            obs, ref_t2 = obs_builder.build_obs(root_quat, root_ang_vel, q_pol, dq_pol)
            assert ref_t2 == ref_t, f"builder ref_t {ref_t2} != cmd ref_t {ref_t}"
        else:
            obs = env.observation_manager.compute_group("policy")[0].cpu().numpy().astype(np.float32)
        assert obs.shape[0] == 770, obs.shape
        prev_act_pol = np.array([env.action_manager.action[0].cpu().numpy()[rt_names.index(n)]
                                 for n in policy.joint_names], dtype=np.float32)

        # 失败判定（与 MuJoCo 侧同公式同 ref_t）
        reason = None
        anc_ref_p = cmd.anchor_pos_w[0].cpu().numpy().astype(np.float64)
        anc_ref_q = cmd.anchor_quat_w[0].cpu().numpy().astype(np.float64)
        rob_anc_p = cmd.robot_anchor_pos_w[0].cpu().numpy().astype(np.float64)
        rob_anc_q = cmd.robot_anchor_quat_w[0].cpu().numpy().astype(np.float64)
        if abs(float(anc_ref_p[2]) - float(rob_anc_p[2])) > 0.4:
            reason = "anchor_z"
        if reason is None:
            g_ref = quat_apply_inverse(anc_ref_q, np.array([0.0, 0.0, -1.0]))
            g_rob = quat_apply_inverse(rob_anc_q, np.array([0.0, 0.0, -1.0]))
            if abs(float(g_ref[2]) - float(g_rob[2])) > 0.8:
                reason = "anchor_ori"
        if reason is None:
            bw = cmd.body_pos_w[0].cpu().numpy().astype(np.float64)
            rb = cmd.robot_body_pos_w[0].cpu().numpy().astype(np.float64)
            drot = yaw_quaternion(quat_mul(rob_anc_q, quat_inv(anc_ref_q)))
            for nm in EE_BODIES:
                i = cmd_body_name_index.get(nm)
                if i is None:
                    continue
                ref_z = float(anc_ref_p[2]) + float(quat_apply(drot, bw[i] - anc_ref_p)[2])
                if abs(ref_z - float(rb[i, 2])) > 0.4:
                    reason = "ee_body_z_" + nm
                    break
        if reason is None and (not np.isfinite(obs).all() or not np.isfinite(q_pol).all() or not np.isfinite(root_quat).all()):
            reason = "nan"
        # action_t
        action = policy.infer(obs, float(ref_t))
        policy.action_buffer[:] = action
        q_des_pol = policy.q_des_from_action(action)
        # 记录（R1：fail-stop 先记录再决定是否 step）
        rec["policy_step"].append(cyc)
        rec["t"].append(t)
        rec["time_step"].append(ref_t)
        rec["obs(770)"].append(obs)
        rec["action(29)"].append(action)
        rec["q_des(29)"].append(q_des_pol)
        rec["q(29)"].append(q_pol)
        rec["dq(29)"].append(dq_pol)
        rec["root_pos(3)"].append(root_pos)
        rec["root_quat(4)"].append(root_quat)
        rec["root_lin_vel(3)"].append(root_lin_vel)
        rec["root_ang_vel(3)"].append(root_ang_vel)
        rec["done"].append(1 if reason else 0)
        rec["fall_reason"].append(reason or "")
        # 14-body reference/actual（body 主任务指标）
        rt_idx = cmd.body_indexes.cpu().numpy()
        ref_bp = cmd.motion.body_pos_w[ref_t].cpu().numpy().astype(np.float32)
        ref_bq = cmd.motion.body_quat_w[ref_t].cpu().numpy().astype(np.float32)
        ref_blv = cmd.motion.body_lin_vel_w[ref_t].cpu().numpy().astype(np.float32)
        ref_bav = cmd.motion.body_ang_vel_w[ref_t].cpu().numpy().astype(np.float32)
        act_bp = art.data.body_pos_w[0, rt_idx].cpu().numpy().astype(np.float32)
        act_bq = art.data.body_quat_w[0, rt_idx].cpu().numpy().astype(np.float32)
        act_blv = art.data.body_lin_vel_w[0, rt_idx].cpu().numpy().astype(np.float32)
        act_bav = art.data.body_ang_vel_w[0, rt_idx].cpu().numpy().astype(np.float32)
        rec["reference_body_pos_w(14,3)"].append(ref_bp)
        rec["reference_body_quat_w(14,4)"].append(ref_bq)
        rec["reference_body_lin_vel_w(14,3)"].append(ref_blv)
        rec["reference_body_ang_vel_w(14,3)"].append(ref_bav)
        rec["actual_body_pos_w(14,3)"].append(act_bp)
        rec["actual_body_quat_w(14,4)"].append(act_bq)
        rec["actual_body_lin_vel_w(14,3)"].append(act_blv)
        rec["actual_body_ang_vel_w(14,3)"].append(act_bav)

        # golden states
        if iface is not None and len(iface["policy_step"]) < _args.save_interface_state_count:
            iw = yaw_quaternion(root_quat.astype(np.float64))
            ia = yaw_quaternion(motion["body_quat_w"][max(ref_t, 0), 0].astype(np.float64))
            wtim = quat_mul(iw, quat_inv(ia)).astype(np.float32)  # R1：完整 (4,)
            iface["policy_step"].append(cyc)
            iface["reference_time"].append(t)
            iface["root_quaternion_wxyz"].append(root_quat)
            iface["root_angular_velocity_body"].append(root_ang_vel)
            iface["joint_position_in_policy_order"].append(q_pol)
            iface["joint_velocity_in_policy_order"].append(dq_pol)
            iface["previous_action"].append(prev_act_pol)
            iface["full_isaac_observation_770"].append(obs)
            iface["history_valid_length"].append(min(cyc + 1, 5))
            iface["reference_joint_position"].append(np.asarray(cmd.joint_pos[0].cpu().numpy(), dtype=np.float32))
            iface["reference_joint_velocity"].append(np.asarray(cmd.joint_vel[0].cpu().numpy(), dtype=np.float32))
            iface["reference_anchor_quaternion"].append(cmd.anchor_quat_w[0].cpu().numpy().astype(np.float32))
            iface["world_to_init_motion_quaternion"].append(wtim)
            iface["obs_source"].append(_args.obs_source)

        # 相机
        if camera is not None and t >= last_vframe + 1 / 30.0 - 1e-6:
            pos = root_pos.astype(np.float64) + np.array([0.35, -2.9, 0.45])
            tgt = root_pos.astype(np.float64) + np.array([0.0, 0.0, 0.75])
            camera.set_world_poses(pos[None, :], look_at(pos, tgt)[None, :])
            env.sim.render()
            camera.update(dt=0.02)
            img = camera.data.output["rgb"][0].cpu().numpy()
            video_frames.append(img)
            if len(video_frames) == 10 and img.max() <= 5:
                log("WARN: first 10 camera frames appear BLACK")
            last_vframe = t

        # R1：actual processed target —— 每周期恰追加一次；失败周期未 step，记 q_des（未执行）并在 fall_reason 说明
        act_rt = np.array([action[policy.joint_names.index(n)] for n in rt_names], dtype=np.float32) if not order_ok else action
        act_t = torch.tensor(act_rt[None, :], device=env.device, dtype=torch.float32)
        if reason is not None:
            rec["actual_processed_q_target(29)"].append(q_des_pol.copy())
            fall_reason = reason
            log(f"FALL at cycle {cyc} t={t:.2f}s reason={reason} (recorded; 不 step)")
            break
        env.step(act_t)
        if obs_builder is not None:
            obs_builder.advance()  # 与 MuJoCo 侧一致：每执行一个控制周期推进一帧
        proc_rt = jp.processed_actions[0].cpu().numpy().astype(np.float32)
        proc_pol = np.array([proc_rt[rt_names.index(n)] for n in policy.joint_names], dtype=np.float32)
        rec["actual_processed_q_target(29)"].append(proc_pol)
        if (cyc + 1) % 100 == 0:
            log(f"cycle {cyc+1}/{n_cycles} t={t:.1f}s wall={time.time()-t0:.1f}s")

    # 说明：Isaac 记录“该周期 step 后 processed target”（= 该周期实际送入 actuator 的目标）；
    # 失败周期未执行 step，target 记录为 q_des 且不会进入执行器。逐帧对齐检查见 G0a。

    # ---------------- 落盘 ----------------
    npz_out = {}
    for k, v in rec.items():
        if k in ("policy_step", "t", "time_step", "done"):
            npz_out[k] = np.asarray(v, dtype=np.float32)
        elif k == "fall_reason":
            npz_out[k] = np.asarray(v)
        elif k.startswith("reference_body") or k.startswith("actual_body"):
            npz_out[k] = np.asarray(v, dtype=np.float32)
        else:
            npz_out[k] = np.asarray(v, dtype=np.float32)
    np.savez_compressed(out_dir / "policy_trace.npz", **npz_out)
    if iface is not None:
        np.savez_compressed(out_dir / "isaac_golden_states.npz",
                            **{k: np.asarray(v) for k, v in iface.items()})
    manifest = {
        "engine": "isaac", "task": _args.task, "duration_s": _args.duration, "control_dt": 0.02,
        "physics_dt": 0.005, "decimation": 4, "cycles_planned": n_cycles, "cycles_done": len(rec["policy_step"]),
        "fall_reason": fall_reason, "seed": _args.seed, "seed_used": {"numpy": _args.seed, "torch": _args.seed},
        "fixed_delay_ms": _args.fixed_delay_ms, "gain_scale": _args.gain_scale, "num_envs": 1,
        "episode_length_s": env.cfg.episode_length_s, "reset_count": reset_count,
        "mode": _args.mode, "start_frame": _args.start_frame, "history_init": _args.history_init,
        "obs_source": _args.obs_source, "initial_state_source": init_from,
        "initial_state_npz_sha256": (sha256_bytes(_args.initial_state) if _args.initial_state else None),
        "runtime_joint_order_matches_onnx": bool(order_ok),
        "runtime_default_q": default_rt.tolist(),
        "action_term_offset_rt": offset_rt.tolist(),
        "action_term_scale_rt": scale_rt.tolist(),
        "offset_vs_metadata_default_q_max_err": off_err,
        "runtime_default_vs_metadata_default_q_max_err": default_err,
        "kp_requested_from_metadata_rt_order": kp_req_rt.tolist(),
        "kd_requested_from_metadata_rt_order": kd_req_rt.tolist(),
        "kp_runtime_readback_rt_order": (readbacks["kp"].tolist() if readbacks["kp"] is not None else "UNAVAILABLE"),
        "kp_runtime_readback_status": ("OK" if readbacks["kp"] is not None else "UNAVAILABLE"),
        "kd_runtime_readback_rt_order": (readbacks["kd"].tolist() if readbacks["kd"] is not None else "UNAVAILABLE"),
        "kd_runtime_readback_status": ("OK" if readbacks["kd"] is not None else "UNAVAILABLE"),
        "armature_runtime_readback_rt_order": (readbacks["armature"].tolist() if readbacks["armature"] is not None else "UNAVAILABLE"),
        "armature_runtime_readback_status": ("OK" if readbacks["armature"] is not None else "UNAVAILABLE"),
        "effort_limit_runtime_readback_rt_order": (readbacks["effort_limit"].tolist() if readbacks["effort_limit"] is not None else "UNAVAILABLE"),
        "effort_limit_runtime_readback_status": ("OK" if readbacks["effort_limit"] is not None else "UNAVAILABLE"),
        "velocity_limit_runtime_readback_rt_order": (readbacks["velocity_limit"].tolist() if readbacks["velocity_limit"] is not None else "UNAVAILABLE"),
        "velocity_limit_runtime_readback_status": ("OK" if readbacks["velocity_limit"] is not None else "UNAVAILABLE"),
        "meta_default_q": policy.default_q.tolist(),
        "obs_total": int(env.observation_manager.compute_group("policy").shape[-1]) if _args.obs_source == "native" else 770,
        "hashes": {
            "policy": sha256_bytes(_args.policy),
            "motion": sha256_bytes(_args.motion),
            "script": sha256_bytes(__file__),
        },
        "repo": {"lab": repo_stamp(REPO)},
        "initial_state_readback": init_readback,
    }
    with open(out_dir / "run_manifest.yaml", "w", encoding="utf-8") as f:
        f.write(yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False, default_flow_style=False))
    if _args.record_video and video_frames:
        try:
            import imageio
            imageio.mimsave(out_dir / "video.mp4", video_frames, fps=30, codec="libx264", quality=8)
        except Exception as e:
            try:
                import cv2
                w = cv2.VideoWriter(str(out_dir / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 720))
                for f in video_frames:
                    w.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
                w.release()
            except Exception as e2:
                log(f"WARN video save failed: {e} / {e2}")
        log(f"video frames={len(video_frames)}")
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"fall_reason": fall_reason, "cycles_done": len(rec["policy_step"]),
                   "cycles_planned": n_cycles, "reset_count": reset_count,
                   "runtime_joint_order_matches_onnx": bool(order_ok),
                   "mode": _args.mode, "obs_source": _args.obs_source, "initial_state_source": init_from,
                   "offset_vs_metadata_default_q_max_err": off_err}, f, indent=2, ensure_ascii=False)
    log(f"DONE fall={fall_reason} cycles={len(rec['policy_step'])} wall={time.time()-t0:.1f}s -> {out_dir}")
    logf.close()
    os._exit(0)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        import traceback
        traceback.print_exc()
        try:
            with open(Path(_args.output) / "stdout.log", "a", encoding="utf-8") as f:
                f.write(traceback.format_exc())
        except Exception:
            pass
    os._exit(1)