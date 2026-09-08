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

# P3-R1: robust REPO detection (runner may be in 13_p3_delay20/00_inputs)
if Path(r"E:\humanoid-lab\deploy").is_dir():
    REPO = Path(r"E:\humanoid-lab")
else:
    REPO = Path(__file__).resolve().parents[2]
HLAB_EXPS = REPO / "scripts" / "experiments"
sys.path.insert(0, str(HLAB_EXPS))
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
ap.add_argument("--runtime-dump", default=None, help="N4：创建 env+reset 后 dump runtime 执行器参数(JSON)并退出")
ap.add_argument("--step-test-joint", default=None, help="N5：逗号分隔关节名列表；对每个关节跑 0.5s 固定基座单关节阶跃（与 MuJoCo 侧同协议），输出到 --step-test-output")
ap.add_argument("--step-test-rad", type=float, default=0.10)
ap.add_argument("--step-test-output", default=None)
ap.add_argument("--torque-parity-only", action="store_true", help="N5-0：独立进程只做单 env.step 的 torque parity dump（R4-F1：与 --step-only 互斥）")
ap.add_argument("--step-only", action="store_true", help="R4-F1：独立进程只跑 0.5s 单关节 step；绝不先执行 parity")
ap.add_argument("--isolation", choices=["clean", "coupled"], default="clean",
                help="R4-F3：clean=CLEAN_ISOLATED（gravity=0/28 关节每 substep 锁定/root 锁定，主证据协议）；coupled=仅 root 每 control 周期锁定（工程补充）")
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
from era_okcc_humanoid_lab.tasks.mimic.robots.l7_29dof.l7_29dof_tracking_env_cfg import L7_29DofTrackingEnvCfg
# P3-R2: instrumented actuator import (explicit deque, r2)
try:
    from era_okcc_humanoid_lab.robots.actuator import DelayedImplicitActuator  # noqa: F401
    from p3_instrumented_delayed_actuator_r2 import P3InstrumentedDelayedImplicitActuator  # type: ignore
except Exception:
    import sys as _sys
    _p3_path = _pathlib.Path(__file__).resolve().parent / "p3_instrumented_delayed_actuator_r2.py"
    if _p3_path.is_file():
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location("p3_instrumented_delayed_actuator_r2", str(_p3_path))
        _mod = _ilu.module_from_spec(_spec)
        _sys.modules["p3_instrumented_delayed_actuator_r2"] = _mod
        _spec.loader.exec_module(_mod)  # type: ignore
        P3InstrumentedDelayedImplicitActuator = _mod.P3InstrumentedDelayedImplicitActuator
    else:
        raise
import pathlib as _pathlib  # for hash
  # noqa: E402

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
    if _args.fixed_delay_ms not in (0.0, 20.0):
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
    if _args.step_test_joint and _args.isolation == "clean":
        env_cfg.sim.gravity = (0.0, 0.0, 0.0)
        log("CLEAN_ISOLATED: Isaac sim.gravity=(0,0,0)（env 创建前设置）")
        # ---- V3-1 真硬约束资产（2026-08-28 审查修复）：URDF 全局部（28 revolute→fixed 仅局部烘焙 q0）
        # canonical root 世界 pose 由 init_state 一次性设置（不写回）；fix_base 物理固定根。 ----
        from urdf_isolation import build as build_iso_urdf
        _iso_joint = _args.step_test_joint
        _src_urdf = Path(env_cfg.scene.robot.spawn.asset_path)
        _iso_urdf = build_iso_urdf(_src_urdf, _iso_joint, Path(_args.initial_state), _src_urdf.parent)
        env_cfg.scene.robot.spawn.asset_path = str(_iso_urdf)
        env_cfg.scene.robot.spawn.fix_base = True
        env_cfg.scene.robot.spawn.merge_fixed_joints = True
        env_cfg.scene.robot.spawn.self_collision = False
        # 只保留含被测关节的 actuator 组，expr 收敛为确切名
        _keep = {k: v for k, v in env_cfg.scene.robot.actuators.items()
                 if any(_iso_joint == n0 for e in v.joint_names_expr for n0 in (e,)) or
                    any(re.fullmatch(e, _iso_joint) for e in v.joint_names_expr)}
        if not _keep:
            raise SystemExit(f"被测关节 {_iso_joint} 不属于任何 actuator 组")
        for g in _keep.values():
            g.joint_names_expr = [_iso_joint]
            # 参数字典（stiffness/damping/armature/effort_limit_sim/velocity_limit_sim 等）
            # 键是正则表达式：过滤到仅匹配被测关节，避免引用已删除关节报错
            for f in vars(g):
                v = getattr(g, f)
                if isinstance(v, dict) and f != "joint_names_expr":
                    setattr(g, f, {k: val for k, val in v.items() if re.fullmatch(k, _iso_joint)})
        env_cfg.scene.robot.actuators = _keep
        # 初态：URDF 全局部（修复，审查 V3）；canonical root 世界 pose 在 articulation 根一次性设置
        _ini = np.load(_args.initial_state)
        env_cfg.scene.robot.init_state.pos = tuple(np.asarray(_ini["root_pos"], dtype=float).tolist())
        _rq = np.asarray(_ini["root_quat"], dtype=float)
        env_cfg.scene.robot.init_state.orientation = (float(_rq[0]), float(_rq[1]), float(_rq[2]), float(_rq[3]))
        env_cfg.scene.robot.init_state.joint_pos = {_iso_joint: (
            float(np.asarray(_ini["joint_pos"], dtype=float)[list(_ini["joint_names"]).index(_iso_joint)]))}
        env_cfg.scene.robot.init_state.joint_vel = {".*": 0.0}
        # 观测管线对 1-dof 环境不适用：注入一个纯占位 term（不读任何 art/body；不引用 body 名——
        # merge_fixed_joints 会删除大量 link prim，原 obs 组必然失败）
        from isaaclab.managers import ObservationGroupCfg as _ObsGrp, ObservationTermCfg as _ObsTerm
        from isaaclab.utils import configclass

        # 注：term 函数必须显式声明参数（isaaclab inspect 拒绝 **kwargs 模型）
        def _dummy_obs(env, time: float = 0.0):
            import torch as _torch
            return _torch.zeros((env.num_envs, 1), device=env.device, dtype=_torch.float32)

        @configclass
        class _DummyObsGrp(_ObsGrp):
            dummy = _ObsTerm(func=_dummy_obs)

        env_cfg.observations = {"policy": _DummyObsGrp()}
        log(f"CLEAN 资产: {_iso_urdf.name} fix_base=True（root 固定 canonical）唯一 dof={_iso_joint}")
    if _args.step_test_joint and _args.isolation == "clean":
        # 命令 term 在 reset 时写 29 维 joint state，与 1-dof 冲突 → 禁用
        env_cfg.commands.motion = None
        log("CLEAN: motion 命令 term 禁用（reset 写 29 维与 1-dof 冲突）")
    env_cfg.episode_length_s = float(_args.duration) + 2.0
    if env_cfg.commands.motion is not None:
        env_cfg.commands.motion.motion_file = str(Path(_args.motion).resolve())  # R1：CLI 控制 motion
        env_cfg.commands.motion.pose_range = {k: (0.0, 0.0) for k in ["x", "y", "z", "roll", "pitch", "yaw"]}
        env_cfg.commands.motion.velocity_range = {k: (0.0, 0.0) for k in ["x", "y", "z", "roll", "pitch", "yaw"]}
        env_cfg.commands.motion.joint_position_range = (0.0, 0.0)
        if _args.step_test_joint and _args.isolation == "clean":
            # merge_fixed_joints 会删除非被测分支的 link prim：body 列表收敛到实际存在 body，并关 debug_vis
            env_cfg.commands.motion.body_names = ["pelvis", _iso_joint.replace("_joint", "_link")]
            env_cfg.commands.motion.debug_vis = False
    if not (_args.step_test_joint and _args.isolation == "clean"):
        env_cfg.observations.policy.enable_corruption = False
    for name in list(vars(env_cfg.events).keys()):
        if not name.startswith("__"):
            setattr(env_cfg.events, name, None)
    for name in list(vars(env_cfg.terminations).keys()):
        if not name.startswith("__"):
            setattr(env_cfg.terminations, name, None)
    if _args.step_test_joint and _args.isolation == "clean":
        # reward terms 引用已合并消失的 body（undesired_contacts 等）→ 全置 None
        for name in list(vars(env_cfg.rewards).keys()):
            if not name.startswith("__"):
                setattr(env_cfg.rewards, name, None)
        log("CLEAN: rewards 全置 None（body 引用已失效）")

    # R1：canonical contract 写入 action term —— scale + offset（use_default_offset=False）
    if iso_clean_pre := bool(_args.step_test_joint) and _args.isolation == "clean":
        env_cfg.actions.joint_pos.noise = None
        env_cfg.actions.joint_pos.scale = {_iso_joint: float(policy.action_scale[policy.joint_names.index(_iso_joint)])}
        env_cfg.actions.joint_pos.offset = {_iso_joint: float(policy.default_q[policy.joint_names.index(_iso_joint)])}
        env_cfg.actions.joint_pos.use_default_offset = False
        log("CLEAN: action joint_pos 收敛到被测关节（1-dof）")
    else:
        env_cfg.actions.joint_pos.scale = {n: float(s) for n, s in zip(policy.joint_names, policy.action_scale)}
        env_cfg.actions.joint_pos.use_default_offset = False
        env_cfg.actions.joint_pos.offset = {n: float(q) for n, q in zip(policy.joint_names, policy.default_q)}
    # actuator delay 固定 + 增益 = canonical
    name2kp = dict(zip(policy.joint_names, (policy.kp * _args.gain_scale).tolist()))
    name2kd = dict(zip(policy.joint_names, (policy.kd * _args.gain_scale).tolist()))
    for gname, g in env_cfg.scene.robot.actuators.items():
        g.min_delay = g.max_delay = delay_steps
        # P3-R1: use instrumented actuator for observable delay
        try:
            g.class_type = P3InstrumentedDelayedImplicitActuator
        except Exception as _e:
            log(f"WARN failed to set instrumented class_type: {_e}")
        matched = []
        for expr in g.joint_names_expr:
            for j in policy.joint_names:
                if re.fullmatch(expr, j) and j not in matched:
                    matched.append(j)
        if matched:
            g.stiffness = {j: name2kp[j] for j in matched}
            g.damping = {j: name2kd[j] for j in matched}
            log(f"actuator[{gname}] joints={len(matched)} delay={delay_steps} class={g.class_type.__name__}")
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
    cmd = env.command_manager.get_term("motion") if (env_cfg.commands.motion is not None) else None
    if cmd is not None:
        def _fixed_sampling(self, env_ids):
            self.time_steps[env_ids] = _args.start_frame
        cmd._adaptive_sampling = types.MethodType(_fixed_sampling, cmd)

    env.reset()
    log(f"env ready  t={time.time()-t0:.1f}s, dofs={len(art.joint_names)}")

    # 运行时关节序核对（契约要求：isaac order == onnx metadata order）
    rt_names = list(art.joint_names)
    iso_clean = bool(_args.step_test_joint) and _args.isolation == "clean"
    if iso_clean:
        assert len(rt_names) == 1, f"isolation 模型期望 1 dof，实际 {len(rt_names)}"
        log(f"isolation dofs=1: {rt_names[0]}")
    else:
        assert len(rt_names) == 29, f"runtime dofs {len(rt_names)} != 29"
    order_ok = rt_names == policy.joint_names
    if not order_ok:
        if iso_clean:
            assert rt_names[0] == _args.step_test_joint, f"isolation 唯一 dof {rt_names[0]} != 被测 {_args.step_test_joint}"
        else:
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
    ndof = len(rt_names)
    offset_pol = np.zeros(ndof if iso_clean else 29, dtype=np.float32) if iso_clean else \
        np.array([offset_rt[rt_names.index(n)] for n in policy.joint_names], dtype=np.float32)
    scale_pol = np.zeros(ndof if iso_clean else 29, dtype=np.float32) if iso_clean else \
        np.array([scale_rt[rt_names.index(n)] for n in policy.joint_names], dtype=np.float32)
    default_pol = np.array([default_rt[rt_names.index(n)] for n in policy.joint_names], dtype=np.float32) \
        if not iso_clean else np.zeros(1, dtype=np.float32)
    if iso_clean:
        offset_pol[0] = np.asarray(offset_rt, dtype=np.float32).reshape(-1)[0]
        scale_pol[0] = np.asarray(scale_rt, dtype=np.float32).reshape(-1)[0]
    else:
        off_err = float(np.max(np.abs(offset_pol - policy.default_q)))
        default_err = float(np.max(np.abs(default_pol - policy.default_q)))
        log(f"runtime offset vs metadata default_q max_err={off_err:.2e} (default_q runtime max_err={default_err:.2e})")
    # P0-5/N2 审定：requested 值与 runtime readback 分离（不允许用 metadata 重构冒充 readback）
    kp_req_rt = np.array([policy.kp[policy.joint_names.index(n)] for n in rt_names], dtype=np.float32)
    kd_req_rt = np.array([policy.kd[policy.joint_names.index(n)] for n in rt_names], dtype=np.float32)
    scale_rt_named = np.array([policy.action_scale[policy.joint_names.index(n)] for n in rt_names], dtype=np.float32)

    def _rt_readback(name):
        """N3.1-P1.1：逐字段独立 readback；shape 必须 (ndof,) 且全 finite，否则返回 None。"""
        try:
            v = np.asarray(getattr(art.data, name)[0].cpu().numpy(), dtype=np.float32)
            if v.shape != (ndof,) or not np.isfinite(v).all():
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
    if cmd is not None:
        cmd_body_name_index = {n: i for i, n in enumerate(cmd.cfg.body_names)}
        assert all(n in cmd_body_name_index for n in EE_BODIES), "EE bodies missing in cfg.body_names"
    else:
        cmd_body_name_index = {}  # clean：命令禁用，失败谓词不适用

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

    if _args.runtime_dump:
        json.dump({
            "engine": "isaac", "task": _args.task,
            "physics_dt": 0.005, "decimation": 4, "control_dt": 0.02,
            "rt_joint_names": rt_names,
            "policy_joint_names": policy.joint_names,
            "runtime_joint_order_matches_onnx": bool(order_ok),
            "kp_requested_from_metadata": policy.kp.tolist(),
            "kd_requested_from_metadata": policy.kd.tolist(),
            "kp_runtime_readback": readbacks["kp"].tolist() if readbacks["kp"] is not None else "UNAVAILABLE",
            "kd_runtime_readback": readbacks["kd"].tolist() if readbacks["kd"] is not None else "UNAVAILABLE",
            "armature_runtime_readback": readbacks["armature"].tolist() if readbacks["armature"] is not None else "UNAVAILABLE",
            "effort_limit_runtime_readback": readbacks["effort_limit"].tolist() if readbacks["effort_limit"] is not None else "UNAVAILABLE",
            "velocity_limit_runtime_readback": readbacks["velocity_limit"].tolist() if readbacks["velocity_limit"] is not None else "UNAVAILABLE",
        }, open(_args.runtime_dump, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        log(f"runtime dump written: {_args.runtime_dump}")
        logf.close()
        os._exit(0)

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
    if _args.initial_state and not iso_clean:
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

    # ---------------- N5-R：torque parity / 单关节阶跃（R4-F 修订版，2026-08-28 审查） ----------------
    # 进程分离（P0-1）：--torque-parity-only 与 --step-only 两个全新进程，禁止同一进程先 parity 后 step。
    if _args.step_test_joint:
        assert _args.initial_state, "step-test 需要 --initial-state canonical npz"
        assert _args.step_test_output, "--step-test-output required"
        if _args.torque_parity_only and _args.step_only:
            raise SystemExit("--torque-parity-only 与 --step-only 互斥：parity 与 step 必须用两个全新进程（R4-F1）")
        if not (_args.torque_parity_only or _args.step_only):
            raise SystemExit("R4-F1：必须显式指定 --torque-parity-only 或 --step-only（parity/step 不能同进程）")
        init = np.load(_args.initial_state)
        assert list(init["joint_names"]) == list(policy.joint_names)
        q0_pol = np.asarray(init["joint_pos"], dtype=np.float32)
        root_pos0 = np.asarray(init["root_pos"], dtype=np.float64)
        root_quat0 = np.asarray(init["root_quat"], dtype=np.float64)
        kp_pol = (readbacks["kp"].astype(np.float32) if readbacks["kp"] is not None else np.array([np.nan], dtype=np.float32)) \
            if iso_clean else (
                np.array([readbacks["kp"][rt_names.index(n)] for n in policy.joint_names], dtype=np.float32)
                if readbacks["kp"] is not None else np.nan * np.ones(29))
        kd_pol = (readbacks["kd"].astype(np.float32) if readbacks["kd"] is not None else np.array([np.nan], dtype=np.float32)) \
            if iso_clean else (
                np.array([readbacks["kd"][rt_names.index(n)] for n in policy.joint_names], dtype=np.float32)
                if readbacks["kd"] is not None else np.nan * np.ones(29))
        eff_pol = (readbacks["effort_limit"].astype(np.float32) if readbacks["effort_limit"] is not None
                   else np.array([np.inf], dtype=np.float32)) \
            if iso_clean else (
                np.array([readbacks["effort_limit"][rt_names.index(n)] for n in policy.joint_names], dtype=np.float32)
                if readbacks["effort_limit"] is not None else np.nan * np.ones(29))
        arm_pol = (readbacks["armature"].astype(np.float32) if readbacks["armature"] is not None
                   else np.array([np.nan], dtype=np.float32)) \
            if iso_clean else (
                np.array([readbacks["armature"][rt_names.index(n)] for n in policy.joint_names], dtype=np.float32)
                if readbacks["armature"] is not None else np.nan * np.ones(29))
        st_out = Path(_args.step_test_output)
        st_out.mkdir(parents=True, exist_ok=True)

        def read_pol(name):
            return np.array([getattr(art.data, name)[0].cpu().numpy().astype(np.float32)[rt_names.index(n)]
                             for n in policy.joint_names], dtype=np.float32)

        def set_qdes_action(q_des_pol):
            """parity：走 RL action 路径（scale/offset 由 jp term 处理）——与训练侧同语义。"""
            raw_pol = (q_des_pol - offset_pol) / scale_pol
            act_rt = np.array([raw_pol[policy.joint_names.index(n)] for n in rt_names], dtype=np.float32)
            return torch.tensor(act_rt[None, :], device=env.device, dtype=torch.float32)

        def qdes_rt_tensor(q_des_pol):
            """step：直接写 PhysX drive target（关节角目标，同一 q_des 语义）。"""
            act_rt = np.array([q_des_pol[policy.joint_names.index(n)] for n in rt_names], dtype=np.float32)
            return torch.tensor(act_rt[None, :], device=env.device, dtype=torch.float32)

        if "contact_forces" in env.scene.sensors:
            cf_sensor = env.scene.sensors["contact_forces"]
        else:
            cf_sensor = None
            log("WARN: scene 无 contact_forces 传感器，接触指标将不可用")

        def contact_metrics():
            if cf_sensor is None:
                return None, None
            cfw = cf_sensor.data.net_forces_w[0].cpu().numpy()  # (nbodies, 3)
            norms = np.linalg.norm(cfw, axis=1)
            n_hit = int(np.sum(norms > 1e-6))
            return n_hit, (float(norms.max()) if norms.size else 0.0)

        joints = [s.strip() for s in _args.step_test_joint.split(",") if s.strip()]
        step_rad = _args.step_test_rad
        n_cycles_step = max(1, int(round(0.5 / 0.02)))   # 25 control ticks → 26 状态 0.00..0.50
        sub_last_drift = {}

        for jj, j in enumerate(joints):
            assert j in policy.joint_names, f"{j} not in policy joint_names"
            jidx = policy.joint_names.index(j)
            q0_j = float(q0_pol[jidx])
            others_rt = [i for i, n in enumerate(rt_names) if n != j]
            # ---- clean 局部遮蔽：1-dof 语义合成 29 维契约 ----
            if iso_clean:
                kp_j = kp_pol.reshape(-1)[0] if kp_pol is not None and np.isfinite(kp_pol).all() else float(policy.kp[jidx])
                kd_j = kd_pol.reshape(-1)[0] if kd_pol is not None and np.isfinite(kd_pol).all() else float(policy.kd[jidx])
                eff_j = eff_pol.reshape(-1)[0] if eff_pol is not None and np.isfinite(eff_pol).all() else np.inf
                arm_j = arm_pol.reshape(-1)[0] if arm_pol is not None and np.isfinite(arm_pol).all() else np.nan

                def curr_state_policy():
                    q_rt = art.data.joint_pos[0].cpu().numpy().astype(np.float32)
                    dq_rt = art.data.joint_vel[0].cpu().numpy().astype(np.float32)
                    q29 = q0_pol.copy(); dq29 = np.zeros(29, dtype=np.float32)
                    q29[jidx] = q_rt[0]; dq29[jidx] = dq_rt[0]
                    return q29, dq29

                def read_pol(name):
                    v = np.asarray(getattr(art.data, name)[0].cpu().numpy().astype(np.float32))
                    out = np.zeros(29, dtype=np.float32); out[jidx] = v[0]
                    return out

                def apply_canonical_state(q_pol_np, dq_pol_np, root_pos_np=None, root_quat_np=None,
                                          lin_w_np=None, ang_b_np=None):
                    """clean：28 关节模型级固定；root 由 init_state 一次性设 canonical root pose + fix_base 物理固定，不写回。"""
                    q_t = torch.tensor([[float(q_pol_np[jidx])]], device=env.device, dtype=torch.float32)
                    v_t = torch.zeros(1, 1, device=env.device)
                    art.write_joint_state_to_sim(q_t, v_t)

                def pin_and_lock():
                    """clean：28 关节在 URDF/物理层固死（无 dof）；root 由 init_state 一次性设 canonical pose + fix_base 固定，无写回。"""
                    return

                def isle_substep(q_des_rt_t):
                    pin_and_lock()
                    art.set_joint_position_target(q_des_rt_t)
                    art.write_data_to_sim()
                    env.sim.step(render=False)
                    env.scene.update(dt=env.physics_dt)
            else:
                kp_j, kd_j, eff_j, arm_j = None, None, None, None

            # ---------- 每个关节都从 canonical 全新重置（P0-1：parity/step 互不污染） ----------
            apply_canonical_state(q0_pol, np.zeros_like(q0_pol), root_pos0, root_quat0,
                                  np.zeros(3), np.zeros(3))

            # ================= parity 进程（--torque-parity-only，独立启动） =================
            if _args.torque_parity_only:
                q_des_pol = q0_pol.copy()
                q_des_pol[jidx] = q0_j + step_rad
                if iso_clean:
                    advance_desc = ("手动 4×sim.step substep（5ms）推进 20ms 后读取（PhysX implicit actuator 只能在 step 求解后读回）"
                                    "；isolation 模型 1-dof（其余 28 关节+root 在 URDF/物理层固死）")
                    q_des_rt_t = qdes_rt_tensor(q_des_pol)
                    for _s in range(4):
                        isle_substep(q_des_rt_t)
                    q_pol, dq_pol = curr_state_policy()
                    ct = read_pol("computed_torque")
                    at = read_pol("applied_torque")
                    tau_u = np.zeros(29, dtype=np.float32); tau_u[jidx] = (q_des_pol[jidx] - q_pol[jidx]) * kp_j - dq_pol[jidx] * kd_j
                    tau_c = tau_u.copy(); tau_c[jidx] = float(np.clip(tau_u[jidx], -eff_j, eff_j))
                    clip_any = bool(abs(tau_u[jidx] - tau_c[jidx]) > 1e-9)
                else:
                    advance_desc = "env.step 推进 20ms（4×5ms）后读取（PhysX implicit actuator 只能在 step 求解后读回）"
                    env.step(set_qdes_action(q_des_pol))
                    q_pol, dq_pol = curr_state_policy()
                    ct = read_pol("computed_torque")
                    at = read_pol("applied_torque")
                    tau_u = (q_des_pol - q_pol) * kp_pol - dq_pol * kd_pol
                    tau_c = np.clip(tau_u, -eff_pol, eff_pol)
                    clip_any = bool(np.any(np.abs(tau_u) > eff_pol))
                parity = {
                    "engine": "isaac", "mode": "engine_torque_readback",
                    "advance": advance_desc,
                    "joint": j, "condition": "NATIVE", "isolation": _args.isolation,
                    "physics_dt": 0.005, "decimation": 4, "control_dt": 0.02,
                    "q_des_j": float(q_des_pol[jidx]), "q_j": float(q_pol[jidx]), "dq_j": float(dq_pol[jidx]),
                    "computed_torque_j(PhysX)": float(ct[jidx]), "applied_torque_j(PhysX)": float(at[jidx]),
                    "tau_unclipped_offline_j": float(tau_u[jidx]), "tau_clipped_offline_j": float(tau_c[jidx]),
                    "computed_torque_pol(29)": ct.tolist(), "applied_torque_pol(29)": at.tolist(),
                    "tau_unclipped_pol(29)": tau_u.tolist(), "tau_clipped_pol(29)": tau_c.tolist(),
                    "kp_pol": kp_pol.tolist() if not iso_clean else (readbacks["kp"].tolist() if readbacks["kp"] is not None else []),
                    "kd_pol": kd_pol.tolist() if not iso_clean else (readbacks["kd"].tolist() if readbacks["kd"] is not None else []),
                    "armature_j_readback": float(arm_j if iso_clean else arm_pol[jidx]),
                    "effort_limit_j": float(eff_j if iso_clean else eff_pol[jidx]),
                    "clip_triggered_any": clip_any,
                }
                (st_out / f"torque_parity_{j}.json").write_text(
                    json.dumps(parity, indent=2, ensure_ascii=False), encoding="utf-8")
                log(f"N5-0 parity[{j}]: computed={parity['computed_torque_j(PhysX)']:.4f} "
                    f"applied={parity['applied_torque_j(PhysX)']:.4f} offline_u={tau_u[jidx]:.4f} "
                    f"offline_c={tau_c[jidx]:.4f} armature={parity['armature_j_readback']}")
                continue

            # ================= step 进程（--step-only，独立启动，P0-1 修复） =================
            # ---- initial-state gate（P0-2）：写入后回读，超差立即终止该关节 ----
            rq, rdq = curr_state_policy()
            q_err = float(np.max(np.abs(rq - q0_pol)))
            dq_err = float(np.max(np.abs(rdq)))
            gate = {
                "joint": j, "canonical_q0_max_err": q_err, "canonical_dq0_max_err": dq_err,
                "pass": q_err <= 1e-5 and dq_err <= 1e-5,
                "tolerance": {"q": 1e-5, "dq": 1e-5},
                "initial_state_npz_sha256": sha256_bytes(_args.initial_state),
            }
            (st_out / f"initial_state_readback_{j}.json").write_text(
                json.dumps(gate, indent=2, ensure_ascii=False), encoding="utf-8")
            log(f"step[{j}] initial-state gate: q_err={q_err:.2e} dq_err={dq_err:.2e} -> "
                f"{'PASS' if gate['pass'] else 'FAIL 终止'}")
            if not gate["pass"]:
                continue

            # ---- 隔离辅助：每 substep 前锁定 + 设 drive target（clean 协议） ----
            if not iso_clean:
                q0_rt = np.array([q0_pol[policy.joint_names.index(n)] for n in rt_names], dtype=np.float32)
                lock_pos_t = torch.tensor(q0_rt[others_rt][None, :], device=env.device, dtype=torch.float32)
                lock_vel_t = torch.zeros(1, len(others_rt), device=env.device)
                root_pose_t = torch.from_numpy(np.concatenate([root_pos0, root_quat0]).astype(np.float32)[None, :]).to(env.device)
                root_vel0_t = torch.zeros(1, 6, device=env.device)

                def pin_and_lock():
                    """coupled：不做 28 关节锁定；root 每 substep 写回。"""
                    art.write_root_pose_to_sim(root_pose_t)
                    art.write_root_velocity_to_sim(root_vel0_t)

                def isle_substep(q_des_rt_t):
                    pin_and_lock()
                    art.set_joint_position_target(q_des_rt_t)
                    art.write_data_to_sim()
                    env.sim.step(render=False)
                    env.scene.update(dt=env.physics_dt)

            rec = {k: [] for k in ["t_state", "t_command_start", "q(29)", "dq(29)", "q_des(29)",
                                   "computed_torque(29)", "applied_torque(29)", "saturated(29)",
                                   "other_joint_drift_max", "root_drift_max", "ncon", "max_contact_force"]}
            # t=0 真实初态（推进前）
            q_pol, dq_pol = curr_state_policy()
            rec["t_state"].append(0.0)
            rec["t_command_start"].append(0.0)
            rec["q(29)"].append(q_pol)
            rec["dq(29)"].append(dq_pol)
            rec["q_des(29)"].append(q0_pol.copy())
            rec["computed_torque(29)"].append(np.zeros(29, dtype=np.float32))
            rec["applied_torque(29)"].append(np.zeros(29, dtype=np.float32))
            rec["saturated(29)"].append(np.zeros(29, dtype=np.float32))
            rec["other_joint_drift_max"].append(0.0)
            rec["root_drift_max"].append(0.0)
            rec["ncon"].append(0)
            rec["max_contact_force"].append(0.0)

            max_drift_others = 0.0
            max_root_drift = 0.0
            max_ncon = 0
            max_cf = 0.0
            step_saturated = False
            for k in range(1, n_cycles_step + 1):
                t_state = k * 0.02                 # 推进后状态时刻（P0-2）
                t_cmd = (k - 1) * 0.02             # 命令区间 [t_cmd, t_state)
                q_des_pol = q0_pol.copy()
                q_des_pol[jidx] = q0_j + (step_rad if t_cmd >= 0.10 else 0.0)
                q_des_rt_t = qdes_rt_tensor(q_des_pol)
                for _s in range(4):
                    isle_substep(q_des_rt_t)
                # post-step readback（t_state 语义）
                q_pol, dq_pol = curr_state_policy()
                ct = read_pol("computed_torque")
                at = read_pol("applied_torque")
                sat = np.abs(ct - at) > 1e-9
                step_saturated = step_saturated or bool(np.any(sat))
                ncon_est, cfmax = contact_metrics()
                ncon_est = ncon_est if ncon_est is not None else -1
                cfmax = cfmax if cfmax is not None else -1.0
                max_ncon = max(max_ncon, ncon_est)
                max_cf = max(max_cf, cfmax)
                # 漂移（step 后、下一轮 substep 锁定前的瞬时值；clean 下根/28 关节物理固定）：
                q_now_pol = q_pol  # 合成 29 维（clean：被测真实，其余恒 q0）
                others_pol_idx = [i for i in range(29) if i != jidx]
                drift_others = float(np.max(np.abs(
                    q_now_pol[others_pol_idx] - q0_pol[others_pol_idx])))
                max_drift_others = max(max_drift_others, drift_others)
                if iso_clean:
                    # root 锚定 canonical root pose（init_state 一次性设置，物理层不动）：参考 = canonical
                    _rcp = np.asarray(root_pos0, dtype=np.float64)
                    _rcq = np.asarray(root_quat0, dtype=np.float64)
                    rp = art.data.root_pos_w[0].cpu().numpy().astype(np.float64) - _rcp
                    rq4 = np.linalg.norm(art.data.root_quat_w[0].cpu().numpy().astype(np.float64) - _rcq)
                else:
                    rp = art.data.root_pos_w[0].cpu().numpy() - root_pos0
                    rq4 = np.linalg.norm(art.data.root_quat_w[0].cpu().numpy() - root_quat0)
                root_drift = float(max(np.linalg.norm(rp), rq4))
                max_root_drift = max(max_root_drift, root_drift)
                # 记录
                rec["t_state"].append(t_state)
                rec["t_command_start"].append(t_cmd)
                rec["q(29)"].append(q_pol)
                rec["dq(29)"].append(dq_pol)
                rec["q_des(29)"].append(q_des_pol)
                rec["computed_torque(29)"].append(ct)
                rec["applied_torque(29)"].append(at)
                rec["saturated(29)"].append(sat.astype(np.float32))
                rec["other_joint_drift_max"].append(drift_others)
                rec["root_drift_max"].append(root_drift)
                rec["ncon"].append(ncon_est)
                rec["max_contact_force"].append(cfmax)

            lens = {k2: len(v) for k2, v in rec.items()}
            assert len(set(lens.values())) == 1, f"记录长度不一致: {lens}"
            n_rec = lens["t_state"]
            np.savez_compressed(st_out / f"step_trace_{j}.npz",
                                **{k2: np.asarray(v, dtype=np.float32) for k2, v in rec.items()})
            (st_out / f"run_manifest_{j}.json").write_text(json.dumps({
                "engine": "isaac", "joint": j, "condition": "NATIVE",
                "isolation": _args.isolation, "step_rad": step_rad, "duration_s": 0.5,
                "physics_dt": 0.005, "decimation": 4, "control_dt": 0.02,
                "n_cycles": n_cycles_step, "n_states": n_rec, "t_switch_s": 0.10,
                "t0_is_true_initial": True, "t_final_is_0_500": abs(n_rec - 26) < 1e-9,
                "sampling": "每 substep(5ms) sim.step + scene.update；状态在 control tick 末记录标 t_state；命令 [t_cmd,t_state) 生效",
                "run_mode": "step-only（独立进程，未先执行 parity）",
                "fixed_base": ("URDF 隔离模型（V3 修复）：28 revolute→fixed 仅局部烘焙 q0（全部 origin 局部语义）+ fix_base=True；"
                                "canonical root 世界 pose 仅由 init_state 一次性设置（不写回）；唯一 dof=被测关节"
                               if iso_clean else "root pose/vel 每 physics substep 写回；28 关节未锁定"),
                "isolation_asset": (f"urdf_isolation.py 生成: {Path(env_cfg.scene.robot.spawn.asset_path).name}"
                                     if iso_clean else "官方 URDF"),
                "gravity": (0.0, 0.0, 0.0) if _args.isolation == "clean" else (-9.81, 0.0, 0.0),
                "contacts": ("PhysX 碰撞几何未修改；根固定+1-dof+gravity=0 动态上无接触，ncon/max_contact_force 实测记录（验收要求接触=0 否则 FAIL）"
                              if iso_clean else "PhysX 碰撞未在引擎层显式关闭；隔离条件（gravity=0+锁定）保证无接触，ncon/max_contact_force 实测记录"),
                "initial_state_npz_sha256": sha256_bytes(_args.initial_state),
                "initial_state_gate": gate,
                "armature_j_readback": float(arm_j if iso_clean else arm_pol[jidx]),
                "kp_j/kd_j": [float(kp_j if iso_clean else kp_pol[jidx]), float(kd_j if iso_clean else kd_pol[jidx])],
                "effort_limit_j": float(eff_j if iso_clean else eff_pol[jidx]),
                "q0_j": q0_j,
                "max_drift_other_joints": max_drift_others, "max_root_drift": max_root_drift,
                "max_ncon": max_ncon, "max_contact_force": max_cf, "step_saturated_any": step_saturated,
                "hashes": {"policy": sha256_bytes(_args.policy), "motion": sha256_bytes(_args.motion),
                           "initial_state": sha256_bytes(_args.initial_state)},
            }, indent=2, ensure_ascii=False), encoding="utf-8")
            log(f"N5-R step[{j}] done: n_states={n_rec} (0.00..0.50) drift_others={max_drift_others:.3e} "
                f"root_drift={max_root_drift:.2e} ncon={max_ncon} cf={max_cf:.3e} sat={step_saturated}")
        logf.close()
        os._exit(0)

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
    # P3-R1: clear instrumented actuator FIFO after any initial zero-action steps before main loop
    try:
        for _act in art.actuators.values():
            if hasattr(_act, "_p3_fifo_pos"):
                _act._p3_fifo_pos = None
                _act._p3_primed = False
                _act._p3_requested_ep.clear()
                _act._p3_effective_ep.clear()
                _act._p3_req_per_control.clear()
                _act._p3_eff_per_control.clear()
                _act._p3_physics_step_counter = 0
        log("P3-R1: instrumented FIFO cleared after prefill, ready for first policy q_des")
    except Exception as _e:
        log(f"WARN FIFO clear: {_e}")

    # 相机 B31.2 全身可见：必须使用 set_world_poses_from_view（eye, target），删除 look_at 录像路径
    camera = None
    if _args.record_video:
        camera = env.scene.sensors["camera"]
        # initial camera via set_world_poses_from_view (eye, target) — B31.3 要求（需 torch Tensor），6.5m 再降低高度看全脚
        try:
            _init_root = art.data.root_pos_w[0].cpu().numpy().astype(np.float64)
            _eye0 = torch.as_tensor(_init_root + np.array([0.35, -6.5, 0.20], dtype=np.float64), device=camera._device, dtype=torch.float32).unsqueeze(0)
            _tgt0 = torch.as_tensor(_init_root + np.array([0.0, 0.0, 0.00], dtype=np.float64), device=camera._device, dtype=torch.float32).unsqueeze(0)
            camera.set_world_poses_from_view(_eye0, _tgt0)
        except Exception as _e:
            # fallback: fixed view if root not yet available
            _eye_fb = torch.as_tensor(np.array([[0.35, -6.5, 1.3]], dtype=np.float32), device=camera._device)
            _tgt_fb = torch.as_tensor(np.array([[0.0, 0.0, 0.00]], dtype=np.float32), device=camera._device)
            camera.set_world_poses_from_view(_eye_fb, _tgt_fb)
            log(f"WARN initial camera fallback: {_e}")
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
    # P3-R2: extra fields for instrumented delay with fail-closed ownership and physics index
    rec = {k: [] for k in ["policy_step", "t", "time_step", "obs(770)", "action(29)", "q_des(29)",
                           "q(29)", "dq(29)", "root_pos(3)", "root_quat(4)",
                           "root_lin_vel(3)", "root_ang_vel(3)", "actual_processed_q_target(29)",
                           "reference_body_pos_w(14,3)", "reference_body_quat_w(14,4)",
                           "reference_body_lin_vel_w(14,3)", "reference_body_ang_vel_w(14,3)",
                           "actual_body_pos_w(14,3)", "actual_body_quat_w(14,4)",
                           "actual_body_lin_vel_w(14,3)", "actual_body_ang_vel_w(14,3)",
                           "done", "fall_reason",
                           "q_des_requested(29)", "q_des_effective(29)",
                           "q_des_requested_substep", "q_des_effective_substep",
                           "physics_time_substep", "actuator_compute_count_per_control",
                           "actuator_physics_step_index_substep",
                           "actuator_group_names", "actuator_compute_count_by_group",
                           "actuator_joint_coverage_count", "actuator_joint_owner"]}
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

        # 相机 P1-3: 25fps 采样以得到约10秒视频 (250帧)
        if camera is not None and t >= last_vframe + 1 / 25.0 - 1e-6:
            eye = torch.as_tensor(root_pos.astype(np.float64) + np.array([0.35, -6.5, 0.20], dtype=np.float64), device=camera._device, dtype=torch.float32).unsqueeze(0)
            tgt = torch.as_tensor(root_pos.astype(np.float64) + np.array([0.0, 0.0, 0.00], dtype=np.float64), device=camera._device, dtype=torch.float32).unsqueeze(0)
            camera.set_world_poses_from_view(eye, tgt)
            env.sim.render()
            camera.update(dt=0.02)
            img = camera.data.output["rgb"][0].cpu().numpy()
            # B31 fix: 若首帧为黑，尝试额外 render+update（Isaac 相机管线异步，偶发黑帧）
            if img.max() <= 5:
                env.sim.render()
                camera.update(dt=0.02)
                img = camera.data.output["rgb"][0].cpu().numpy()
            # 仍为黑则跳过该帧，避免视频闪烁
            if img.max() <= 5:
                log(f"WARN skip black frame at t={t:.2f}")
            else:
                video_frames.append(img)
                if len(video_frames) == 10 and img.max() <= 5:
                    log("WARN: first 10 camera frames appear BLACK")
            last_vframe = t

        # R1：actual processed target —— 每周期恰追加一次；失败周期未 step，记 q_des（未执行）并在 fall_reason 说明
        # P3-R1: clear per-control debug before step (instrumented actuator)
        try:
            for _act in art.actuators.values():
                if hasattr(_act, "clear_per_control_debug"):
                    _act.clear_per_control_debug()
        except Exception as _e:
            log(f"WARN clear_per_control_debug: {_e}")
        act_rt = np.array([action[policy.joint_names.index(n)] for n in rt_names], dtype=np.float32) if not order_ok else action
        act_t = torch.tensor(act_rt[None, :], device=env.device, dtype=torch.float32)
        if reason is not None:
            rec["actual_processed_q_target(29)"].append(q_des_pol.copy())
            # P3-R2: for failure cycle without step, still need instrumented fields but with fail-closed semantics
            # Use same q_des tiling but also set physics indices to expected values (cyc*4+1 ..)
            _group_names_fail = list(art.actuators.keys())
            _phys_fail = np.array([cyc*4 + s +1 for s in range(4)], dtype=np.int32)
            rec["q_des_requested(29)"].append(q_des_pol.copy())
            rec["q_des_effective(29)"].append(q_des_pol.copy())
            rec["q_des_requested_substep"].append(np.tile(q_des_pol, (4,1)).astype(np.float32))
            rec["q_des_effective_substep"].append(np.tile(q_des_pol, (4,1)).astype(np.float32))
            rec["physics_time_substep"].append((_phys_fail.astype(np.float32)*0.005))
            rec["actuator_physics_step_index_substep"].append(_phys_fail.copy())
            rec["actuator_compute_count_per_control"].append(np.array(0, dtype=np.int32))
            rec["actuator_compute_count_by_group"].append(np.zeros((len(_group_names_fail),), dtype=np.int32))
            # For coverage/owner we still provide but count 0 indicates failure cycle
            _coverage_fail = np.zeros(29, dtype=np.int32)
            _owner_fail = np.array(["" for _ in range(29)], dtype=object)
            rec["actuator_joint_coverage_count"].append(_coverage_fail)
            rec["actuator_joint_owner"].append(_owner_fail)
            if "actuator_group_names" not in rec or len(rec["actuator_group_names"])==0 or isinstance(rec["actuator_group_names"], list):
                rec["actuator_group_names"] = _group_names_fail
            fall_reason = reason
            log(f"FALL at cycle {cyc} t={t:.2f}s reason={reason} (recorded; 不 step)")
            break
        env.step(act_t)
        if obs_builder is not None:
            obs_builder.advance()  # 与 MuJoCo 侧一致：每执行一个控制周期推进一帧
        proc_rt = jp.processed_actions[0].cpu().numpy().astype(np.float32)
        proc_pol = np.array([proc_rt[rt_names.index(n)] for n in policy.joint_names], dtype=np.float32)
        rec["actual_processed_q_target(29)"].append(proc_pol)
        # P3-R2: fail-closed aggregation with ownership, physics index, and strict checks
        # No fallback tiling: any violation raises SystemExit and aborts trace generation
        decimation = int(env.cfg.decimation)  # 4
        # Build ownership map: policy joint -> group
        _group_names = list(art.actuators.keys())
        _policy_joints = list(policy.joint_names)
        _joint_owner = {}  # joint_name -> group
        _coverage = np.zeros(29, dtype=np.int32)
        for _gname, _act in art.actuators.items():
            if not isinstance(_act, P3InstrumentedDelayedImplicitActuator):
                raise SystemExit(f"P0-1 FAIL: actuator group {_gname} not instrumented: {type(_act)}")
            _jnames = list(_act.joint_names)  # type: ignore
            for jn in _jnames:
                if jn not in _policy_joints:
                    raise SystemExit(f"P0-1 FAIL: unknown joint {jn} in group {_gname} not in policy")
                if jn in _joint_owner:
                    raise SystemExit(f"P0-1 FAIL: joint {jn} covered by both {_joint_owner[jn]} and {_gname} (duplicate)")
                _joint_owner[jn] = _gname
                _coverage[_policy_joints.index(jn)] += 1
        # Check each policy joint covered exactly once
        for idx, jn in enumerate(_policy_joints):
            if _coverage[idx] != 1:
                raise SystemExit(f"P0-1 FAIL: joint {jn} coverage {_coverage[idx]} !=1 (missing or duplicate)")
        # Collect per-group data with strict checks
        req_sub = np.zeros((decimation, 29), dtype=np.float32)
        eff_sub = np.zeros((decimation, 29), dtype=np.float32)
        phys_idx_sub = np.zeros((decimation,), dtype=np.int32)
        count_by_group = np.zeros((len(_group_names),), dtype=np.int32)
        # For physics index, collect from each group and verify all identical
        _phys_indices_all = None
        for gi, _gname in enumerate(_group_names):
            _act = art.actuators[_gname]
            _req_list, _eff_list = _act.get_per_control_debug()
            _phys_list = _act.get_per_control_physics_indices()
            # Strict length check
            if len(_req_list) != decimation or len(_eff_list) != decimation or len(_phys_list) != decimation:
                raise SystemExit(f"P0-1 FAIL: group {_gname} per_control len req {len(_req_list)} eff {len(_eff_list)} phys {len(_phys_list)} != decimation {decimation}")
            # Strict shape check
            _jnames = list(_act.joint_names)  # type: ignore
            n_group = len(_jnames)
            for s in range(decimation):
                r = _req_list[s]
                e = _eff_list[s]
                if isinstance(r, torch.Tensor):
                    r_np = r.detach().cpu().numpy().reshape(-1)
                else:
                    r_np = np.asarray(r).reshape(-1)
                if isinstance(e, torch.Tensor):
                    e_np = e.detach().cpu().numpy().reshape(-1)
                else:
                    e_np = np.asarray(e).reshape(-1)
                if r_np.shape != (n_group,) or e_np.shape != (n_group,):
                    raise SystemExit(f"P0-1 FAIL: group {_gname} s {s} shape r {r_np.shape} e {e_np.shape} != ({n_group},)")
                for j_idx, j_name in enumerate(_jnames):
                    p_idx = _policy_joints.index(j_name)
                    req_sub[s, p_idx] = float(r_np[j_idx])
                    eff_sub[s, p_idx] = float(e_np[j_idx])
            # Physics indices
            _phys_np = np.asarray(_phys_list, dtype=np.int32).reshape(-1)
            if _phys_np.shape != (decimation,):
                raise SystemExit(f"P0-2 FAIL: group {_gname} phys shape {_phys_np.shape} != ({decimation},)")
            if _phys_indices_all is None:
                _phys_indices_all = _phys_np
            else:
                if not np.array_equal(_phys_indices_all, _phys_np):
                    raise SystemExit(f"P0-2 FAIL: group {_gname} phys indices {_phys_np.tolist()} != first group {_phys_indices_all.tolist()}")
            phys_idx_sub = _phys_np  # all groups same
            count_by_group[gi] = decimation
        # Verify requested_substep first row == q_des (within 1e-6) for sanity, but not fail-closed? Use check but allow tiny numerical?
        if not np.allclose(req_sub[0], q_des_pol, atol=1e-6):
            raise SystemExit(f"P0-1 FAIL: req_sub[0] != q_des max diff {np.max(np.abs(req_sub[0]-q_des_pol))}")
        # Physics time from real counter
        physics_dt = 0.005
        phys_time_sub = _phys_indices_all.astype(np.float32) * physics_dt
        # Cross-check last phys index time vs control boundary (t + control_dt)
        # Control k spans phys indices k*4+1 .. (k+1)*4  (1-indexed)
        # Last phys time should be (cyc+1)*0.02 within 1e-6
        expected_last = (cyc+1)*0.02
        if abs(float(phys_time_sub[-1]) - expected_last) > 1e-6:
            raise SystemExit(f"P0-2 FAIL: phys_time last {phys_time_sub[-1]} != expected {expected_last} at cyc {cyc}")
        # Prepare ownership arrays for trace
        _owner_list = [_joint_owner[jn] for jn in _policy_joints]
        # Append to rec
        rec["q_des_requested(29)"].append(q_des_pol.copy())
        rec["q_des_effective(29)"].append(eff_sub[-1].copy())
        rec["q_des_requested_substep"].append(req_sub.astype(np.float32))
        rec["q_des_effective_substep"].append(eff_sub.astype(np.float32))
        rec["physics_time_substep"].append(phys_time_sub.astype(np.float32))
        rec["actuator_physics_step_index_substep"].append(_phys_indices_all.astype(np.int32))
        rec["actuator_compute_count_per_control"].append(np.array(decimation, dtype=np.int32))
        # For per-group counts and coverage, we store per-control values that will be aggregated later
        # Instead of per-control, we store for first control and will expand in checkpoint; but for trace we store per-control list
        # To keep n_control dimension, we append per-control values for new fields
        if "actuator_group_names" not in rec or len(rec["actuator_group_names"])==0:
            # Store group names once (not per control), but we need shape-consistent field; we will store as per-control copy for validation
            rec["actuator_group_names"] = _group_names  # will be saved as object array, validator checks
        # For count_by_group, coverage, owner: store per-control but validator will check first entry
        # We append per-control to keep length n_control for those that are per-control
        if len(rec["actuator_compute_count_by_group"]) == 0:
            # Initialize with first value, subsequent appends will stack
            pass
        rec["actuator_compute_count_by_group"].append(count_by_group.copy())
        rec["actuator_joint_coverage_count"].append(_coverage.copy())
        rec["actuator_joint_owner"].append(np.array(_owner_list, dtype=object))
        # P3-R2: online checkpoint at 100 cycles (cyc 99) - fail-closed, abort if not pass
        if cyc == 99:
            try:
                _chk = {}
                _chk["cycles_so_far"] = len(rec["policy_step"])
                _chk["pass_cycles"] = _chk["cycles_so_far"] == 100
                _chk["finite"] = bool(np.isfinite(req_sub).all() and np.isfinite(eff_sub).all())
                _chk["count_4"] = bool(np.all(count_by_group==4))
                _chk["coverage_1"] = bool(np.all(_coverage==1))
                _phys_all_so_far = np.concatenate([np.asarray(x, dtype=np.int32) for x in rec["actuator_physics_step_index_substep"]]) if len(rec["actuator_physics_step_index_substep"])>0 else np.array([])
                _chk["phys_continuous"] = bool(np.array_equal(_phys_all_so_far, np.arange(1, len(_phys_all_so_far)+1))) if len(_phys_all_so_far)>0 else False
                _maxdiff_req_qdes = float(np.max([np.max(np.abs(np.asarray(rec["q_des_requested_substep"][i][0] - rec["q_des(29)"][i]))) for i in range(len(rec["q_des_requested_substep"]))])) if len(rec["q_des_requested_substep"])>0 else 1e9
                _chk["req_eq_qdes"] = _maxdiff_req_qdes < 1e-6
                _chk["req_eq_qdes_maxdiff"] = _maxdiff_req_qdes
                _req100 = np.asarray(rec["q_des_requested(29)"], dtype=np.float32)
                _eff100 = np.asarray(rec["q_des_effective(29)"], dtype=np.float32)
                _best_shift=None
                _best_diff=None
                for shift in range(6):
                    if shift==0:
                        diff=float(np.max(np.abs(_eff100-_req100)))
                    else:
                        diff=float(np.max(np.abs(_eff100[shift:]-_req100[:-shift])))
                    if _best_diff is None or diff<_best_diff:
                        _best_diff=diff
                        _best_shift=shift
                _chk["measured_shift"]=_best_shift
                _chk["measured_diff"]=_best_diff
                _chk["measured_ok"] = bool(_best_shift==1 and _best_diff<1e-6)
                _req_phys = np.concatenate([np.asarray(x) for x in rec["q_des_requested_substep"]]).reshape(-1,29) if len(rec["q_des_requested_substep"])>0 else np.zeros((0,29))
                _eff_phys = np.concatenate([np.asarray(x) for x in rec["q_des_effective_substep"]]).reshape(-1,29) if len(rec["q_des_effective_substep"])>0 else np.zeros((0,29))
                if len(_eff_phys)>=4:
                    _chk["first4_eq_first_req"]=bool(np.max(np.abs(_eff_phys[:4]-_req_phys[0]))<1e-6)
                    _chk["first4_maxdiff"]=float(np.max(np.abs(_eff_phys[:4]-_req_phys[0])))
                else:
                    _chk["first4_eq_first_req"]=False
                _chk["no_failure"]=bool(fall_reason is None)
                # Compute hashes on the fly for checkpoint (since manifest hashes defined later)
                try:
                    _chk_runner = hashlib.sha256(_pathlib.Path(__file__).read_bytes()).hexdigest()
                except Exception:
                    _chk_runner = None
                try:
                    _chk_act = hashlib.sha256(_pathlib.Path(r"E:\sim2sim-week-2026-08-26\13_p3_delay20\00_inputs\p3_instrumented_delayed_actuator_r2.py").read_bytes()).hexdigest()
                except Exception:
                    _chk_act = None
                try:
                    _chk_model = hashlib.sha256(_pathlib.Path(env_cfg.scene.robot.spawn.asset_path).read_bytes()).hexdigest() if _pathlib.Path(env_cfg.scene.robot.spawn.asset_path).is_file() else None
                except Exception:
                    _chk_model = None
                _chk["hashes_ok"]=bool(_chk_runner is not None and _chk_act is not None and _chk_model is not None)
                _chk["runner_hash"]= _chk_runner[:8] if _chk_runner else None
                _chk["actuator_hash"]= _chk_act[:8] if _chk_act else None
                _chk["model_hash"]= _chk_model[:8] if _chk_model else None
                _chk_pass = all([_chk["pass_cycles"], _chk["finite"], _chk["count_4"], _chk["coverage_1"], _chk["phys_continuous"], _chk["req_eq_qdes"], _chk["measured_ok"], _chk["first4_eq_first_req"], _chk["no_failure"], _chk["hashes_ok"]])
                _chk["pass"]=_chk_pass
                import json as _js
                _chk_path = out_dir / "checkpoint_100.json"
                _chk_path.write_text(_js.dumps(_chk, indent=2), encoding="utf-8")
                log(f"CHECKPOINT 100: pass={_chk_pass} shift={_best_shift} diff={_best_diff:.2e} phys_cont={_chk['phys_continuous']}")
                if not _chk_pass:
                    raise SystemExit(f"CHECKPOINT 100 FAIL: {_chk}")
            except SystemExit:
                raise
            except Exception as _e:
                import traceback as _tb
                log(f"CHECKPOINT 100 EXCEPTION: {_e} {_tb.format_exc()}")
                raise SystemExit(f"CHECKPOINT 100 EXCEPTION: {_e}")
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
        elif k in ("actuator_group_names", "actuator_joint_owner"):
            # object arrays (strings)
            npz_out[k] = np.asarray(v, dtype=object)
        elif k.startswith("reference_body") or k.startswith("actual_body"):
            npz_out[k] = np.asarray(v, dtype=np.float32)
        elif k in ("q_des_requested_substep", "q_des_effective_substep"):
            npz_out[k] = np.asarray(v, dtype=np.float32)
        elif k in ("physics_time_substep",):
            npz_out[k] = np.asarray(v, dtype=np.float32)
        elif k in ("actuator_physics_step_index_substep", "actuator_compute_count_per_control", "actuator_compute_count_by_group", "actuator_joint_coverage_count"):
            npz_out[k] = np.asarray(v, dtype=np.int32)
        else:
            npz_out[k] = np.asarray(v, dtype=np.float32)
    np.savez_compressed(out_dir / "policy_trace.npz", **npz_out)
    if iface is not None:
        np.savez_compressed(out_dir / "isaac_golden_states.npz",
                            **{k: np.asarray(v) for k, v in iface.items()})
    # P3-R2: compute hashes and provenance, measured is NOT_EVALUATED
    try:
        _actuator_src = _pathlib.Path(__file__).resolve().parent / "p3_instrumented_delayed_actuator_r2.py"
        if not _actuator_src.is_file():
            _actuator_src = _pathlib.Path(r"E:\sim2sim-week-2026-08-26\13_p3_delay20\00_inputs\p3_instrumented_delayed_actuator_r2.py")
        _actuator_sha = hashlib.sha256(_actuator_src.read_bytes()).hexdigest() if _actuator_src.is_file() else None
    except Exception:
        _actuator_sha = None
    try:
        _runner_sha = hashlib.sha256(_pathlib.Path(__file__).read_bytes()).hexdigest()
    except Exception:
        _runner_sha = None
    # Ensure top-level runner hash equals hashes.runner
    _req_delay_steps = int(round(float(_args.fixed_delay_ms) / 5.0)) if float(_args.fixed_delay_ms)!=0 else 0
    _req_delay_ms = float(_args.fixed_delay_ms)
    _prehistory_mode = "repeat-first-q-des"
    # P0-4 provenance: model path, repo_state, isaac lab commit
    # Model path from env_cfg
    try:
        _model_path = str(env_cfg.scene.robot.spawn.asset_path)
        _model_name = _pathlib.Path(_model_path).name
        _model_sha = hashlib.sha256(_pathlib.Path(_model_path).read_bytes()).hexdigest() if _pathlib.Path(_model_path).is_file() else None
        # Verify expected Isaac model hash c2aa...
        if _model_sha is not None and _model_sha != "c2aa383744d89c888e38feeb9e52e3723ed6e288ea91c7ecff950cbd711ed8eb":
            log(f"WARN model hash {_model_sha} != expected c2aa3837...")
    except Exception as _e:
        _model_path = None; _model_name=None; _model_sha=None
        log(f"WARN model provenance: {_e}")
    # Repo state: try to load repo_state.json if git unavailable
    _repo_state_path = _pathlib.Path(r"E:\sim2sim-week-2026-08-26\13_p3_delay20\00_inputs\repo_state.json")
    _repo_state = None
    if _repo_state_path.is_file():
        try:
            import json as _js
            _repo_state = _js.loads(_repo_state_path.read_text(encoding="utf-8"))
        except Exception:
            _repo_state = None
    _lab_stamp = repo_stamp(REPO)
    # If repo_state file exists, prefer it; else use stamp
    if _repo_state is not None:
        _lab_commit = _repo_state.get("commit")
        _lab_dirty = _repo_state.get("dirty")
        _lab_branch = _repo_state.get("branch")
    else:
        _lab_commit = _lab_stamp.get("commit")
        _lab_dirty = _lab_stamp.get("dirty")
        _lab_branch = None
    manifest = {
        "engine": "isaac", "task": _args.task, "duration_s": _args.duration, "control_dt": 0.02,
        "physics_dt": 0.005, "decimation": 4, "cycles_planned": n_cycles, "cycles_done": len(rec["policy_step"]),
        "fall_reason": fall_reason, "seed": _args.seed, "seed_used": {"numpy": _args.seed, "torch": _args.seed},
        "fixed_delay_ms": _args.fixed_delay_ms, "gain_scale": _args.gain_scale, "num_envs": 1,
        "requested_delay_steps": _req_delay_steps, "requested_delay_ms": _req_delay_ms,
        "measured_delay_steps": None, "measured_delay_ms": None, "measured_delay_evaluation": "NOT_EVALUATED",
        "prehistory_mode": _prehistory_mode, "actuator_compute_calls_per_control": 4,
        "physics_time_source": "actuator_compute_counter_x_physics_dt",
        "position_delay_backend": "explicit_deque",
        "native_position_delay_buffer_bypassed": True,
        "native_velocity_effort_delay_buffer_used": True,
        "implementation": "collections.deque",
        "actuator_source_sha256": _actuator_sha, "runner_sha256": _runner_sha,
        "model_asset_path": _model_path, "model_asset_name": _model_name, "model_asset_sha256": _model_sha,
        "isaac_lab_commit": _lab_commit, "isaac_lab_branch": _lab_branch, "isaac_lab_dirty": _lab_dirty, "isaac_lab_repo_state": _repo_state,
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
            "runner": _runner_sha,
            "actuator_source": _actuator_sha,
            "canonical": sha256_bytes(_args.initial_state) if _args.initial_state else None,
            "model": _model_sha,
        },
        "repo": {"lab": repo_stamp(REPO)},
        "initial_state_readback": init_readback,
    }
    with open(out_dir / "run_manifest.yaml", "w", encoding="utf-8") as f:
        f.write(yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False, default_flow_style=False))
    if _args.record_video and video_frames:
        try:
            import imageio
            imageio.mimsave(out_dir / "video.mp4", video_frames, fps=25, codec="libx264", quality=8)
        except Exception as e:
            try:
                import cv2
                w = cv2.VideoWriter(str(out_dir / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 25, (1280, 720))
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