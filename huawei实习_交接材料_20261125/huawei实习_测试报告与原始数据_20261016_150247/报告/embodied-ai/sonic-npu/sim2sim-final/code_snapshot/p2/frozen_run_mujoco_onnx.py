#!/usr/bin/env python3
"""
run_mujoco_onnx.py — Humanoid Lab dance_9 直接 ONNX runner（MuJoCo 侧，本周替代链）

R1 修订（2026-08-27，依据 2026-08-27 修订计划 §R1）：
  - CLI 增加 --mode as_shipped|aligned、--start-frame、--initial-state、--history-init
  - 正式物理对比固定：--mode aligned --history-init repeat-first（第 0 帧预填 5 槽，与 Isaac reset 一致）
  - 严格的按名映射（policy/sim2sim/mjcf joint order → mjcf actuator order 独立映射，
    缺失关节断言报错，不再静默填 0）
  - 初始状态从 canonical_initial_state.npz 写入，mj_forward 后回读、保存 readback 哈希
  - applied(29) 保存完整 d.actuator_force（不再 [6:] 截成 23 维），另存 qfrc_actuator[6:]
  - initial_root_z / final_root_z 分开记录
  - 延迟 FIFO 按物理步推进（20ms=10 步），非零延迟在 unit test 通过前直接拒绝执行
  - --delay-self-test 离线验证 FIFO 精确滞后 N 个物理步
  - 每 run 写 run_manifest.yaml（含模式、seed、五类哈希、commit、回读初态、控制参数）

历史保留语义（as_shipped）：观测 = deploy mimic_rl_interface field-major 5 帧 ring、
q_des = action*scale+default、PD 每物理步、EGL 录像、失败谓词镜像 Isaac TerminationsCfg。

用法（WSL，.venv_lab_sim）：
  python scripts/experiments/run_mujoco_onnx.py \
      --config deploy/src/era_rl_controller/configs/mimic_dance_9.yaml \
      --mode aligned --start-frame 0 --history-init repeat-first \
      --initial-state /mnt/e/sim2sim-week-2026-08-26/01_lab_interface_gate/aligned_v1/canonical_initial_state.npz \
      --duration 5 --seed 42 --output <run_dir> [--record-video]
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import yaml

_cand = Path(__file__).resolve().parents[2]
if (_cand / "deploy").is_dir():
    REPO = _cand
elif Path("/mnt/e/humanoid-lab/deploy").is_dir():
    REPO = Path("/mnt/e/humanoid-lab")
elif Path("E:/humanoid-lab/deploy").is_dir():
    REPO = Path("E:/humanoid-lab")
else:
    REPO = _cand
DEPLOY = REPO / "deploy"


# ================================================================ 四元数工具（wxyz，与 deploy utils/math.py 一致）
def quat_inv(q):
    q = np.asarray(q, dtype=np.float64)
    return np.concatenate((q[..., 0:1], -q[..., 1:]), axis=-1) / np.clip(np.sum(q**2, axis=-1, keepdims=True), 1e-9, None)


def quat_mul(q1, q2):
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    flat = q1.ndim == 1 and q2.ndim == 1
    a = q1.reshape(-1, 4)
    b = q2.reshape(-1, 4)
    w1, x1, y1, z1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    r = np.stack([w, x, y, z], axis=-1)
    return r[0] if flat else r


def quat_apply(q, v):
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    t = 2.0 * np.cross(q[1:], v)
    return v + q[0] * t + np.cross(q[1:], t)


def quat_apply_inverse(q, v):
    return quat_apply(quat_inv(q), v)


def matrix_from_quat(q):
    q = np.asarray(q, dtype=np.float64)
    r, i, j, k = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    o = np.empty(q.shape[:-1] + (3, 3))
    o[..., 0, 0] = 1 - 2 * (j * j + k * k)
    o[..., 0, 1] = 2 * (i * j - k * r)
    o[..., 0, 2] = 2 * (i * k + j * r)
    o[..., 1, 0] = 2 * (i * j + k * r)
    o[..., 1, 1] = 1 - 2 * (i * i + k * k)
    o[..., 1, 2] = 2 * (j * k - i * r)
    o[..., 2, 0] = 2 * (i * k - j * r)
    o[..., 2, 1] = 2 * (j * k + i * r)
    o[..., 2, 2] = 1 - 2 * (i * i + j * j)
    return o


def yaw_quaternion(q):
    q = np.asarray(q, dtype=np.float32)
    flat = q.ndim == 1
    qq = q.reshape(-1, 4)
    w, x, y, z = qq[..., 0], qq[..., 1], qq[..., 2], qq[..., 3]
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half = 0.5 * yaw
    ret = np.zeros_like(qq)
    ret[..., 0] = np.cos(half)
    ret[..., 3] = np.sin(half)
    ret = ret / np.clip(np.linalg.norm(ret, axis=-1, keepdims=True), 1e-9, None)
    return ret[0] if flat else ret


# ================================================================ 按名映射（严格，缺失即报错；禁止静默填 0）
def map_by_name(arr, from_order, to_order, context=""):
    arr = np.asarray(arr)
    miss_from = [n for n in to_order if n not in from_order]
    if miss_from:
        raise ValueError(f"{context}: to_order 中 {len(miss_from)} 个名字不在 from_order: {miss_from[:6]}")
    idx = [from_order.index(n) for n in to_order]
    return np.asarray(arr[idx], dtype=arr.dtype)


# ================================================================ G0b adapter 注入状态（两侧同名同序）
def quat_from_euler_xyz(roll, pitch, yaw):
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dtype=np.float64)


ADAPTER_STATES = {
    "id0_identity": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
    "id1_yaw_p90": np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)], dtype=np.float64),
    "id2_yaw_m90": np.array([np.cos(np.pi / 4), 0.0, 0.0, -np.sin(np.pi / 4)], dtype=np.float64),
    "id3_roll_pitch": quat_from_euler_xyz(0.3, -0.2, 0.0),
}
ADAPTER_BASE_LIN = np.array([0.3, -0.2, 0.1], dtype=np.float64)
ADAPTER_BASE_ANG_B = np.array([0.2, -0.1, 0.15], dtype=np.float64)


# ================================================================ 延迟 FIFO（每物理步推进一次）
class PerStepDelayFifo:
    """q_des 进入 PD 前的物理步级 FIFO。1 个 q_des 采样会连续推送 ppc 次（同值），
    因此 FIFO 长度 N = delay_steps 时，PD 看到的命令恰好来自 N 个物理步（=N×dt）之前。

    语义：pd_in[k] = history[k-N]，其中 history 在控制周期内每步追加当前 q_des。
    """

    def __init__(self, delay_steps, pre_history):
        assert delay_steps > 0
        self.delay_steps = int(delay_steps)
        self.buf = [np.asarray(pre_history, dtype=np.float64).copy() for _ in range(self.delay_steps)]

    def push_pop(self, cur):
        cur = np.asarray(cur, dtype=np.float64)
        self.buf.append(cur.copy())
        return np.asarray(self.buf.pop(0), dtype=np.float64)


def delay_fifo_unit_test(delay_steps=10, ppt=10, n_steps=30):
    """脉冲测试：k=0 时推入 1.0，其余推 0.0；验证 pd_in[k] = history[k-delay_steps]。"""
    fifo = PerStepDelayFifo(delay_steps, pre_history=0.0)
    pd_in = []
    for k in range(n_steps):
        cur = 1.0 if k == 0 else 0.0
        pd_in.append(fifo.push_pop(cur).item())
    pd_in = np.asarray(pd_in)
    assert np.allclose(pd_in[:delay_steps], 0.0), f"pre-history 应全 0: {pd_in[:delay_steps]}"
    assert abs(pd_in[delay_steps] - 1.0) < 1e-9, f"脉冲应恰好滞后 {delay_steps} 步: pd_in[{delay_steps}]={pd_in[delay_steps]}"
    assert abs(pd_in[delay_steps - 1]) < 1e-9 and abs(pd_in[delay_steps + 1]) < 1e-9
    print(f"delay FIFO unit test PASS: 滞后恰好 {delay_steps} 物理步 = {delay_steps * 0.002 * 1000} ms")


# ================================================================ 历史 ring（与 deploy hist_observations.py 一致）
class RingHistory:
    def __init__(self, capacity, dim):
        self.capacity, self.dim = int(capacity), int(dim)
        self.buf = np.zeros((self.capacity, self.dim), dtype=np.float32)
        self.head = 0
        self.size = 0

    def push(self, x):
        x = np.asarray(x, dtype=np.float32).reshape(self.dim)
        tail = (self.head + self.size) % self.capacity
        self.buf[tail] = x
        if self.size < self.capacity:
            self.size += 1
        else:
            self.head = (self.head + 1) % self.capacity

    def flatten(self):
        if self.size < self.capacity:
            out = np.zeros((self.capacity, self.dim), dtype=np.float32)
            if self.size > 0:
                out[-self.size:] = self.buf[: self.size]
            return out.reshape(-1)
        return np.concatenate([self.buf[self.head:], self.buf[: self.head]], axis=0).reshape(-1)


class ObsHistory:
    def __init__(self, history_cfg):
        self.rings = {n: RingHistory(c["history_length"], c["dim"]) for n, c in history_cfg.items()}
        self.order = list(history_cfg.keys())

    def push_item(self, name, vec):
        self.rings[name].push(vec)

    def build_obs(self):
        return np.concatenate([self.rings[n].flatten() for n in self.order]).astype(np.float32)


# ================================================================ 契约
def load_contract(cfg_path):
    p = Path(cfg_path)
    if not p.is_file():
        p = DEPLOY / p
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    rl = cfg["rl_model"]
    obs = rl["observations"]
    total = sum(v["dim"] * v["history_length"] for v in obs.values())
    return {
        "cfg": cfg,
        "control_dt": float(cfg["control_dt"]),
        "obs_groups": obs,
        "obs_total": total,
        "motion_path": str((DEPLOY / rl["motion_path"]).resolve()),
        "policy_path": str((DEPLOY / rl["policy_path"]).resolve()),
        "xml_path": str((DEPLOY / cfg["robot"]["sim2sim"]["xml_path"]).resolve()),
        "sim2sim_order": list(cfg["robot"]["sim2sim"]["joint_sequence"]),
    }


# ================================================================ ONNX 策略
# 显式、版本化的 metadata joint-name 兼容表：{policy 文件后缀: 29 个关节名（policy order）}
# 当前 dance_9 policy 为 29 名，无需兼容表；历史 28 名格式在确认后在下方注册并写明来源。
METADATA_JOINT_COMPAT_TABLE = {}


def validate_policy_metadata(names, default_q, kp, kd, action_scale, policy_path="",
                             compat_table=None):
    """metadata joint-name/数组维度校验（loader 与 T5 共用同一 validation path，N3.1-P1.2/M6）。
    通过则返回 29 个 joint names；28-name 且已注册兼容表 -> 返回兼容表 names；
    否则 ValueError（禁止隐式补齐）。"""
    compat_table = METADATA_JOINT_COMPAT_TABLE if compat_table is None else compat_table
    names = list(names)
    for _suffix, _names29 in compat_table.items():
        if str(policy_path).endswith(_suffix):
            return list(_names29)
    if len(names) != 29:
        raise ValueError(
            f"metadata joint_names={len(names)} != 29：禁止自动补齐。"
            f"若确为历史 28-name/29-array 格式，请先注册 METADATA_JOINT_COMPAT_TABLE。"
        )
    if not (len(names) == len(default_q) == len(kp) == len(kd) == len(action_scale) == 29):
        raise ValueError(
            f"metadata 维度不一致: names={len(names)} default_q={len(default_q)} "
            f"kp={len(kp)} kd={len(kd)} scale={len(action_scale)}; 需 29"
        )
    return names


class MimicOnnxPolicy:
    def __init__(self, policy_path, log=print, require_exact=False):
        self.sess = ort.InferenceSession(policy_path, providers=["CPUExecutionProvider"])
        m = self.sess.get_modelmeta().custom_metadata_map
        names = [x for x in m["joint_names"].split(",")]
        self.default_q = np.array([float(x) for x in m["default_joint_pos"].split(",")], dtype=np.float32)
        self.kp = np.array([float(x) for x in m["joint_stiffness"].split(",")], dtype=np.float32)
        self.kd = np.array([float(x) for x in m["joint_damping"].split(",")], dtype=np.float32)
        self.action_scale = np.array([float(x) for x in m["action_scale"].split(",")], dtype=np.float32)
        self.anchor_body_name = m["anchor_body_name"]
        self.anchor_body_id = int(m["anchor_body_id"])
        self.aux_body_names = [x for x in m["body_names"].split(",")]
        self.last_aux = None
        self.AUX_OUT_NAMES = ["actions", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"]
        # N2/N3.1：禁止隐式补齐；28-name 需显式兼容表（validate_policy_metadata 与 T5 共用）
        names = validate_policy_metadata(
            names, self.default_q, self.kp, self.kd, self.action_scale, policy_path,
        )
        if len(names) != len([x for x in m["joint_names"].split(",")]) and len(names) == 29:
            log(f"INFO joint_names={len([x for x in m['joint_names'].split(',')])} -> 使用兼容表")
        self.joint_names = names
        self.action_buffer = np.zeros((29,), dtype=np.float32)
        log(f"policy joints n={len(self.joint_names)}, anchor={self.anchor_body_name}(id {self.anchor_body_id})")

    def infer(self, obs, time_step):
        out = self.sess.run(self.AUX_OUT_NAMES, {
            "obs": obs.reshape(1, -1),
            "time_step": np.array([time_step], dtype=np.float32).reshape(1, 1),
        })
        self.last_aux = {n: np.asarray(v) for n, v in zip(self.AUX_OUT_NAMES, out)}
        return np.asarray(out[0], dtype=np.float32).reshape(-1)

    def q_des_from_action(self, action):
        return action * self.action_scale + self.default_q


# ================================================================ 观测构造器（镜像 deploy mimic_rl_interface.perform_inference）
class MimicObsBuilder:
    def __init__(self, policy: MimicOnnxPolicy, motion, obs_groups):
        self.policy = policy
        self.motion_joint_pos = motion["joint_pos"].astype(np.float32)
        self.motion_joint_vel = motion["joint_vel"].astype(np.float32)
        self.motion_body_quat_w = motion["body_quat_w"].astype(np.float32)
        self.motion_body_pos_w = motion["body_pos_w"].astype(np.float32)
        self.anchor_body_id = policy.anchor_body_id
        self.motion_length = self.motion_joint_pos.shape[0]
        self.world_to_init_motion_quat = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.init_flag = False
        self.time_step = 0
        self.obs_hist = ObsHistory({n: {"dim": int(v["dim"]), "history_length": int(v["history_length"])}
                                    for n, v in obs_groups.items()})

    # -- R1 新增：repeat-first 预填（与 Isaac reset 后的 history 一致）--------------------------
    def terms(self, root_quat, root_ang_vel, mes_q, mes_qdot):
        """计算当前帧六个 observation term（不写入 ring、不推进 time_step）。"""
        t = int(np.clip(self.time_step, 0, self.motion_length - 1))
        curr = np.concatenate([self.motion_joint_pos[t], self.motion_joint_vel[t]]).astype(np.float32)
        anchor_q = self.motion_body_quat_w[t, self.anchor_body_id]
        ori_q = quat_mul(quat_mul(quat_inv(root_quat), self.world_to_init_motion_quat), anchor_q)
        ori_mat = matrix_from_quat(ori_q).astype(np.float32)[..., :2]
        qpos_rel = (np.asarray(mes_q, dtype=np.float32) - self.policy.default_q).astype(np.float32)
        prev_act = self.policy.action_buffer.copy().astype(np.float32)
        return {
            "curr_motion_data": curr,
            "motion_anchor_ori_b": ori_mat.reshape(-1),
            "base_ang_vel": np.asarray(root_ang_vel, dtype=np.float32).reshape(-1),
            "qpos_rel": qpos_rel,
            "qvel": np.asarray(mes_qdot, dtype=np.float32).reshape(-1),
            "action": prev_act,
        }

    def prefill_repeat_first(self, root_quat, root_ang_vel, mes_q, mes_qdot, n=5):
        """第 0 帧 terms 预填 n 个 history 槽（不推进 time_step）。"""
        self.set_init_yaw(root_quat)
        terms = self.terms(root_quat, root_ang_vel, mes_q, mes_qdot)
        for _ in range(n):
            for name, vec in terms.items():
                self.obs_hist.push_item(name, vec)

    # -------------------------------------------------------------------------------------------
    def set_init_yaw(self, root_quat):
        if not self.init_flag:
            self.init_flag = True
            iw = yaw_quaternion(root_quat)
            ia = yaw_quaternion(self.motion_body_quat_w[self.time_step, self.anchor_body_id])
            self.world_to_init_motion_quat = quat_mul(iw, quat_inv(ia)).astype(np.float32)

    def build_obs(self, root_quat, root_ang_vel, mes_q, mes_qdot):
        """镜像 deploy：推 6 个字段到 ring，返回 (obs, ref_frame_idx)"""
        self.set_init_yaw(root_quat)
        terms = self.terms(root_quat, root_ang_vel, mes_q, mes_qdot)
        for name, vec in terms.items():
            self.obs_hist.push_item(name, vec)
        return self.obs_hist.build_obs(), int(np.clip(self.time_step, 0, self.motion_length - 1))

    def advance(self):
        self.time_step += 1


# ================================================================ 失败判定（镜像 Isaac TerminationsCfg v1）
EE_BODIES = ["left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_roll_link", "right_wrist_roll_link"]


def frame_failure(policy, motion, t, root_quat, ee, anchor_z_actual, proj_g_robot_z):
    """纯帧判定：anchor_z → anchor_ori → ee_z → (NaN 调用方处理)。"""
    t = int(t)
    anchor_posz = float(motion["body_pos_w"][t, policy.anchor_body_id, 2])
    if abs(anchor_posz - anchor_z_actual) > 0.4:
        return "anchor_z"
    anchor_q = motion["body_quat_w"][t, policy.anchor_body_id]
    g_ref = quat_apply_inverse(anchor_q, np.array([0.0, 0.0, -1.0]))
    if abs(g_ref[2] - proj_g_robot_z) > 0.8:
        return "anchor_ori"
    anchor_p = motion["body_pos_w"][t, policy.anchor_body_id]
    delta_ori = yaw_quaternion(quat_mul(root_quat, quat_inv(anchor_q)))
    for idx, (ref_z, robot_z) in ee.items():
        if abs(ref_z - robot_z) > 0.4:
            return f"ee_body_z_{idx}"
    return None


# ================================================================ 视频（EGL 离屏 + 确定性跟随相机）
class VideoWriter:
    def __init__(self, model, data, path, width=1280, height=720, fps=30.0):
        self.path = path
        self.width, self.height, self.fps = width, height, fps
        self.last = -1e9
        self.frames = []
        try:
            import mujoco
            if model.vis.global_.offwidth < width:
                model.vis.global_.offwidth = width
            if model.vis.global_.offheight < height:
                model.vis.global_.offheight = height
            self.renderer = mujoco.Renderer(model, height, width)
        except Exception as e:  # pragma: no cover
            print(f"WARN video disabled (renderer): {e}")
            self.renderer = None

    def capture(self, data, sim_time):
        """按仿真时间采样，1/fps 间隔（B31.2 全身可见：独立 MjvCamera distance 4.8 elevation -10 lookat 0.45，单次 update_scene）"""
        if self.renderer is None or sim_time < self.last + 1.0 / self.fps - 1e-9:
            return
        import mujoco
        r = self.renderer
        # B31.3: 独立 MjvCamera，距离 4.0m + 俯角 -10° + 视点 0.30m，再降低高度与 Isaac 6.5m/0.20 对齐
        try:
            cam = mujoco.MjvCamera()
            x, y, z = float(data.qpos[0]), float(data.qpos[1]), float(data.qpos[2])
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat = np.array([x, y, z + 0.30], dtype=np.float64)
            cam.distance = 4.0
            cam.azimuth = 90.0
            cam.elevation = -10.0
            r.update_scene(data, camera=cam)
        except Exception as e:
            print(f"WARN video capture failed (MjvCamera): {e}")
            # fallback: try once more with same independent camera
            try:
                cam = mujoco.MjvCamera()
                x, y, z = float(data.qpos[0]), float(data.qpos[1]), float(data.qpos[2])
                cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                cam.lookat = np.array([x, y, z + 0.30], dtype=np.float64)
                cam.distance = 4.0
                cam.azimuth = 90.0
                cam.elevation = -10.0
                r.update_scene(data, camera=cam)
            except Exception as e2:
                print(f"WARN video fallback failed: {e2}")
                return
        frame = r.render().copy()
        self.frames.append(frame)
        self.last = sim_time

    def save(self):
        if not self.frames:
            return
        try:
            import imageio
            imageio.mimsave(self.path, self.frames, fps=self.fps, codec="libx264", quality=8)
            print(f"video written: {self.path} ({len(self.frames)} frames)")
        except Exception as e:
            try:
                import cv2
                w = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (self.width, self.height))
                for f in self.frames:
                    w.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
                w.release()
                print(f"video written (cv2): {self.path} ({len(self.frames)} frames)")
            except Exception as e2:
                print(f"WARN video save failed: {e} / {e2}")


def body_vel_self_test(config):
    """M4/M5：已知纯平移/纯转动状态验证 mj_objectVelocity(mjOBJ_XBODY) 的 rot:lin 顺序。
    mjOBJ_XBODY 使用与 d.xpos/d.xquat 相同的 body frame；mj_objectVelocity 返回 [rot(3), lin(3)]。"""
    import mujoco
    c = load_contract(config)
    m = mujoco.MjModel.from_xml_path(c["xml_path"])
    d = mujoco.MjData(m)
    bodies = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
    # 选一个远离 root 的 body 放大空间速度差异
    bid = bodies.index("left_wrist_roll_link")
    tmp = np.zeros(6, dtype=np.float64)

    def probe(label, lin, ang, quat, tol_lin=1e-4, tol_ang=1e-4):
        d.qpos[0:3] = np.array([1.0, 2.0, 1.2])
        d.qpos[3:7] = quat
        d.qvel[0:3] = np.asarray(lin, dtype=np.float64)
        d.qvel[3:6] = np.asarray(ang, dtype=np.float64)
        mujoco.mj_forward(m, d)
        tmp[:] = 0.0
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_XBODY, bid, tmp, 0)
        rot_w, lin_w = tmp[0:3].copy(), tmp[3:6].copy()
        print(f"  {label}: lin_w={lin_w.round(6)} rot_w={rot_w.round(6)}")
        return lin_w, rot_w

    # M4 纯平移：无转动 -> 全部 body 线速度=root 线速度，角速度≈0
    lin, rot = probe("pure translation", [1.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                     np.array([1.0, 0.0, 0.0, 0.0]))
    assert abs(lin[0] - 1.0) < 1e-4 and abs(lin[1]) < 1e-4 and abs(lin[2]) < 1e-4, f"M4 lin FAIL: {lin}"
    assert np.max(np.abs(rot)) < 1e-4, f"M4 rot FAIL: {rot}"
    print("M4 PASS: 纯平移 -> lin≈(1,0,0), rot≈0")

    # M5 纯转动：绕 z 角速度 2 rad/s，root 在原点 -> v = ω×r
    quat_id = np.array([1.0, 0.0, 0.0, 0.0])
    d.qpos[0:3] = np.zeros(3)
    d.qpos[3:7] = quat_id
    d.qvel[0:3] = np.zeros(3)
    d.qvel[3:6] = np.array([0.0, 0.0, 2.0])
    mujoco.mj_forward(m, d)
    tmp[:] = 0.0
    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_XBODY, bid, tmp, 0)
    rot_w, lin_w = tmp[0:3].copy(), tmp[3:6].copy()
    r = d.xpos[bid].copy()
    exp_lin = np.cross(np.array([0.0, 0.0, 2.0]), r)  # ω×r（刚体瞬时速度）
    print(f"  pure rotation: r={r.round(4)} lin_w={lin_w.round(5)} exp=ω×r={exp_lin.round(5)} rot_w={rot_w.round(5)}")
    assert abs(rot_w[2] - 2.0) < 1e-4 and abs(rot_w[0]) < 1e-4 and abs(rot_w[1]) < 1e-4, f"M5 rot FAIL: {rot_w}"
    assert np.max(np.abs(lin_w - exp_lin)) < 1e-3, f"M5 lin FAIL: {lin_w} vs {exp_lin}"
    print("M5 PASS: 纯转动 -> rot_w≈(0,0,2), lin_w≈ω×r")
    print("body-vel self-test PASS (M4/M5)")


# ================================================================ 通用工具
def sha256_bytes(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def repo_stamp(repo):
    try:
        head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True, timeout=10)
        return {"commit": head.stdout.strip(), "dirty": bool(dirty.stdout.strip()), "dirty_files": len(dirty.stdout.splitlines())}
    except Exception as e:
        return {"commit": "unavailable", "dirty": None, "error": str(e)}


# ================================================================ main
def main():
    ap = argparse.ArgumentParser(description="MuJoCo direct ONNX runner (Lab dance_9, R1)")
    ap.add_argument("--config", default=None, help="mimic_deploy yaml (relative deploy/)")
    ap.add_argument("--duration", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fixed-delay-ms", type=float, default=0.0)
    ap.add_argument("--gain-scale", type=float, default=1.0)
    ap.add_argument("--mode", choices=["as_shipped", "aligned"], default="aligned")
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--initial-state", default=None, help="canonical_initial_state.npz（优先于 motion start_frame）")
    ap.add_argument("--history-init", choices=["zero", "repeat-first"], default="repeat-first")
    ap.add_argument("--delay-self-test", action="store_true", help="只跑延迟 FIFO 离线单元测试后退出")
    ap.add_argument("--adapter-test", default=None, help="G0b：4 组静态注入状态 dump canonical readback 到 npz")
    ap.add_argument("--body-vel-self-test", action="store_true", help="M4/M5：纯平移/纯转动验证 mjOBJ_XBODY velocity 顺序")
    ap.add_argument("--output", default=None)
    ap.add_argument("--record-video", action="store_true")
    ap.add_argument("--offscreen", action="store_true")
    args = ap.parse_args()
    if args.delay_self_test:
        delay_fifo_unit_test()
        return
    if args.body_vel_self_test:
        if args.config is None:
            ap.error("--body-vel-self-test 需要 --config")
        body_vel_self_test(args.config)
        return
    if args.config is None or args.output is None:
        ap.error("--config 与 --output 必填（非 --delay-self-test 模式）")
    if args.fixed_delay_ms != 0.0:
        raise SystemExit(
            f"非零 --fixed-delay-ms 当前被拒绝：FIFO 修复后必须先通过 --delay-self-test 才能执行 delay run（R1/R7 硬规则）。"
        )

    os.makedirs(args.output, exist_ok=True)
    log_path = Path(args.output) / "stdout.log"
    _logf = open(log_path, "w", encoding="utf-8")
    def log(msg=""):
        print(msg, flush=True)
        _logf.write(str(msg) + "\n")
        _logf.flush()

    import mujoco
    c = load_contract(args.config)
    assert c["obs_total"] == 770, f"obs total {c['obs_total']} != 770"
    policy = MimicOnnxPolicy(c["policy_path"], log=log)
    motion = {k: np.load(c["motion_path"])[k] for k in
              ["joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"]}
    m = mujoco.MjModel.from_xml_path(c["xml_path"])
    d = mujoco.MjData(m)

    physics_dt = float(m.opt.timestep)
    control_dt = c["control_dt"]
    ppc = max(1, int(round(control_dt / physics_dt)))  # 期望 10（2ms×10=20ms）
    log(f"physics_dt={physics_dt} control_dt={control_dt} ppc={ppc} mode={args.mode} "
        f"history_init={args.history_init} start_frame={args.start_frame}")

    # ---------------- MJCF 顺序断言（R1：按名严格映射，禁止静默填 0）----------------
    # N3.1-P0.1：hinge 保留原始 jntid（m.jnt_dofadr 的索引是原始 jntid，不是过滤后序号）
    hinge_joint_ids = [jid for jid in range(m.njnt)
                       if m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE]
    hinge_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
                   for jid in hinge_joint_ids]
    joint_id_by_name = dict(zip(hinge_names, hinge_joint_ids))
    act_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
    s2s = c["sim2sim_order"]
    jnt_dof = len(d.qpos) - 7
    assert jnt_dof == len(hinge_names) == 29, f"hinge dofs={jnt_dof} names={len(hinge_names)}; 期望 29"
    assert len(act_names) == len(s2s) == 29, f"actuators={len(act_names)} sim2sim_order={len(s2s)}; 期望 29"
    if set(hinge_names) != set(s2s) or set(act_names) != set(s2s):
        raise SystemExit(f"MJCF joint/actuator 集合 != yaml sim2sim_order（需更新兼容表）:\n"
                         f"hinge 缺 {set(s2s) - set(hinge_names)} 多 {set(hinge_names) - set(s2s)}\n"
                         f"act  缺 {set(s2s) - set(act_names)} 多 {set(act_names) - set(s2s)}")
    hinge_ok = hinge_names == s2s
    act_ok = act_names == s2s
    log(f"mjcf hinge order == sim2sim_order: {hinge_ok}; actuator order == sim2sim_order: {act_ok}")
    # 映射器（name-based，rj=raw joint order, ra=raw actuator order）
    def j2s(arr):   # raw joint order -> sim2sim order
        return map_by_name(arr, hinge_names, s2s, "joint->s2s")
    def s2j(arr):   # sim2sim order -> raw joint order
        return map_by_name(arr, s2s, hinge_names, "s2s->joint")
    def a2s(arr):   # raw actuator order -> sim2sim order
        return map_by_name(arr, act_names, s2s, "actuator->s2s")
    def s2a(arr):   # sim2sim order -> raw actuator order
        return map_by_name(arr, s2s, act_names, "s2s->actuator")

    # ---------------- 初始状态（R2 canonical 优先；否则 motion start_frame）----------------
    init_from = "motion_frame"
    if args.initial_state:
        init_npz = np.load(args.initial_state)
        assert list(init_npz["joint_names"]) == list(policy.joint_names), \
            f"canonical_initial_state joint_names 与 policy 不一致"
        q_pol = np.asarray(init_npz["joint_pos"], dtype=np.float64)
        dq_pol = np.asarray(init_npz["joint_vel"], dtype=np.float64)
        root_pos = np.asarray(init_npz["root_pos"], dtype=np.float64)
        root_quat = np.asarray(init_npz["root_quat"], dtype=np.float64)          # wxyz
        root_lin_vel_w = np.asarray(init_npz["root_lin_vel_w"], dtype=np.float64)
        root_ang_vel_b = np.asarray(init_npz["root_ang_vel_b"], dtype=np.float64)  # body frame
        init_from = "canonical_npz"
    else:
        f0 = int(np.clip(args.start_frame, 0, motion["joint_pos"].shape[0] - 1))
        q_pol = motion["joint_pos"][f0].astype(np.float64)
        dq_pol = motion["joint_vel"][f0].astype(np.float64)
        root_pos = motion["body_pos_w"][f0, 0].astype(np.float64)
        root_quat = motion["body_quat_w"][f0, 0].astype(np.float64)
        root_lin_vel_w = motion["body_lin_vel_w"][f0, 0].astype(np.float64)
        root_ang_vel_b = quat_apply_inverse(root_quat, motion["body_ang_vel_w"][f0, 0].astype(np.float64))
    d.qpos[7:] = s2j(map_by_name(q_pol, policy.joint_names, s2s, "policy->s2s(q)"))
    d.qvel[6:] = s2j(map_by_name(dq_pol, policy.joint_names, s2s, "policy->s2s(dq)"))
    d.qpos[0:3] = root_pos
    d.qpos[3:7] = root_quat
    d.qvel[3:6] = root_ang_vel_b   # MuJoCo free joint qvel[3:6] 为 body-frame 角速度
    d.qvel[0:3] = root_lin_vel_w
    mujoco.mj_forward(m, d)
    initial_root_z = float(d.qpos[2])
    readback_q_pol = map_by_name(j2s(d.qpos[7:].copy()), s2s, policy.joint_names, "readback q")
    readback_dq_pol = map_by_name(j2s(d.qvel[6:].copy()), s2s, policy.joint_names, "readback dq")
    init_q_err = float(np.max(np.abs(readback_q_pol - q_pol)))
    init_dq_err = float(np.max(np.abs(readback_dq_pol - dq_pol)))
    log(f"init from={init_from} root_pos={d.qpos[0:3]} root_z={initial_root_z:.4f} "
        f"quat={d.qpos[3:7]} | readback q max_err={init_q_err:.2e} dq max_err={init_dq_err:.2e}")
    initial_state_readback = {
        "joint_pos_policy": readback_q_pol.tolist(),
        "joint_vel_policy": readback_dq_pol.tolist(),
        "root_pos": d.qpos[0:3].tolist(),
        "root_quat": d.qpos[3:7].tolist(),
        "root_lin_vel_w": d.qvel[0:3].tolist(),
        "root_ang_vel_b": d.qvel[3:6].tolist(),
        "q_max_err_vs_requested": init_q_err,
        "dq_max_err_vs_requested": init_dq_err,
    }

    # ---- G0b adapter test：4 组静态注入状态，dump canonical readback（平铺 key）----
    if args.adapter_test:
        base_root_pos = d.qpos[0:3].copy()
        npz_keys = {}
        for nm, quat in ADAPTER_STATES.items():
            d.qpos[7:] = s2j(map_by_name(q_pol, policy.joint_names, s2s, "ad bq"))
            d.qvel[6:] = s2j(map_by_name(dq_pol, policy.joint_names, s2s, "ad bdq"))
            d.qpos[0:3] = base_root_pos
            d.qpos[3:7] = quat
            d.qvel[0:3] = ADAPTER_BASE_LIN
            d.qvel[3:6] = ADAPTER_BASE_ANG_B
            mujoco.mj_forward(m, d)
            rq_pol = map_by_name(j2s(d.qpos[7:].copy()), s2s, policy.joint_names, "ad rbq")
            rdq_pol = map_by_name(j2s(d.qvel[6:].copy()), s2s, policy.joint_names, "ad rbdq")
            vals = {"joint_pos": rq_pol, "joint_vel": rdq_pol, "root_pos": d.qpos[0:3].copy(),
                    "root_quat": d.qpos[3:7].copy(), "root_lin_vel_w": d.qvel[0:3].copy(),
                    "root_ang_vel_b": d.qvel[3:6].copy()}
            for fld, arr in vals.items():
                npz_keys[f"state_{nm}_{fld}"] = np.asarray(arr, dtype=np.float32)
        np.savez(args.adapter_test, **npz_keys)
        log(f"G0b adapter dump written: {args.adapter_test} ({len(npz_keys)} keys)")
        _logf.close()
        return

    # ---------------- 控制参数与顺序 ----------------
    kp_s2s = map_by_name(policy.kp * args.gain_scale, policy.joint_names, s2s, "kp").astype(np.float64)
    kd_s2s = map_by_name(policy.kd * args.gain_scale, policy.joint_names, s2s, "kd").astype(np.float64)
    scale_s2s = map_by_name(policy.action_scale, policy.joint_names, s2s, "scale").astype(np.float64)
    builder = MimicObsBuilder(policy, motion, c["obs_groups"])
    builder.time_step = int(args.start_frame)
    writer = VideoWriter(m, d, str(Path(args.output) / "video.mp4")) if args.record_video else None

    # 失败判定辅助：ONNX aux 14-body 名 -> MJCF body id
    mj_body_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
    def mj_body_id(name): return mj_body_names.index(name)
    missing_mj = [n for n in policy.aux_body_names if n not in mj_body_names]
    assert not missing_mj, f"MJCF 缺 body: {missing_mj}"
    assert policy.anchor_body_name in policy.aux_body_names and policy.anchor_body_id == 0

    # N3.1-P0.2：motion 30-body -> policy 14-body 按名映射（motion order == MJCF robot body order, 扣 world）
    motion_body_names = [n for n in mj_body_names if n != "world"]
    assert len(motion_body_names) == motion["body_pos_w"].shape[1], \
        f"motion 全身 body 数 {motion['body_pos_w'].shape[1]} != MJCF robot body 数 {len(motion_body_names)}（需确认导出顺序）"
    motion_body_indices = []
    for _nm in policy.aux_body_names:
        if _nm not in motion_body_names:
            raise ValueError(f"policy 14-body {_nm} 不在 motion body 表: {motion_body_names}")
        motion_body_indices.append(motion_body_names.index(_nm))
    assert len(motion_body_indices) == len(policy.aux_body_names) == 14
    motion_body_indices = np.asarray(motion_body_indices, dtype=np.int64)
    log(f"14-body 按名映射: policy_aux[{len(policy.aux_body_names)}] -> motion idx "
        f"{motion_body_indices.tolist()}")

    np.random.seed(args.seed)  # 本 runner 无随机源，仅记录；MuJoCo 无 seeded RNG

    n_cycles = int(round(args.duration / control_dt))
    rec = {k: [] for k in ["policy_step", "t", "time_step", "obs(770)", "action(29)", "q_des(29)",
                           "q(29)", "dq(29)", "tau_pd_raw(29)", "applied(29)", "applied_raw_mj(29)",
                           "qfrc_actuator(29)",
                           "tau_pd_raw_policy(29)", "applied_policy(29)", "qfrc_actuator_policy(29)",
                           "root_pos(3)", "root_quat(4)", "root_lin_vel(3)", "root_ang_vel(3)",
                           "done", "fall_reason", "actual_processed_q_target(29)",
                           "reference_body_pos_w(14,3)", "reference_body_quat_w(14,4)",
                           "reference_body_lin_vel_w(14,3)", "reference_body_ang_vel_w(14,3)",
                           "actual_body_pos_w(14,3)", "actual_body_quat_w(14,4)",
                           "actual_body_lin_vel_w(14,3)", "actual_body_ang_vel_w(14,3)"]}
    tmp6 = np.zeros(6, dtype=np.float64)  # mj_objectVelocity 输出缓冲
    fall_reason = None
    fail_cycle = None
    delay_fifo = None
    fifo_used = False
    t0 = time.time()

    # repeat-first 预填（R1：aligned 默认；as_shipped 保持官方前置补零）
    if args.mode == "aligned" and args.history_init == "repeat-first":
        builder.prefill_repeat_first(d.qpos[3:7].copy(), d.qvel[3:6].copy(),
                                     readback_q_pol, readback_dq_pol)
        log("history prefill: repeat-first (5 槽同第 0 帧 terms)")

    for cyc in range(n_cycles):
        t = cyc * control_dt
        # state_t（raw joint order -> s2s -> policy order）
        q_s2s = j2s(d.qpos[7:].copy()).astype(np.float32)
        dq_s2s = j2s(d.qvel[6:].copy()).astype(np.float32)
        root_quat = d.qpos[3:7].copy().astype(np.float32)
        root_ang_vel = d.qvel[3:6].copy().astype(np.float32)
        root_pos = d.qpos[0:3].copy()
        q_pol = map_by_name(q_s2s, s2s, policy.joint_names, "q pol").astype(np.float32)
        dq_pol = map_by_name(dq_s2s, s2s, policy.joint_names, "dq pol").astype(np.float32)
        obs, ref_t = builder.build_obs(root_quat, root_ang_vel, q_pol, dq_pol)
        action = policy.infer(obs, float(ref_t))
        policy.action_buffer[:] = action
        q_des_pol = policy.q_des_from_action(action)
        q_des_s2s = map_by_name(q_des_pol.astype(np.float64), policy.joint_names, s2s, "q_des s2s")

        # 失败判定（镜像 isaac TerminationsCfg；参考帧=ref_t；aux 来自 ONNX 14-body 输出）
        reason = None
        aux_bp = policy.last_aux["body_pos_w"][0].astype(np.float64)
        aux_bq = policy.last_aux["body_quat_w"][0].astype(np.float64)
        anchor_posz = float(aux_bp[policy.anchor_body_id, 2])
        anchor_z_actual = float(d.xpos[mj_body_id(policy.anchor_body_name), 2])
        g_robot_b = quat_apply_inverse(root_quat, np.array([0.0, 0.0, -1.0]))
        if abs(anchor_posz - anchor_z_actual) > 0.4:
            reason = "anchor_z"
        if reason is None:
            g_ref = quat_apply_inverse(aux_bq[policy.anchor_body_id], np.array([0.0, 0.0, -1.0]))
            if abs(g_ref[2] - g_robot_b[2]) > 0.8:
                reason = "anchor_ori"
        if reason is None:
            drot = yaw_quaternion(quat_mul(root_quat, quat_inv(aux_bq[policy.anchor_body_id])))
            for nm_aux in EE_BODIES:
                if nm_aux not in policy.aux_body_names:
                    continue
                ai = policy.aux_body_names.index(nm_aux)
                ref_z = anchor_posz + float(quat_apply(drot, aux_bp[ai] - aux_bp[policy.anchor_body_id])[2])
                robot_z = float(d.xpos[mj_body_id(nm_aux), 2])
                if abs(ref_z - robot_z) > 0.4:
                    reason = "ee_body_z_" + nm_aux
                    break
        if reason is None and (not np.isfinite(obs).all() or not np.isfinite(q_pol).all() or not np.isfinite(dq_pol).all()):
            reason = "nan"

        # 记录（tau/applied 在物理步内累积，见下）
        rec["policy_step"].append(cyc)
        rec["t"].append(t)
        rec["time_step"].append(ref_t)
        rec["obs(770)"].append(obs)
        rec["action(29)"].append(action)
        rec["q_des(29)"].append(q_des_pol.astype(np.float32))
        rec["q(29)"].append(q_pol)
        rec["dq(29)"].append(dq_pol)
        rec["root_pos(3)"].append(root_pos.astype(np.float32))
        rec["root_quat(4)"].append(root_quat)
        rec["root_lin_vel(3)"].append(d.qvel[0:3].copy().astype(np.float32))
        rec["root_ang_vel(3)"].append(root_ang_vel)
        rec["done"].append(1 if reason else 0)
        rec["fall_reason"].append(reason or "")
        # 14-body reference/actual（N3.1：reference 按名筛选到 policy 14-body；actual 用 mjOBJ_XBODY+rot:lin）
        rec["reference_body_pos_w(14,3)"].append(motion["body_pos_w"][ref_t, motion_body_indices].astype(np.float32))
        rec["reference_body_quat_w(14,4)"].append(motion["body_quat_w"][ref_t, motion_body_indices].astype(np.float32))
        rec["reference_body_lin_vel_w(14,3)"].append(motion["body_lin_vel_w"][ref_t, motion_body_indices].astype(np.float32))
        rec["reference_body_ang_vel_w(14,3)"].append(motion["body_ang_vel_w"][ref_t, motion_body_indices].astype(np.float32))
        _act_bp = []
        _act_bq = []
        _act_blv = []
        _act_bav = []
        for _nm in policy.aux_body_names:
            _bid = mj_body_id(_nm)
            _act_bp.append(d.xpos[_bid].astype(np.float32))
            _act_bq.append(d.xquat[_bid].astype(np.float32))
            # mjOBJ_XBODY 与 d.xpos/d.xquat 同一 body frame；mj_objectVelocity 返回 [rot(3), lin(3)]
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_XBODY, _bid, tmp6, 0)
            _act_bav.append(tmp6[0:3].astype(np.float32))  # rot 在前
            _act_blv.append(tmp6[3:6].astype(np.float32))  # lin 在后
        rec["actual_body_pos_w(14,3)"].append(np.stack(_act_bp))
        rec["actual_body_quat_w(14,4)"].append(np.stack(_act_bq))
        rec["actual_body_lin_vel_w(14,3)"].append(np.stack(_act_blv))
        rec["actual_body_ang_vel_w(14,3)"].append(np.stack(_act_bav))

        # 延迟 FIFO（每物理步推进；启动前历史 target = 首个 q_des，repeat-first 语义）
        if delay_fifo is None and args.fixed_delay_ms != 0.0:
            delay_fifo = PerStepDelayFifo(int(round(args.fixed_delay_ms / (physics_dt * 1000))), pre_history=q_des_s2s)
        # PD 每物理步（v1 语义：每步读当前 q/dq；q_des_eff 为预延迟/无延迟目标）
        step_tau = []
        step_applied = []
        step_applied_raw = []
        step_qfrc = []
        for _ in range(ppc):
            if delay_fifo is not None:
                fifo_used = True
                q_des_eff = delay_fifo.push_pop(q_des_s2s)
            else:
                q_des_eff = q_des_s2s
            q_now = j2s(d.qpos[7:].copy())
            dq_now = j2s(d.qvel[6:].copy())
            tau_s2s = (q_des_eff - q_now) * kp_s2s - dq_now * kd_s2s
            d.ctrl[:] = s2a(tau_s2s.astype(np.float64))
            step_tau.append(tau_s2s.astype(np.float32))
            mujoco.mj_step(m, d)
            if writer is not None:
                writer.capture(d, float(d.time))
            step_applied.append(a2s(d.actuator_force.copy()).astype(np.float32))
            step_applied_raw.append(d.actuator_force.copy().astype(np.float32))
            step_qfrc.append(d.qfrc_actuator[6:].copy().astype(np.float32))
        rec["tau_pd_raw(29)"].append(np.mean(step_tau, axis=0))                       # s2s order
        rec["applied(29)"].append(step_applied[-1])                                   # s2s order
        rec["applied_raw_mj(29)"].append(step_applied_raw[-1])                        # raw hinge order
        rec["qfrc_actuator(29)"].append(step_qfrc[-1])                                # raw hinge order
        # N3.1-P1.6：torque 统一 policy order 副本（q/dq/action 均为 policy order）
        rec["tau_pd_raw_policy(29)"].append(map_by_name(np.mean(step_tau, axis=0).astype(np.float64), s2s, policy.joint_names, "tau->pol").astype(np.float32))
        rec["applied_policy(29)"].append(map_by_name(step_applied[-1].astype(np.float64), s2s, policy.joint_names, "app->pol").astype(np.float32))
        rec["qfrc_actuator_policy(29)"].append(map_by_name(step_qfrc[-1].astype(np.float64), s2s, policy.joint_names, "qfrc->pol").astype(np.float32))
        rec["actual_processed_q_target(29)"].append(q_des_pol.astype(np.float32))  # 本 runner 目标=\记录 q_des
        builder.advance()
        if reason is not None:
            fall_reason = reason
            fail_cycle = cyc
            log(f"FALL at cycle {cyc} t={t:.2f}s reason={reason}")
            break
        if (cyc + 1) % 100 == 0:
            log(f"cycle {cyc+1}/{n_cycles} t={t:.1f}s z={d.qpos[2]:.3f} wall={time.time()-t0:.1f}s")

    final_root_z = float(d.qpos[2])
    log(f"done: cycles={len(rec['policy_step'])} fall={fall_reason} execute_ratio="
        f"{(fail_cycle if fail_cycle is not None else n_cycles)/n_cycles:.3f} fifo_used={fifo_used}")

    # ---------------- 保存 ----------------
    npz_out = {k: (np.asarray(v, dtype=np.float32) if k.endswith(")") else np.array(v)) for k, v in rec.items()}
    np.savez_compressed(Path(args.output) / "policy_trace.npz", **npz_out)
    with open(Path(args.output) / "summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "engine": "mujoco", "duration_s": args.duration, "control_dt": control_dt, "physics_dt": physics_dt,
            "ppc": ppc, "cycles_planned": n_cycles, "cycles_done": len(rec["policy_step"]),
            "fall_reason": fall_reason, "fall_cycle": fail_cycle, "seed": args.seed,
            "fixed_delay_ms": args.fixed_delay_ms, "gain_scale": args.gain_scale,
            "mode": args.mode, "start_frame": args.start_frame, "history_init": args.history_init,
            "initial_state_source": init_from,
            "obs_total": int(c["obs_total"]), "fifo_used": fifo_used,
            "initial_root_z": initial_root_z, "final_root_z": final_root_z,
            "initial_q_max_err": init_q_err, "initial_dq_max_err": init_dq_err,
        }, f, indent=2, ensure_ascii=False)
    cfg_path = _resolve_cfg(args.config, DEPLOY)
    # N2/N4：从 mjModel 读逐关节 runtime 参数（armature/gain/effort/damping/frictionloss/velocity limit）
    # N3.1-P0.1：joint id 用原始 jntid（jnt_dofadr 索引=原始 jntid）；附 armature 断言（M1/M2）
    runtime_actuator_params = {}
    armature_sanity = {}
    for _nm in s2s:
        _jid = joint_id_by_name[_nm]
        _did = int(m.jnt_dofadr[_jid])
        _aid = act_names.index(_nm)
        _arm = float(m.dof_armature[_did])
        armature_sanity[_nm] = _arm
        runtime_actuator_params[_nm] = {
            "mujoco_joint_id": _jid,
            "mujoco_dof_id": _did,
            "mujoco_actuator_id": _aid,
            "dof_armature": _arm,
            "dof_damping": float(m.dof_damping[_did]),
            "dof_frictionloss": float(m.dof_frictionloss[_did]),
            "dof_velocity_limit": "UNAVAILABLE(mujoco 3.2.7 无 dof_velocity 字段; joint 级未设 velocity 限制)",
            "actuator_dyntype": str(mujoco.mjtDyn(m.actuator_dyntype[_aid]).name),
            "actuator_gaintype": str(mujoco.mjtGain(m.actuator_gaintype[_aid]).name),
            "actuator_biastype": str(mujoco.mjtBias(m.actuator_biastype[_aid]).name),
            "actuator_gear": float(m.actuator_gear[_aid, 0]),
            "actuator_gainprm0": float(m.actuator_gainprm[_aid, 0]),
            "actuator_ctrlrange": m.actuator_ctrlrange[_aid].tolist(),
            "actuator_force_range": m.actuator_forcerange[_aid].tolist(),
            "actuator_velrange": "UNAVAILABLE(mujoco 3.2.7 无 actuator_velrange/vel_limited 字段；本 XML 关节未见 velocity 限制)",
        }
    # M1/M2 验收：官方 XML 下 armature 必须命中 class 基准值，否则断言失败（防止 joint/dof id 再错位）
    if cfg_path.name.endswith("mimic_dance_9.yaml"):
        assert abs(armature_sanity.get("right_hip_pitch_joint", -1.0) - 0.0968) < 1e-6, \
            f"M1 FAIL right_hip_pitch armature={armature_sanity.get('right_hip_pitch_joint')} (期望 0.0968)"
        assert abs(armature_sanity.get("right_elbow_pitch_joint", -1.0) - 0.0685) < 1e-6, \
            f"M2 FAIL right_elbow_pitch armature={armature_sanity.get('right_elbow_pitch_joint')} (期望 0.0685)"
    log(f"armature sanity: right_hip_pitch={armature_sanity.get('right_hip_pitch_joint'):.4f} "
        f"right_elbow_pitch={armature_sanity.get('right_elbow_pitch_joint'):.4f} (M1/M2)")
    _mujoco_model_sha = sha256_bytes(c["xml_path"])
    _mujoco_runner_sha = sha256_bytes(__file__)
    _mujoco_canon_sha = sha256_bytes(args.initial_state) if args.initial_state else None
    manifest = {
        "engine": "mujoco",
        "mode": args.mode, "start_frame": args.start_frame, "history_init": args.history_init,
        "initial_state_source": init_from, "initial_state_npz_sha256": _mujoco_canon_sha,
        "canonical_npz_sha256": _mujoco_canon_sha,
        "model_asset_path": c["xml_path"],
        "model_asset_sha256": _mujoco_model_sha,
        "model_used_sha256": _mujoco_model_sha,
        "model_used_name": Path(c["xml_path"]).name,
        "runner_sha256": _mujoco_runner_sha,
        "duration_s": args.duration, "seed": args.seed, "seed_used": {"numpy": args.seed, "mujoco": "not_used"},
        "control_dt": control_dt, "physics_dt": physics_dt, "ppc": ppc,
        "requested_delay_ms": args.fixed_delay_ms, "measured_delay_ms": 0.0,
        "gain_scale": args.gain_scale,
        "hashes": {
            "policy": sha256_bytes(c["policy_path"]),
            "motion": sha256_bytes(c["motion_path"]),
            "xml": _mujoco_model_sha,
            "model": _mujoco_model_sha,
            "config": sha256_bytes(str(cfg_path)),
            "script": _mujoco_runner_sha,
            "runner": _mujoco_runner_sha,
            "initial_state": _mujoco_canon_sha,
            "canonical": _mujoco_canon_sha,
        },
        "repo": {"lab": repo_stamp(REPO)},
        "policy_metadata": {
            "joint_names": policy.joint_names,
            "default_q": policy.default_q.tolist(), "action_scale": policy.action_scale.tolist(),
            "kp": policy.kp.tolist(), "kd": policy.kd.tolist(),
            "body_names": policy.aux_body_names,
            "anchor_body_name": policy.anchor_body_name, "anchor_body_id": policy.anchor_body_id,
        },
        "runtime_mjmodel_actuator_params": runtime_actuator_params,
        "torque_array_order": {
            "note": "tau_pd_raw/applied(29) 为 sim2sim_order；applied_raw_mj/qfrc_actuator(29) 为 raw hinge order；",
            "policy_order_copies": ["tau_pd_raw_policy(29)", "applied_policy(29)", "qfrc_actuator_policy(29)"],
            "hinge_order_matches_s2s": hinge_ok,
        },
        "motion_body_mapping": {
            "motion_body_names": motion_body_names,
            "motion_body_indices_14": motion_body_indices.tolist(),
            "policy_aux_body_names": policy.aux_body_names,
            "motion_body_count": int(motion["body_pos_w"].shape[1]),
        },
        "runtime_mjmodel_solver": {
            "timestep": float(m.opt.timestep),
            "integrator": str(mujoco.mjtIntegrator(m.opt.integrator).name),
            "solver": str(mujoco.mjtSolver(m.opt.solver).name),
            "iterations": int(m.opt.iterations),
            "tolerance": float(m.opt.tolerance),
        },
        "orders": {
            "policy": [policy.joint_names.index(n) for n in policy.joint_names],
            "sim2sim_order": s2s,
            "mjcf_hinge_order_matches_s2s": hinge_ok,
            "mjcf_actuator_order_matches_s2s": act_ok,
        },
        "kp_s2s": kp_s2s.tolist(), "kd_s2s": kd_s2s.tolist(), "action_scale_s2s": scale_s2s.tolist(),
        "initial_state_readback": initial_state_readback,
        "initial_root_z": initial_root_z, "final_root_z": final_root_z,
        "fall_reason": fall_reason, "cycles_done": len(rec["policy_step"]), "cycles_planned": n_cycles,
        "joint_order_note": "qpos[7:] 与 actuator 顺序若与 sim2sim_order 不同，已按名重映射（见 orders）",
    }
    with open(Path(args.output) / "run_manifest.yaml", "w", encoding="utf-8") as f:
        f.write(yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False, default_flow_style=False))
    if writer is not None:
        writer.save()
    log(f"OK: {args.output}")
    _logf.close()


def _resolve_cfg(config, deploy):
    p = Path(config)
    return p if p.is_file() else deploy / p


if __name__ == "__main__":
    main()