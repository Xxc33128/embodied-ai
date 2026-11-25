#!/usr/bin/env python3
"""P0-1 决定性探针：量测动作→实现的运动映射（对抗审查 P0-1 修复）。

设计（对照 v1 探针的三处不足）：
1. 满幅 × 长序列：action=1.0 持续 30 步，逐步记录 eef 位移 → 分辨
   「设定点语义（稳态 ≈ 0.05 m/步）」vs「严重滞后（≪）」。
2. 旋转满幅 × 20 步 → 累计转角。
3. body/world 判别：先把 EEF 转离轴对齐姿态（绕 z 15 步），再绕 x 探测，
   比较响应旋转轴在世界系还是体系。
4. 同时记录 setpoint 侧证据：controller.goal_pos 每步变化量（如果可读），
   直接区分「设定点动了但手臂没跟上」vs「设定点本身慢」。
"""
import json
import sys

sys.path.insert(0, "/workspace/repo/src")
from gap_repro.sim.libero_session import _ensure_runtime_env  # noqa: E402
_ensure_runtime_env()

import numpy as np  # noqa: E402

OUT = "/workspace/data/l2_action_probe_v2.json"
out = {"schema": "gap_repro.l2_action_probe.v2"}

from gap_repro.sim.libero_session import LiberoSession  # noqa: E402

sess = LiberoSession(suite="libero_goal", task_index=0)
sim = sess.env.env.sim


def goal_pos():
    c = sess.env.env.robots[0].controller
    return np.array(c.goal_pos, dtype=np.float64).ravel()


def eef(obs):
    return np.asarray(obs["robot0_eef_pos"], dtype=np.float64).ravel()


def quat(obs):
    return np.asarray(obs["robot0_eef_quat"], dtype=np.float64).ravel()


def rot_angle_deg(q0, q1):
    # (x,y,z,w) → wxyz
    a = np.array([q0[3], q0[0], q0[1], q0[2]])
    b = np.array([q1[3], q1[0], q1[1], q1[2]])
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    d = abs(float(np.dot(a, b)))
    return float(np.degrees(2 * np.arccos(min(1.0, d))))


# ---- P1: 满幅平移 30 步 ----
obs = sess.reset_to(0, wait_steps=10)
p0, g0 = eef(obs), goal_pos()
traj, gtraj = [p0.copy()], [g0.copy()]
for _ in range(30):
    obs, _, _, _ = sess.step([1.0, 0, 0, 0, 0, 0, -1.0])
    traj.append(eef(obs))
    gtraj.append(goal_pos())
disp = np.array([np.linalg.norm(traj[i + 1] - traj[i]) for i in range(30)])
gdisp = np.array([np.linalg.norm(gtraj[i + 1] - gtraj[i]) for i in range(30)])
out["pos_full"] = {
    "cmd": 1.0, "steps": 30,
    "setpoint_step_mm_first5": (gdisp[:5] * 1000).round(3).tolist(),
    "setpoint_step_mm_last5": (gdisp[-5:] * 1000).round(3).tolist(),
    "eef_step_mm_first5": (disp[:5] * 1000).round(3).tolist(),
    "eef_step_mm_last5": (disp[-5:] * 1000).round(3).tolist(),
    "eef_total_mm": round(float(disp.sum()) * 1000, 2),
    "setpoint_total_mm": round(float(gdisp.sum()) * 1000, 2),
}

# ---- P2: 满幅旋转 20 步 ----
obs = sess.reset_to(0, wait_steps=10)
q0 = quat(obs)
for _ in range(20):
    obs, _, _, _ = sess.step([0, 0, 0, 1.0, 0, 0, -1.0])
out["rot_full"] = {"cmd": 1.0, "steps": 20,
                   "total_deg": round(rot_angle_deg(q0, quat(obs)), 3),
                   "per_step_mean_deg": round(rot_angle_deg(q0, quat(obs)) / 20, 4)}

# ---- P3: body/world 判别（先转离轴对齐，再绕 x 探测）----
obs = sess.reset_to(0, wait_steps=10)
for _ in range(15):  # 改变姿态
    obs, _, _, _ = sess.step([0, 0, 0, 0, 0, 0.5, -1.0])
q_before = quat(obs)
# 绕世界 x 的单位四元数 → 反解动作 rotvec：动作是 delta 旋转向量，
# OSC 内部以某种系解释（这里直接发 x 方向 delta，看响应轴在世界系还是体系）
obs_prev = obs
for _ in range(10):
    obs, _, _, _ = sess.step([0, 0, 0, 0.5, 0, 0, -1.0])


def axis_of(q0, q1):
    a = np.array([q0[3], q0[0], q0[1], q0[2]])
    b = np.array([q1[3], q1[0], q1[1], q1[2]])
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    if float(np.dot(a, b)) < 0:
        b = -b
    # 相对旋转 q_rel = b * a^-1（wxyz）
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    q = np.array([w2 * w1 + x2 * x1 + y2 * y1 + z2 * z1,
                  w2 * -x1 + x2 * w1 + y2 * -z1 + z2 * y1,
                  w2 * -y1 - x2 * z1 + y2 * w1 + z2 * x1,
                  w2 * -z1 + x2 * y1 - y2 * x1 + z2 * w1])
    q /= np.linalg.norm(q)
    w = min(1.0, max(-1.0, q[0]))
    ang = 2 * np.arccos(w)
    s = np.sqrt(max(1e-12, 1 - w * w))
    return q[1:] / s * ang


w_ax = axis_of(q_before, quat(obs))  # 世界系响应轴
# 体系响应轴 = R_world_body^T * w_ax
R = np.zeros(9)
import mujoco  # noqa: E402
mujoco.mju_quat2Mat(R, np.array([q_before[3], q_before[0], q_before[1],
                                 q_before[2]]))
R = R.reshape(3, 3)
b_ax = R.T @ w_ax
out["axis_discrimination"] = {
    "world_axis": np.round(w_ax, 4).tolist(),
    "body_axis": np.round(b_ax, 4).tolist(),
    "closer_to_world_x": bool(abs(w_ax[0]) > abs(b_ax[0])),
}

json.dump(out, open(OUT, "w"), indent=1)
print("PROBE_V2_DONE", json.dumps({k: out[k] for k in
                                   ("pos_full", "rot_full")}, indent=0)[:400],
      flush=True)
