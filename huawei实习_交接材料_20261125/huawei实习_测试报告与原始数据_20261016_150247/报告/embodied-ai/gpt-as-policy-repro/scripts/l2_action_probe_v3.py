#!/usr/bin/env python3
"""P1-新1 决定性探针 v3：delta 旋转的动作系判别（世界系 vs EEF 体系）。

v2 的 axis_of 有四元数叉积符号错误 + 子探针自污染（预旋后未沉降），
其 axis_discrimination 证据作废。v3 修正：
1. 相对旋转用已测 quat_mul（wxyz，contract.py，测试钉定）：
   q_rel = q_after ⊗ conj(q_before)，轴 = q_rel 虚部归一。
2. 预旋（改变姿态使其非轴对齐）后加 20 步沉降（零动作），再探测。
3. 指令量级对账：0.5 × 10 步 × 0.085 rad/单位 = 0.425 rad 预期转角，
   与实测对账（偏离 >20% 即环境状态污染，报告并重跑）。
4. 判别：命令轴为 x。若响应轴（世界系）≈ x → 世界系解释；
   若 R_world_body^T @ a_world ≈ x（即体系轴 ≈ x）→ 体系解释。
"""
import json
import sys

sys.path.insert(0, "/workspace/repo/src")
from gap_repro.sim.libero_session import _ensure_runtime_env  # noqa: E402
_ensure_runtime_env()

import numpy as np  # noqa: E402

from gap_repro.agent.contract import quat_mul  # noqa: E402

OUT = "/workspace/data/l2_action_probe_v3.json"

from gap_repro.sim.libero_session import LiberoSession  # noqa: E402

sess = LiberoSession(suite="libero_goal", task_index=0)


def wxyz(obs):
    q = np.asarray(obs["robot0_eef_quat"], dtype=np.float64).ravel()
    return np.array([q[3], q[0], q[1], q[2]])  # (x,y,z,w) → wxyz


def relative_axis(q0_w, q1_w):
    """q_rel = q1 ⊗ conj(q0)（世界系），返回 (axis_world, angle)。"""
    conj = np.array([q0_w[0], -q0_w[1], -q0_w[2], -q0_w[3]])
    q_rel = quat_mul(q1_w, conj)
    q_rel = q_rel / np.linalg.norm(q_rel)
    if q_rel[0] < 0:
        q_rel = -q_rel
    w = float(np.clip(q_rel[0], -1.0, 1.0))
    angle = 2.0 * np.arccos(w)
    s = np.sqrt(max(1e-12, 1.0 - w * w))
    return q_rel[1:] / s, angle


def quat_to_mat(q_w):
    """wxyz → 3×3 旋转矩阵（世界←体）。"""
    w, x, y, z = q_w
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


out = {"schema": "gap_repro.l2_action_probe.v3"}

# 1) 预旋：绕 z 0.5 × 12 步（改变姿态，离开轴对齐）
obs = sess.reset_to(0, wait_steps=10)
for _ in range(12):
    obs, _, _, _ = sess.step([0, 0, 0, 0, 0, 0.5, -1.0])
# 2) 沉降 20 步（零动作），消除动力学残留
for _ in range(20):
    obs, _, _, _ = sess.step([0, 0, 0, 0, 0, 0, -1.0])
q_before = wxyz(obs)
R_wb = quat_to_mat(q_before / np.linalg.norm(q_before))

# 3) 探测：delta rotvec 沿动作 x，0.5 × 10 步
for _ in range(10):
    obs, _, _, _ = sess.step([0, 0, 0, 0.5, 0, 0, -1.0])
q_after = wxyz(obs)

axis_world, angle = relative_axis(q_before, q_after)
axis_body = R_wb.T @ axis_world  # 世界系轴变换回体系

expected_angle = 10 * 0.5 * 0.085
out["probe"] = {
    "cmd_per_step": 0.5, "steps": 10,
    "expected_angle_rad_pinned_scale": round(expected_angle, 4),
    "measured_angle_rad": round(float(angle), 4),
    "scale_ratio_measured_over_pinned": round(float(angle) / expected_angle, 3),
    "axis_world": np.round(axis_world, 4).tolist(),
    "axis_body": np.round(axis_body, 4).tolist(),
    "quat_before_wxyz": np.round(q_before, 4).tolist(),
}
# 判别：命令轴是 x。哪个系的 x 分量占比高且其余分量小？
wx, bx = abs(axis_world[0]), abs(axis_body[0])
purity = lambda a: float(a[0] / (np.linalg.norm(a) + 1e-12))
out["verdict"] = {
    "world_axis_x_purity": round(purity(axis_world), 4),
    "body_axis_x_purity": round(purity(axis_body), 4),
    "interpretation": "world_frame" if purity(axis_world) > purity(axis_body)
                      else "body_frame",
}
json.dump(out, open(OUT, "w"), indent=1)
print("PROBE_V3_DONE", json.dumps(out["verdict"]), flush=True)
print("scale_ratio:", out["probe"]["scale_ratio_measured_over_pinned"],
      flush=True)
