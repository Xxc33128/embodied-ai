#!/usr/bin/env python3
"""L2 动作契约物理探针（计划 L2 验收项：实测平移、旋转、夹爪正负）。

在钉定环境（gap-sim, LIBERO-PRO@eafdb809, drawer 任务）上实测：
1. 夹爪符号：robosuite 1.4.0 PandaGripper 文档称 −1=open/+1=closed，
   用 gripper_qpos 开合方向实测验证（文档不作为证据，实测才算）。
2. Δpos 方向与尺度：单轴 delta 动作 → EEF 位移（验证米制 + 坐标轴方向）。
3. Δrot 方向：绕 x 轴正 rotvec → EEF 四元数变化（右手系验证）。
4. 夹爪 qpos 行程（policy state 编码输入域）。

每段独立 reset_to(0)，消除漂移。证据落 /workspace/data/l2_action_probe.json。
"""
import json
import sys

sys.path.insert(0, "/workspace/repo/src")
from gap_repro.sim.libero_session import _ensure_runtime_env  # noqa: E402
_ensure_runtime_env()

import numpy as np  # noqa: E402

OUT = "/workspace/data/l2_action_probe.json"
out = {"schema": "gap_repro.l2_action_probe.v1",
       "env": "LIBERO-PRO@eafdb809 libero_goal #0 (open the middle drawer)"}

from gap_repro.sim.libero_session import LiberoSession  # noqa: E402

OPEN_STEPS = 30


def aperture(q):
    """Panda 双指镜像关节：开口度 = q0 − q1（开时两指远离、差值增大）。"""
    q = np.asarray(q, dtype=np.float64).ravel()
    return float(q[0] - q[1]) if q.size >= 2 else float(q[0])


def eef_pose(obs):
    return (np.asarray(obs["robot0_eef_pos"], dtype=np.float64).ravel(),
            np.asarray(obs["robot0_eef_quat"], dtype=np.float64).ravel())


def probe_gripper(sess):
    obs = sess.reset_to(0, wait_steps=10)
    q_base = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64).ravel()
    hold = [0.0] * 6
    for _ in range(OPEN_STEPS):
        obs, _, _, _ = sess.step(hold + [-1.0])
    q_minus = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64).ravel()
    obs = sess.reset_to(0, wait_steps=10)
    for _ in range(OPEN_STEPS):
        obs, _, _, _ = sess.step(hold + [+1.0])
    q_plus = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64).ravel()
    a_minus, a_plus = aperture(q_minus), aperture(q_plus)
    # ctrlrange 直接读模型（actuator 目标方向 vs 关节开口方向）
    sim = sess.env.env.sim
    names = [sim.model.actuator_id2name(i) for i in range(sim.model.nu)]
    gripper_ctrls = {}
    for i, n in enumerate(names):
        if n and ("gripper" in n.lower() or "finger" in n.lower()):
            lo, hi = sim.model.actuator_ctrlrange[i]
            gripper_ctrls[n] = [float(lo), float(hi)]
    verdict = "plus1_opens" if a_plus > a_minus else "minus1_opens"
    return {"gripper_qpos_init": q_base.tolist(),
            "gripper_qpos_after_minus1_x30": q_minus.tolist(),
            "gripper_qpos_after_plus1_x30": q_plus.tolist(),
            "aperture_after_minus1": round(a_minus, 6),
            "aperture_after_plus1": round(a_plus, 6),
            "gripper_actuator_ctrlranges": gripper_ctrls,
            "verdict": verdict}


def probe_pos_axis(sess, axis, delta):
    obs = sess.reset_to(0, wait_steps=10)
    p0, _ = eef_pose(obs)
    act = [0.0] * 7
    act[axis] = delta
    for _ in range(2):
        obs, _, _, _ = sess.step(act)
    p1, _ = eef_pose(obs)
    d = (p1 - p0)[axis]
    return {"delta_cmd_per_step": delta, "steps": 2,
            "eef_delta_axis%d" % axis: round(float(d), 5),
            "expected_sign": "positive" if delta > 0 else "negative"}


def probe_rot(sess, axis, delta):
    obs = sess.reset_to(0, wait_steps=10)
    _, quat0 = eef_pose(obs)
    act = [0.0] * 7
    act[3 + axis] = delta
    for _ in range(2):
        obs, _, _, _ = sess.step(act)
    _, quat1 = eef_pose(obs)
    return {"rotvec_cmd_per_step": delta, "steps": 2,
            "eef_quat_before": np.round(quat0, 5).tolist(),
            "eef_quat_after": np.round(quat1, 5).tolist()}


def main():
    sess = LiberoSession(suite="libero_goal", task_index=0)
    out["gripper"] = probe_gripper(sess)
    out["pos_x"] = probe_pos_axis(sess, 0, +0.1)
    out["pos_y"] = probe_pos_axis(sess, 1, +0.1)
    out["pos_z"] = probe_pos_axis(sess, 2, +0.1)
    out["rot_x"] = probe_rot(sess, 0, +0.2)
    json.dump(out, open(OUT, "w"), indent=1)
    print("PROBE_DONE verdict=%s" % out["gripper"]["verdict"], flush=True)


if __name__ == "__main__":
    main()
