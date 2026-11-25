"""L2：LIBERO 单臂 7D 动作契约（审计 F08：RoboDojo 14D 双臂 schema 不得下发）。

锚定（全部钉定实测/源码，非推断）：
- 动作向量 [Δpos(3), Δrot-vec(3), gripper(1)]：openpi examples/libero/main.py
  （client.infer → env.step 原样下发，无二值化/无裁剪）。
- 控制器缩放（P0-1 决定性探针 l2_action_probe_v2 实测，2026-09-20）：
  配置 output_max=[0.05,0.05,0.05,0.5,0.5,0.5] 是设定点上限，但设定点
  逐步被安全钳位：满幅 30 步实测 EEF ~10.4–11.4 mm/步（设定点同量级），
  旋转满幅 20 步共 96.9°（4.85°/步）。钉定 POS_ACTION_SCALE=0.011、
  ROT_ACTION_SCALE=0.085（实测满幅传递；小幅度有滞后非线性，见 v1 探针
  l2-action-probe.json）。学生实测最大 |value|=0.97（lp-a2-episodes）
  → ~10.7 mm/步，物理合理。
  旋转向量在世界系解释（probe v3 决定性判别：命令 x 轴 → 响应轴世界系
  x 纯度 0.9975 vs 体系 0.674；v2 的 body-frame 判别系四元数公式错误
  + 子探针自污染，已作废）。direct_target_to_chunk 的世界系 geodesic
  等分与该裁决一致。传递非线性：0.5 幅度实测 0.112 rad/单位（满幅
  0.085）——决策界按满幅口径钉定，小幅度实现偏差由每决策闭环重测吸收。
  v1 探针的"2 步 1.3mm"是滞后未收敛段，不能直接反推斜率——P0-1 审查
  正确指出该矛盾，v2 满幅长序列消解。
- 夹爪符号（scripts/l2_action_probe.py 实测证据
  /workspace/data/l2_action_probe.json）：**−1 = 开，+1 = 合**
  （robosuite 1.4.0 PandaGripper：开口度 0.0794 after −1×30 vs 0.0012
  after +1×30；ctrlrange finger_joint1 [0, 0.04] 开口方向）。
  学生实测输出可略超界（−1.003），学生路径不裁剪（客户端一致性，
  robosuite 内部 clip）；工具路径严格 |value| ≤ 1。
- 观测约定：robot0_eef_quat 为 robosuite (x,y,z,w)；本模块内部统一 wxyz
  （与 agent/contract.py 一致），边界处显式转换。
- DUMMY_ACTION = [0]*6 + [−1]（= 张开；openpi 客户端逐字一致，probe 佐证
  settle 阶段张爪合理）。

与 agent/contract.py 的关系：四元数数学与 target 结构门单一实现源（直接复用）；
决策限界、步数限、缩放为 LIBERO 专属常量，不继承 RoboDojo 的
DECISION_MAX_*(0.05 m/0.35 rad，为双臂 DLS 机器人而设)。
"""
import math

import numpy as np

from gap_repro.agent.contract import (
    ContractError,
    _check_finite,
    _check_unit_quat,
    quat_mul,
    rotation_vector_to_quat,
    validate_target,
)

LIBERO_ACTION_DIM = 7
GRIPPER_OPEN = -1.0    # 实测：−1×30 步开口度 0.0794（probe verdict=minus1_opens）
GRIPPER_CLOSE = +1.0   # 实测：+1×30 步开口度 0.0012
LIBERO_DUMMY_ACTION = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, GRIPPER_OPEN)

POS_ACTION_SCALE = 0.011   # m / 单位 action（probe v2 满幅实测 10.4–11.4 mm/步）
ROT_ACTION_SCALE = 0.085   # rad / 单位 action（probe v2 满幅 4.85°/步）

# 预注册（L2；L5 冻结复核）：Direct/修正每决策（≤5 步）相对当前位姿的界。
# 上界 = 5 步实测可达（5×0.011 m ≈ 0.055 m；5×0.085 ≈ 0.425 rad）。
# 注意：这是设定点/实现混合的实测口径；纸面 5cm/0.35rad 是 RoboDojo
# 机器人的决策界，不移植——记录为显式差异。
LIBERO_DECISION_MAX_TRANSLATION = 0.055  # m
LIBERO_DECISION_MAX_ROTATION = 0.425     # rad

LIBERO_STUDENT_CHUNK_STEPS = 5    # openpi replan_steps=5（E1）
LIBERO_CORRECTION_MAX_STEPS = 5   # Direct/修正（纸面语义不变）
TOOL_VALUE_BOUND = 1.0            # 工具路径归一化值硬界（学生路径不受此限）


def gripper_closed_to_action(gripper_closed: bool) -> float:
    """工具语义 gripper_closed(bool) → 归一化动作值。实测符号：True→+1（合）。"""
    return GRIPPER_CLOSE if gripper_closed else GRIPPER_OPEN


def validate_student_chunk(chunk) -> np.ndarray:
    """学生 chunk 结构门（有限性/形状），不裁剪不设值域（客户端一致性）。

    服务器响应为全程 chunk（10,7）；本门只做结构校验。replan 执行预算
    (≤5) 由调用方切片后另行保证（policy.infer_chunk / runner），不在此
    拒收全程 chunk。实测学生输出可略超 [−1,1]（−1.003）；robosuite 控制
    器内部 clip，与 openpi 客户端行为逐字一致。
    """
    a = np.asarray(chunk, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != LIBERO_ACTION_DIM:
        raise ContractError(
            f"student chunk must be (n, {LIBERO_ACTION_DIM}), got {a.shape}")
    if a.shape[0] < 1:
        raise ContractError("student chunk must have at least 1 step")
    if not np.all(np.isfinite(a)):
        raise ContractError("student chunk contains non-finite values")
    return a


def validate_tool_action(action) -> np.ndarray:
    """工具路径单步门（L4：格式错误返回同会话修正，不触发未校验动作）。"""
    a = _check_finite(action, "action")
    if a.shape != (LIBERO_ACTION_DIM,):
        raise ContractError(
            f"action must be ({LIBERO_ACTION_DIM},), got {a.shape}")
    if np.any(np.abs(a) > TOOL_VALUE_BOUND):
        raise ContractError(
            f"tool action values must be within ±{TOOL_VALUE_BOUND} "
            f"(normalized controller input), got {a.tolist()}")
    return a


def quat_to_rotation_vector(q_wxyz) -> np.ndarray:
    """单位四元数（wxyz）→ 旋转向量（轴*角，rad）。log 映射（目标几何用）。"""
    q = _check_unit_quat(q_wxyz, "quat")
    q = q / np.linalg.norm(q)
    if q[0] < 0:  # 半球约定：w ≥ 0，|rotvec| ≤ π
        q = -q
    w = min(1.0, max(-1.0, float(q[0])))
    angle = 2.0 * math.acos(w)
    s = math.sqrt(max(0.0, 1.0 - w * w))
    if s < 1e-12:
        return np.zeros(3)
    axis = q[1:4] / s
    return axis * angle


def axisangle_from_xyzw(quat_xyzw) -> np.ndarray:
    """state 编码专用：openpi main.py::_quat2axisangle 逐字移植
    （= robosuite transform_utils quat2axisangle）。

    输入为 robosuite 观测约定 (x,y,z,w)——注意 w 在 index 3；无半球翻转、
    无归一化：w<0 时角度沿原 axis 超过 π。state 向量分布必须与训练
    预处理逐字一致（P2-7 修正：此前自创 log-map 版与 openpi 在 w<0
    时差 2π）。"""
    import numpy as _np
    q = _np.asarray(quat_xyzw, dtype=_np.float64).ravel()
    if q.shape != (4,):
        raise ContractError(f"quat_xyzw must have 4 elements, got {q.shape}")
    if not _np.all(_np.isfinite(q)):
        raise ContractError("quat contains non-finite values")
    w = min(1.0, max(-1.0, float(q[3])))
    den = math.sqrt(1.0 - w * w)
    if math.isclose(den, 0.0):
        return _np.zeros(3)
    return (q[:3] * 2.0 * math.acos(w)) / den


def quat_xyzw_to_wxyz(q) -> np.ndarray:
    """robosuite 观测 (x,y,z,w) → 内部 wxyz。边界转换唯一入口。"""
    a = _check_finite(q, "eef_quat")
    if a.shape != (4,):
        raise ContractError(f"eef_quat must have 4 elements, got {a.shape}")
    return a[[3, 0, 1, 2]]


def pose_to_policy_state(eef_pos, eef_quat_xyzw, gripper_qpos) -> np.ndarray:
    """policy state 向量（openpi 客户端 concat 顺序）：
    [eef_pos(3, m), axisangle(eef_quat)(3, rad), gripper_qpos(2, raw)] = 8 维。

    LP-A2 契约钉定（lp_a2_closed_loop.py assert (8,)，6/6 成功集）：Panda
    双指 mirrored qpos (2,) 原样 concat；四元数直接收 robosuite 观测约定
    (x,y,z,w) 并经 axisangle_from_xyzw（openpi 逐字）编码。
    norm stats 在模型边界处理，此处不缩放。
    """
    pos = _check_finite(eef_pos, "eef_pos")
    if pos.shape != (3,):
        raise ContractError(f"eef_pos must have 3 elements, got {pos.shape}")
    rot = axisangle_from_xyzw(eef_quat_xyzw)
    g = np.asarray(gripper_qpos, dtype=np.float64)
    if g.size not in (1, 2):
        raise ContractError(
            f"gripper_qpos must be Panda dual-finger (2,) or scalar, "
            f"got shape {g.shape}")
    return np.concatenate([pos, rot, g.ravel()])


def _target_chunk(delta_pos_m, delta_rot_rad, steps, label):
    """目标/修正增量 → 归一化 7D chunk（每步等分，几何精确到达）。

    旋转沿同一轴等分角（geodesic），位置线性等分；归一化值
    = 物理增量 / (steps × scale)，超 ±1 即目标在 steps 内不可达 → 契约错误
    （L4：同会话把错误反馈给模型修正，不得静默截断）。
    """
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ContractError("steps must be a positive integer")
    if steps > LIBERO_CORRECTION_MAX_STEPS:
        raise ContractError(
            f"correction allows at most {LIBERO_CORRECTION_MAX_STEPS} steps, "
            f"got {steps}")
    dp = _check_finite(delta_pos_m, f"{label}.delta_position")
    if dp.shape != (3,):
        raise ContractError(f"{label}.delta_position must have 3 elements")
    dr = _check_finite(delta_rot_rad, f"{label}.delta_rotation")
    if dr.shape != (3,):
        raise ContractError(f"{label}.delta_rotation must have 3 elements")
    dpm = float(np.linalg.norm(dp))
    drm = float(np.linalg.norm(dr))
    if dpm > LIBERO_DECISION_MAX_TRANSLATION + 1e-9:
        raise ContractError(
            f"{label} translation {dpm:.4f} m > "
            f"{LIBERO_DECISION_MAX_TRANSLATION}")
    if drm > LIBERO_DECISION_MAX_ROTATION + 1e-9:
        raise ContractError(
            f"{label} rotation {drm:.4f} rad > "
            f"{LIBERO_DECISION_MAX_ROTATION}")
    pos_norm = dp / (steps * POS_ACTION_SCALE)
    rot_norm = dr / (steps * ROT_ACTION_SCALE)
    if np.any(np.abs(pos_norm) > TOOL_VALUE_BOUND) or \
            np.any(np.abs(rot_norm) > TOOL_VALUE_BOUND):
        raise ContractError(
            f"{label} unreachable in {steps} steps (per-step normalized "
            f"value exceeds ±1)")
    return pos_norm, rot_norm


def direct_target_to_chunk(target, current_pose, steps):
    """Direct（eef 模式）目标 → 归一化 chunk。

    target: {"position": [3] m, "quaternion_wxyz": [4], "gripper_closed": bool}
    ——与 RoboDojo 工具结构逐字段一致（跨域 GPT 接口统一），但单臂、无
    left/right 包装（F08）。current_pose 同构（无 gripper 段）。
    返回 (chunk (steps,7), info dict)。
    """
    validate_target(target)
    _check_finite(current_pose["position"], "current.position")
    _check_unit_quat(current_pose["quaternion_wxyz"], "current.quaternion_wxyz")
    cur_q = np.asarray(current_pose["quaternion_wxyz"], dtype=np.float64)
    cur_q = cur_q / np.linalg.norm(cur_q)
    tgt_q = np.asarray(target["quaternion_wxyz"], dtype=np.float64)
    tgt_q = tgt_q / np.linalg.norm(tgt_q)
    dpos = (np.asarray(target["position"], dtype=np.float64)
            - np.asarray(current_pose["position"], dtype=np.float64))
    # 世界系 geodesic：dq = tgt ∘ cur⁻¹ → rotvec；沿同一轴等分角。
    q_cur_inv = np.array([cur_q[0], -cur_q[1], -cur_q[2], -cur_q[3]])
    dq = quat_mul(tgt_q, q_cur_inv)
    if dq[0] < 0:
        dq = -dq
    drot = quat_to_rotation_vector(dq / np.linalg.norm(dq))
    pos_norm, rot_norm = _target_chunk(dpos, drot, steps, "target")
    g = gripper_closed_to_action(target["gripper_closed"])
    chunk = np.tile(np.concatenate([pos_norm, rot_norm, [g]]), (steps, 1))
    info = {"steps": steps,
            "translation_m": round(float(np.linalg.norm(dpos)), 6),
            "rotation_rad": round(float(np.linalg.norm(drot)), 6),
            "gripper_action": g}
    return chunk, info


def edit_to_chunk(edit, current_pose, steps, gripper_hold=None):
    """Hybrid 修正段（delta_position/delta_rotation_vector/gripper 三值）
    → 归一化 chunk。schema 与 RoboDojo 逐字段一致（米/rad），但语义为
    叠加于当前实测位姿的增量（upstream 是编辑学生轨迹—— LIBERO 侧
    显式差异，闭式 chunk 不可编辑学生轨迹）。gripper="keep"（P1-4 修正）：
    保持学生夹爪值，
    由调用方经 gripper_hold 传入（学生提案 chunk 的末位夹爪或当前实测
    夹爪符号）；未传则拒绝（不静默猜值）。"""
    if edit.get("gripper") not in ("open", "closed", "keep"):
        raise ContractError(
            f"edit.gripper must be open/closed/keep, got {edit.get('gripper')!r}")
    if edit["gripper"] == "keep" and gripper_hold is None:
        raise ContractError(
            "edit.gripper='keep' requires gripper_hold "
            "(student's last gripper value) from the caller")
    dpos = _check_finite(edit["delta_position"], "edit.delta_position")
    drot = _check_finite(edit["delta_rotation_vector"],
                         "edit.delta_rotation_vector")
    if dpos.shape != (3,) or drot.shape != (3,):
        raise ContractError("edit deltas must be 3-vectors")
    pos_norm, rot_norm = _target_chunk(dpos, drot, steps, "edit")
    if edit["gripper"] == "keep":
        g = float(gripper_hold)
    else:
        g = gripper_closed_to_action(edit["gripper"] == "closed")
    chunk = np.tile(np.concatenate([pos_norm, rot_norm, [g]]), (steps, 1))
    info = {"steps": steps,
            "translation_m": round(float(np.linalg.norm(dpos)), 6),
            "rotation_rad": round(float(np.linalg.norm(drot)), 6),
            "gripper_action": g}
    return chunk, info
