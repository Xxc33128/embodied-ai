"""LP3：GPT 路线契约常量与动作几何（模型无关部分）。

移植源（E1，钉 commit 8f3d362b077d8efb77e2a7274d5b2c20e2243846）：
- hybrid_rollout/robodojo/skill/schema.py（响应 schema）
- hybrid_rollout/robodojo/robodojo_server/gate_assessment.py（takeover 真值）
- 主计划 §7/§F 表（动作界、14 维顺序、夹爪映射）
坐标约定：environment origin、米、单位四元数 wxyz；夹爪学生值 0=闭/1=开。
"""
import json
import math

import numpy as np

UPSTREAM_COMMIT = "8f3d362b077d8efb77e2a7274d5b2c20e2243846"

# 主计划 §F（E1，S7、S27）：两层动作约束不得混为一次限幅
DECISION_MAX_TRANSLATION = 0.05   # 决策目标最大 5 cm
DECISION_MAX_ROTATION = 0.35      # 决策目标最大 0.35 rad
DLS_MAX_TRANSLATION = 0.02        # 每执行 ACK 的 DLS 局部推进最大 2 cm
DLS_MAX_ROTATION = 0.10           # DLS 局部最大 0.1 rad
DLS_MAX_PER_JOINT = 0.05          # 单关节最大 0.05 rad

# 主计划 §7（E1，S3–S8）：14 维动作顺序 [left_joint6, left_opening, right_joint6, right_opening]
ACTION_DIM = 14
STUDENT_CHUNK_MAX_STEPS = 15      # Hybrid 学生前缀 1–15
CORRECTION_MAX_STEPS = 5          # Direct / Hybrid 修正段 1–5
MODES = ("student", "edit", "eef", "stop")


class ContractError(ValueError):
    """契约违反（动作越界/模式非法/评估不完整）。"""


QUAT_UNIT_TOL = 1e-4  # 与上游 validation.py 的 |‖q‖−1|≤1e-4 闸一致（审查 A-P2-1）


def _check_finite(vec, name):
    a = np.asarray(vec, dtype=np.float64)
    if not np.all(np.isfinite(a)):
        raise ContractError(f"{name} contains non-finite values")
    return a


def _check_unit_quat(q, name):
    a = _check_finite(q, name)
    if a.shape != (4,):
        raise ContractError(f"{name} must have 4 elements (wxyz), got {a.shape}")
    if abs(float(np.linalg.norm(a)) - 1.0) > QUAT_UNIT_TOL:
        raise ContractError(f"{name} is not a unit quaternion (|‖q‖−1| > {QUAT_UNIT_TOL})")
    return a


def validate_target(target):
    """target 段结构契约：position(3 finite)/quaternion_wxyz(4 finite, 单位)/gripper_closed(bool)。"""
    for key in ("position", "quaternion_wxyz", "gripper_closed"):
        if key not in target:
            raise ContractError(f"target missing {key}")
    pos = _check_finite(target["position"], "target.position")
    if pos.shape != (3,):
        raise ContractError(f"target.position must have 3 elements, got {pos.shape}")
    _check_unit_quat(target["quaternion_wxyz"], "target.quaternion_wxyz")
    if not isinstance(target["gripper_closed"], bool):
        raise ContractError("target.gripper_closed must be boolean")


def quat_angle_distance(q1, q2):
    """单位四元数（wxyz）间的旋转角（rad）。"""
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    for q in (q1, q2):
        n = np.linalg.norm(q)
        if n == 0:
            raise ContractError("zero quaternion")
        q /= n
    dot = abs(float(np.dot(q1, q2)))  # q 与 -q 同旋转
    return 2.0 * math.acos(min(1.0, dot))


def validate_eef_target(target, current_eef):
    """Direct/eef 目标段契约：相对当前实测 link6 位姿的决策界（5 cm / 0.35 rad）。

    target/current_eef: {"position": [3], "quaternion_wxyz": [4]}。
    先做结构/有限性/单位四元数闸（与上游 validation.py 一致），再做界比较。
    """
    validate_target(target)
    _check_finite(current_eef["position"], "current_eef.position")
    _check_unit_quat(current_eef["quaternion_wxyz"], "current_eef.quaternion_wxyz")
    dp = np.linalg.norm(np.asarray(target["position"], dtype=np.float64)
                        - np.asarray(current_eef["position"], dtype=np.float64))
    if dp > DECISION_MAX_TRANSLATION + 1e-9:
        raise ContractError(f"decision target translation {dp:.4f} m > {DECISION_MAX_TRANSLATION}")
    dr = quat_angle_distance(target["quaternion_wxyz"], current_eef["quaternion_wxyz"])
    if dr > DECISION_MAX_ROTATION + 1e-9:
        raise ContractError(f"decision target rotation {dr:.4f} rad > {DECISION_MAX_ROTATION}")
    return {"translation_m": round(dp, 6), "rotation_rad": round(dr, 6)}


def rotation_vector_to_quat(w):
    """旋转向量（轴*角）→ 单位四元数 wxyz。"""
    w = np.asarray(w, dtype=np.float64)
    angle = float(np.linalg.norm(w))
    if angle == 0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = w / angle
    s = math.sin(angle / 2.0)
    return np.concatenate([[math.cos(angle / 2.0)], axis * s])


def quat_mul(a, b):
    """wxyz 四元数乘法 a*b。"""
    w1, x1, y1, z1 = np.asarray(a, dtype=np.float64)
    w2, x2, y2, z2 = np.asarray(b, dtype=np.float64)
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def apply_edit(current_pose, edit):
    """edit 段（delta_position + delta_rotation_vector + gripper）作用于当前位姿。

    返回新位姿 {"position", "quaternion_wxyz"}（gripper 由调用方按 keep/open/closed
    另行映射，见 EDIT_GRIPPER_TO_STUDENT）。有限性闸防 NaN/inf 传入执行器。
    """
    dpos = _check_finite(edit["delta_position"], "edit.delta_position")
    drot = _check_finite(edit["delta_rotation_vector"], "edit.delta_rotation_vector")
    cur_pos = _check_finite(current_pose["position"], "current.position")
    _check_unit_quat(current_pose["quaternion_wxyz"], "current.quaternion_wxyz")
    pos = cur_pos + dpos
    dq = rotation_vector_to_quat(drot)
    quat = quat_mul(dq, current_pose["quaternion_wxyz"])
    quat = quat / np.linalg.norm(quat)
    return {"position": pos, "quaternion_wxyz": quat}


def dls_clamp_joint_step(delta_joints):
    """执行边界（DLS 局部推进）限幅：单关节 ≤0.05 rad（主计划 §F）。

    输入/输出为关节增量向量；限幅是执行器的安全界，不替代决策界校验。
    """
    d = np.asarray(delta_joints, dtype=np.float64)
    over = np.abs(d) > DLS_MAX_PER_JOINT
    d = np.where(over, np.sign(d) * DLS_MAX_PER_JOINT, d)
    return d


# 主计划：夹爪学生值 0=闭/1=开；edit 段三值映射（schema: keep/open/closed）
EDIT_GRIPPER_TO_STUDENT = {"keep": None, "open": 1.0, "closed": 0.0}


def validate_mode_steps(mode, steps):
    """mode 与 steps 的协议界：student ≤15（Hybrid 前缀）；edit/eef ≤5（修正/Direct）。

    本地策略差异（审查 A-P3 登记）：stop 段本地同限 ≤5（上游不做此检查）。
    """
    if mode not in MODES:
        raise ContractError(f"unknown mode {mode!r}")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ContractError("steps must be a positive integer")
    limit = STUDENT_CHUNK_MAX_STEPS if mode == "student" else CORRECTION_MAX_STEPS
    if steps > limit:
        raise ContractError(f"mode {mode!r} allows at most {limit} steps, got {steps}")


def response_schema(request_id=None):
    """逐字移植 upstream skill/schema.py::response_schema（JSON-schema dict）。"""
    number = {"type": "number"}
    vector3 = {"type": "array", "items": number, "minItems": 3, "maxItems": 3}
    vector4 = {"type": "array", "items": number, "minItems": 4, "maxItems": 4}
    string = {"type": "string"}
    progress = {"type": "object", "additionalProperties": False, "properties": dict(
        verified_completed={"type": "array", "items": string},
        currently_attempting=string,
        remaining={"type": "array", "items": string},
    ), "required": ["verified_completed", "currently_attempting", "remaining"]}
    assessment = {"type": "object", "additionalProperties": False, "properties": dict(
        task_progress=progress,
        current_subgoal=string,
        execution_status={"type": "string", "enum": [
            "not_started", "progressing", "failed", "uncertain", "recovered"]},
        execution_evidence=string,
        expected_next_intent=string,
        predicted_next_intent=string,
        intent_status={"type": "string", "enum": ["aligned", "misaligned", "uncertain"]},
        intent_evidence=string,
    ), "required": ["task_progress", "current_subgoal", "execution_status",
                    "execution_evidence", "expected_next_intent",
                    "predicted_next_intent", "intent_status", "intent_evidence"]}
    identifier = string if request_id is None else {"type": "string", "enum": [request_id]}
    schema = {"type": "object", "additionalProperties": False, "properties": dict(
        request_id=identifier,
        mode={"type": "string", "enum": list(MODES)},
        steps={"type": "integer", "minimum": 1, "maximum": 15},
        reason=string,
        edit={"type": "object", "additionalProperties": False, "properties": dict(
            delta_position=vector3, delta_rotation_vector=vector3,
            gripper={"type": "string", "enum": ["keep", "open", "closed"]}),
            "required": ["delta_position", "delta_rotation_vector", "gripper"]},
        target={"type": "object", "additionalProperties": False, "properties": dict(
            position=vector3, quaternion_wxyz=vector4,
            gripper_closed={"type": "boolean"}),
            "required": ["position", "quaternion_wxyz", "gripper_closed"]},
        assessment=assessment,
    ), "required": ["request_id", "mode", "steps", "reason", "edit", "target",
                    "assessment"]}
    # 原实现：edit/target 各自包一层 left/right 双臂
    for field in ("edit", "target"):
        arm = schema["properties"][field]
        schema["properties"][field] = {"type": "object", "additionalProperties": False,
                                       "properties": dict(left=arm, right=arm),
                                       "required": ["left", "right"]}
    return schema


def direct_response_schema():
    """逐字移植 upstream skill/schema.py::direct_response_schema（gpt_only Direct）。"""
    target = response_schema()["properties"]["target"]
    return {"type": "object", "additionalProperties": False, "properties": dict(
        request_id={"type": "string"},
        mode={"type": "string", "enum": ["eef"]},
        steps={"type": "integer", "minimum": 1, "maximum": 5}, reason={"type": "string"},
        target=target),
        "required": ["request_id", "mode", "steps", "reason", "target"]}


def tool_specs(method="pi05_plus_gpt"):
    """逐字移植 upstream skill/run.py::tool_specs（工具表；描述文本不改一字）。"""
    string = {"type": "string"}

    def spec(name, description, **properties):
        return dict(type="function", name=name, description=description,
                    inputSchema={"type": "object", "additionalProperties": False,
                                 "properties": properties,
                                 "required": list(properties)})

    if method == "gpt_only":
        return [spec("robodojo_start", "Blocking: start the requested episode and return RGB/proprio.",
                     task=string, output_dir=string),
                spec("robodojo_act", "Blocking: execute GPT-authored bounded dual-arm EEF targets "
                     "(1-5 steps), then return RGB/proprio and native outcome. "
                     "No pi05 inference, proposals or direct joint-action commands exist.",
                     observation_path=string, response=direct_response_schema(), output_dir=string)]
    return [
        spec("robodojo_start", "Blocking: start the requested RoboDojo episode; save and return current RGB/proprio.",
             task=string, output_dir=string),
        spec("pi05_infer", "Blocking: infer pi05 once from the latest observation; save chunk and return full robot-only FK. "
             "action_diagnostics provides shape, finiteness, opening ranges and maximum successive joint step. "
             "Use these for routine arithmetic; native tools remain available for independent or additional checks. "
             "Diagnostics are not a safety or gate verdict: perform the unchanged visual and intent assessment.",
             observation_path=string, output_dir=string),
        spec("robodojo_execute", "Blocking: validate and execute a reviewed fresh chunk or short EEF correction; save and return new observation and terminal result.",
             proposal_path=string, response=response_schema(), output_dir=string),
    ]


GATE_PROMPT = """Use the environment's task instruction as the objective.
Derive ordered subgoals and prerequisites from that instruction; do not assume
any particular object, destination or task family. Maintain task_progress:
verified_completed (list), currently_attempting (string), remaining (list).
Update completion only from visual execution evidence; undo a completed
subgoal if later observations show it has been lost (e.g. a structure collapses).
Keep the original task instruction as the student's input.
At each chunk boundary, assess two separate questions:
1. What happened during the LAST executed chunk? Compare before/after RGB,
   measured robot state, executed gripper commands and same-episode history.
   A closed command alone proves neither grasp success nor failure. Look for
   object motion during lifting, slipping, missed placement or sustained lack
   of progress. Distinguish pending/uncertain results from an observed failure.
2. Given the current task phase, does the NEXT student chunk pursue an
   appropriate subgoal? Infer intent from its robot-only FK trajectory and
   gripper sequence; do not claim access to the student's internal intention.
   Advancing to a dependent subgoal after its prerequisite failed is wrong.
   Also check selected object, destination, required order and grasp/release phase.
   Realigning for a retry can be appropriate: allow student self-recovery.
Return a concise assessment with task_progress, current_subgoal, execution_status,
execution_evidence, expected_next_intent, predicted_next_intent, intent_status,
and intent_evidence. Statuses: execution not_started/progressing/failed/
uncertain/recovered; intent aligned/misaligned/uncertain.
An edit/eef takeover requires execution_status=failed or intent_status=misaligned.
Uncertainty alone or an aesthetically imperfect pose is not a takeover reason.
After recovery, hand back when the current state and student subgoal are suitable.
Do not rewind. Do not use object truth, reward or future simulated object states.
"""


def dumps(obj):
    """schema 的规范序列化（审计与测试对账用）。"""
    return json.dumps(obj, sort_keys=True)
