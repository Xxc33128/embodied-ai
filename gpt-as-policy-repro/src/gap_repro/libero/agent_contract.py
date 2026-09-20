"""L2：LIBERO GPT 响应 schema（单臂）与工具表。

移植纪律（与 agent/contract.py 相同）：结构逐字段对齐 upstream
hybrid_rollout/robodojo/skill/schema.py @ 8f3d362b 的 response_schema /
direct_response_schema / tool_specs；机器人专属内容做显式域适配：
- edit/target 单臂化：无 left/right 包装（审计 F08：14D 双臂 schema
  不得下发到 LIBERO）；四元数/夹爪语义见 libero/actions.py。
- steps 上限：student ≤5（LIBERO replan=5，openpi E1；RoboDojo 学生
  前缀 15 不移植——记录为显式差异）；edit/eef ≤5（纸面语义不变）。
- 工具描述文本：robodojo_* → libero_*，"dual-arm"→单臂语义；其余措辞
  逐字保留（mode/gate 语义零漂移）。
"""
import json

from gap_repro.agent.contract import dumps  # 规范序列化单一实现源
from gap_repro.libero.actions import (
    LIBERO_CORRECTION_MAX_STEPS,
    LIBERO_STUDENT_CHUNK_STEPS,
)

LIBERO_UPSTREAM_COMMIT = "8f3d362b077d8efb77e2a7274d5b2c20e2243846"
MODES = ("student", "edit", "eef", "stop")


def libero_response_schema(request_id=None):
    """单臂 response_schema：结构对齐 upstream，edit/target 无双臂包装。"""
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
        steps={"type": "integer", "minimum": 1,
               "maximum": LIBERO_STUDENT_CHUNK_STEPS},
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
    # F08：单臂——无 left/right 包装（与 upstream 双臂版的唯一结构差异，
    # 由 test_no_dual_arm_wrapper 双向锁定）
    return schema


def libero_direct_response_schema():
    """单臂 direct（gpt_only Direct）：mode 枚举 ["eef"]，target 单臂。"""
    target = libero_response_schema()["properties"]["target"]
    return {"type": "object", "additionalProperties": False, "properties": dict(
        request_id={"type": "string"},
        mode={"type": "string", "enum": ["eef"]},
        steps={"type": "integer", "minimum": 1,
               "maximum": LIBERO_CORRECTION_MAX_STEPS},
        reason={"type": "string"}, target=target),
        "required": ["request_id", "mode", "steps", "reason", "target"]}


def validate_mode_steps(mode, steps):
    """LIBERO 步数协议界：student ≤5（replan）；edit/eef/stop ≤5（修正）。"""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    limit = (LIBERO_STUDENT_CHUNK_STEPS if mode == "student"
             else LIBERO_CORRECTION_MAX_STEPS)
    if steps > limit:
        raise ValueError(f"mode {mode!r} allows at most {limit} steps, got {steps}")


def libero_tool_specs(method="pi05_plus_gpt"):
    """工具表：结构对齐 upstream run.py::tool_specs，机器人词域适配。"""
    string = {"type": "string"}

    def spec(name, description, **properties):
        return dict(type="function", name=name, description=description,
                    inputSchema={"type": "object", "additionalProperties": False,
                                 "properties": properties,
                                 "required": list(properties)})

    if method == "gpt_only":
        return [spec("libero_start", "Blocking: start the requested LIBERO episode "
                     "and return RGB/proprio.", task=string, output_dir=string),
                spec("libero_act", "Blocking: execute GPT-authored bounded single-arm "
                     "EEF targets (1-5 steps), then return RGB/proprio and native "
                     "outcome. No pi05 inference, proposals or joint-action commands "
                     "exist.", observation_path=string,
                     response=libero_direct_response_schema(), output_dir=string)]
    return [
        spec("libero_start", "Blocking: start the requested LIBERO episode; save and "
             "return current RGB/proprio.", task=string, output_dir=string),
        spec("pi05_infer", "Blocking: infer pi05 once from the latest observation; "
             "save chunk and return full robot-only FK. "
             "action_diagnostics provides shape, finiteness, opening ranges and "
             "maximum successive joint step. "
             "Use these for routine arithmetic; native tools remain available for "
             "independent or additional checks. "
             "Diagnostics are not a safety or gate verdict: perform the unchanged "
             "visual and intent assessment.",
             observation_path=string, output_dir=string),
        spec("libero_execute", "Blocking: validate and execute a reviewed fresh chunk "
             "or short EEF correction; save and return new observation and terminal "
             "result.", proposal_path=string,
             response=libero_response_schema(), output_dir=string),
    ]


def validate_response_envelope(response, request_id):
    """upstream validation.py::validate_response 信箱检查（gate 无关段）：
    request_id 匹配 + 可见决策理由文本必填。gate_policy 分岔由调用方执行
    （hybrid → validate_assessment；gpt_only → 无接管门）。"""
    if response.get("request_id") not in (None, request_id):
        raise ValueError("Response is stale or belongs to another observation")
    reason = response.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("A visible observation/decision explanation is required")
