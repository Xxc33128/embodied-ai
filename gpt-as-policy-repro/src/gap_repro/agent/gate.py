"""LP3：takeover gate 真值函数（逐字移植 upstream gate_assessment.py::validate_assessment）
与观测白名单（工具可见范围，主计划 W9：agent 不可读评测真值/布局答案/未来轨迹）。
"""
from gap_repro.agent.contract import ContractError

EXECUTION_STATUSES = ("not_started", "progressing", "failed", "uncertain", "recovered")
INTENT_STATUSES = ("aligned", "misaligned", "uncertain")

# 观测白名单：只含当前 episode 的 RGB、本体、原指令与同集历史（E1，S28；主计划 W9）。
# request_id（P1-2 修正）：模型须在响应中回显该值（envelope 新鲜度检查），
# upstream packet 即含此键；remaining_steps（P2-6）：预算感知决策所需。
OBSERVATION_KEYS = frozenset({
    "step_id", "images", "current_eef", "current_state", "instruction",
    "executed_steps", "executed_gripper_closed", "source_detail",
    "request_id", "remaining_steps",
})


def validate_assessment(response, request):
    """gate 真值：edit/eef 接管要求 observed failure 或 wrong intent。

    逐字移植 upstream robodojo_server/gate_assessment.py::validate_assessment
    （commit 8f3d362b，L34-65），错误文本不改。
    request 需含 step_id；response 需含 mode/assessment。
    返回触发类别：'both' / 'execution_failure' / 'wrong_intent' / 'none'。
    """
    assessment = response.get("assessment")
    if not isinstance(assessment, dict):
        raise ValueError("Outcome and intent assessment required")
    progress = assessment.get("task_progress")
    if not isinstance(progress, dict):
        raise ValueError("Generic task_progress required")
    for key in ("verified_completed", "remaining"):
        if not isinstance(progress.get(key), list) or any(
                not isinstance(item, str) or not item.strip() for item in progress[key]):
            raise ValueError(f"task_progress.{key} must be a list of subgoal strings")
    if not isinstance(progress.get("currently_attempting"), str) or not progress["currently_attempting"].strip():
        raise ValueError("task_progress.currently_attempting required")
    for key in ("current_subgoal", "execution_evidence", "expected_next_intent",
                "predicted_next_intent", "intent_evidence"):
        if not isinstance(assessment.get(key), str) or not assessment[key].strip():
            raise ValueError(f"Assessment requires visible-evidence text: {key}")
    execution = assessment.get("execution_status")
    intent = assessment.get("intent_status")
    if execution not in EXECUTION_STATUSES:
        raise ValueError("Unknown execution assessment status")
    if intent not in INTENT_STATUSES:
        raise ValueError("Unknown intent assessment status")
    if (request["step_id"] == 0) != (execution == "not_started"):
        raise ValueError("not_started is required only before the first control step")
    failure, wrong_intent = execution == "failed", intent == "misaligned"
    if response["mode"] in ("edit", "eef"):
        if not (failure or wrong_intent):
            raise ValueError("Takeover requires observed failure or wrong task intent")
        return "both" if failure and wrong_intent else ("execution_failure" if failure else "wrong_intent")
    # A failed action does not mandate takeover if the student can self-recover.
    return "none"


def build_observation_packet(**fields):
    """构造工具可见的观测包：白名单外的键（真值/布局/未来信息）一律拒绝。"""
    extra = set(fields) - OBSERVATION_KEYS
    if extra:
        raise ContractError(f"observation packet rejects non-whitelisted keys: {sorted(extra)}")
    return dict(fields)
