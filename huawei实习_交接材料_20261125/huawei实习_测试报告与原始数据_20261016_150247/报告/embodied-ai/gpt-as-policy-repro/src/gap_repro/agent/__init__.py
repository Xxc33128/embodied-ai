"""LP3：GPT 路线契约层（模型无关部分）。

- contract.py：schema/工具表/gate prompt/动作几何（逐字移植，钉上游 commit）
- gate.py：takeover 真值 + 观测白名单
- client.py：路线 A/B 传输接口（接入信息未到位前 connect() 给出缺失清单）
接入信息到位后仅需提供 transport 配置，契约层不再改动。
"""
from gap_repro.agent.contract import (  # noqa: F401
    UPSTREAM_COMMIT, GATE_PROMPT, MODES, ACTION_DIM,
    DECISION_MAX_TRANSLATION, DECISION_MAX_ROTATION,
    DLS_MAX_TRANSLATION, DLS_MAX_ROTATION, DLS_MAX_PER_JOINT,
    STUDENT_CHUNK_MAX_STEPS, CORRECTION_MAX_STEPS,
    ContractError, response_schema, direct_response_schema, tool_specs,
    validate_eef_target, validate_target, validate_mode_steps, apply_edit,
    dls_clamp_joint_step, EDIT_GRIPPER_TO_STUDENT, quat_angle_distance,
)
from gap_repro.agent.gate import (  # noqa: F401
    validate_assessment, build_observation_packet, OBSERVATION_KEYS,
)
from gap_repro.agent.client import (  # noqa: F401
    Transport, AppServerTransport, ApiGatewayTransport, ConfigMissingError,
    dispatch_tool_call,
)
