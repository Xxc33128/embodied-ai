"""L4：LIBERO GPT 工具循环（observe→candidate→assess→execute→feedback→stop）。

L4 第一阶段：假传输走通全路径（无需 GPT 凭据）；真传输
（agent/client.py 的 connect/request/next_message）在联调阶段接入，
接口契约与假实现逐字段一致。

语义（复用单源实现，不得复制）：
- gate 真值：agent/gate.py::validate_assessment（upstream 8f3d362b 逐字）
  ——edit/eef 接管要求 observed failure 或 wrong intent；uncertainty 单独
  不构成接管理由；接管后恢复则交还。
- 观测白名单：agent/gate.py::build_observation_packet（真值/布局答案/
  未来轨迹不可见，主计划 W9）。
- 动作转换：libero/actions.py::direct_target_to_chunk / edit_to_chunk
  （单臂 F08；归一化 chunk；不可达拒绝）。
- 终局：env done → 原生成功；GPT stop → 提前终止 + 终态原生判据读数
  （stop 只结束采集，不产生成绩）；预算耗尽 → VALID_FAILURE。
"""
import time

import numpy as np

from gap_repro.agent.gate import (
    build_observation_packet,
    validate_assessment,
)
from gap_repro.libero.agent_contract import validate_response_envelope
from gap_repro.libero.actions import (
    direct_target_to_chunk,
    edit_to_chunk,
    quat_xyzw_to_wxyz,
)
from gap_repro.libero.policy import (
    REPLAN_STEPS,
    WAIT_STEPS,
    LiberoStudentPolicy,
    build_state,
)
from gap_repro.results import (
    INFRA_INCOMPLETE,
    VALID_FAILURE,
    VALID_SUCCESS,
    EpisodeResult,
)

ASSESS_MAX_ATTEMPTS = 2  # 首次 + 1 次同会话修正（计划 L4）


def _current_pose(obs):
    return {"position": np.asarray(obs["robot0_eef_pos"], dtype=np.float64),
            "quaternion_wxyz": quat_xyzw_to_wxyz(
                np.asarray(obs["robot0_eef_quat"], dtype=np.float64))}


def _last_gripper_closed(chunk_rows):
    if chunk_rows is None:
        return None
    return bool(chunk_rows[-1][6] > 0)  # 实测符号：+1 合


def run_hybrid_episode(case, *, policy: LiberoStudentPolicy, gpt_client,
                       session=None, session_factory=None,
                       audit_sink=None, wait_steps: int = WAIT_STEPS):
    """Hybrid 闭环：学生提案 + GPT 评估/修正/接管/交还 + 原生判据。

    gpt_client.assess(packet: dict, request_id: str) → response dict
    （response_schema 结构；gate.validate_assessment 校验接管正当性）。
    mode=stop 时提前终止，终局 success 以终态 is_success() 原生读数为准
    （stop 只结束采集；成绩永远来自环境原生判据）。
    """
    t0_wall = time.time()
    record = {"schema": "gap_repro.lp_runner_episode.v1",
              "method": "hybrid",
              "case": {"campaign_id": case.campaign_id,
                       "episode_id": case.episode_id,
                       "benchmark": case.benchmark,
                       "condition": case.condition,
                       "task_index": case.task_index,
                       "task_name": case.task_name,
                       "init_index": case.init_index,
                       "max_steps": case.max_steps,
                       "instruction": case.instruction},
              "cycles": [], "chunks": [], "success": None,
              "done_step": None, "stop_requested": False,
              "final_is_success_check": None, "infra_error": None}
    try:
        if session is None:
            session = (session_factory(case) if session_factory is not None
                       else _lazy_session(case))
        obs = session.reset_to(case.init_index, wait_steps=wait_steps)
        record["reset_fingerprint_state"] = session.fingerprint_state(obs)
        record["reset_fingerprint_full"] = session.fingerprint_full(obs)
        record["n_init_states"] = int(len(session.init_states))

        executed_gripper_closed = None
        t = 0
        done = False
        while t < case.max_steps and not record["stop_requested"]:
            cycle_idx = len(record["cycles"])
            request_id = f"{case.campaign_id}|{case.episode_id}|h{cycle_idx}"
            cycle = {"request_id": request_id, "t_start": t,
                     "fingerprint_state_before": session.fingerprint_state(obs)}
            proposal_chunk, proposal_identity = policy.infer_chunk(
                obs, case.instruction, request_id)
            packet = build_observation_packet(
                step_id=t,  # 控制步号（P1-1：wait 已在 reset_to 内）
                images={"agentview": np.ascontiguousarray(
                            obs["agentview_image"][::-1, ::-1]),
                        "wrist": np.ascontiguousarray(
                            obs["robot0_eye_in_hand_image"][::-1, ::-1])},
                current_eef=_current_pose(obs),
                current_state=build_state(obs),
                instruction=case.instruction,
                executed_steps=t,
                executed_gripper_closed=executed_gripper_closed,
                source_detail={"student_identity": proposal_identity},
                request_id=request_id,
                remaining_steps=case.max_steps - t,
            )
            # 评估 + 路由 + 转换的反馈-修正循环（计划 L4：格式错误返回同
            # 会话修正，不触发未校验动作；仍失败才升格 infra）
            chunk = None
            info = None
            feedback = None
            response = None
            trigger = None
            for attempt in range(ASSESS_MAX_ATTEMPTS):
                response = gpt_client.assess(packet, request_id,
                                             feedback=feedback)
                try:
                    validate_response_envelope(response, request_id)
                    trigger = validate_assessment(response, {"step_id": t})
                    if response["mode"] == "student":
                        chunk = proposal_chunk
                        origin = "student"
                    elif response["mode"] == "edit":
                        hold = None
                        if response["edit"].get("gripper") == "keep":
                            hold = float(proposal_chunk[-1][6])  # 学生末位夹爪
                        chunk, info = edit_to_chunk(
                            response["edit"], _current_pose(obs),
                            response["steps"], gripper_hold=hold)
                        origin = f"edit({trigger})"
                    elif response["mode"] == "eef":
                        chunk, info = direct_target_to_chunk(
                            response["target"], _current_pose(obs),
                            response["steps"])
                        origin = f"takeover({trigger})"
                    elif response["mode"] == "stop":
                        break
                    else:  # schema/gate 已挡；防御性分支
                        raise ValueError(
                            f"unroutable mode {response['mode']!r}")
                    break
                except (ValueError, KeyError, TypeError,
                        AttributeError) as e:
                    # upstream validation 语义：畸形响应一律转可修正输入错误
                    # （P1-3：此前裸 KeyError 逃逸成 infra）
                    feedback = (f"response rejected by gate/contract: {e}; "
                                f"re-issue for the same request_id per schema")
                    record.setdefault("assess_retries", []).append({
                        "request_id": request_id, "attempt": attempt,
                        "error": str(e)})
            else:
                raise RuntimeError(
                    f"GPT failed to correct response after "
                    f"{ASSESS_MAX_ATTEMPTS} attempts: {feedback}")
            cycle["gpt_mode"] = response["mode"]
            cycle["gate_trigger"] = trigger
            if response["mode"] == "stop":
                cycle["t_end"] = t
                record["cycles"].append(cycle)
                record["stop_requested"] = True
                break
            cycle["edit_info" if response["mode"] == "edit"
                  else "takeover_info" if response["mode"] == "eef"
                  else "_"] = info
            cycle["origin"] = origin
            for row in chunk:
                obs, _, done, _ = session.step(
                    np.asarray(row, dtype=np.float64))
                t += 1
                if done:
                    break
            executed_gripper_closed = _last_gripper_closed(chunk)
            cycle["t_end"] = t
            cycle["steps_executed"] = t - cycle["t_start"]
            record["cycles"].append(cycle)
            record["chunks"].append({
                "origin": origin, "t_start": cycle["t_start"],
                "t_end": t,
                "identity": {"student": proposal_identity,
                             "gpt_mode": response["mode"]}})
            if done:
                break
        record["final_is_success_check"] = bool(session.is_success())
        record["success"] = bool(done) or (
            record["stop_requested"] and record["final_is_success_check"])
        record["done_step"] = t if done else \
            (record["cycles"][-1]["t_end"] if record["cycles"] else 0)
        status = VALID_SUCCESS if record["success"] else VALID_FAILURE
        native = 1.0 if record["success"] else 0.0
    except Exception as e:
        record["infra_error"] = f"{type(e).__name__}: {e}"
        status, native = INFRA_INCOMPLETE, None
    record["wall_s"] = round(time.time() - t0_wall, 2)
    if audit_sink is not None:
        audit_sink(record)
    n_chunks = len(record.get("chunks", []))
    result = EpisodeResult(
        campaign_id=case.campaign_id, case_id=case.episode_id,
        method="hybrid", route="file_npz", status=status,
        native_score=native, attempt_id=0,
        entered_policy_stage=(n_chunks > 0),
        reason=(record.get("infra_error")
                or f"done_step={record.get('done_step')} "
                   f"stop={record.get('stop_requested')} "
                   f"cycles={len(record.get('cycles', []))} "
                   f"final_is_success_check={record.get('final_is_success_check')}"),
        init_fingerprint=record.get("reset_fingerprint_state"))
    return result, record


def _lazy_session(case):
    from gap_repro.sim.libero_session import LiberoSession
    return LiberoSession(suite=case.benchmark, task_index=case.task_index)


class FakeGptClient:
    """假传输（L4 第一阶段/单测）：按脚本回放响应序列。

    responses: [{"mode": "student"|"edit"|"eef"|"stop", ...}]，耗尽后
    恒返 student（默认通过），便于测试聚焦被检路径。
    """

    def __init__(self, responses=None, default="student"):
        self.responses = list(responses or [])
        self.default = default
        self.packets = []

    def assess(self, packet, request_id, feedback=None):
        self.packets.append((request_id, packet, feedback))
        if self.responses:
            return self.responses.pop(0)
        if self.default == "eef":
            return {"mode": "eef", "steps": 5, "reason": "direct eef move",
                    "target": {"position": [0.42, 0.38, 0.36],
                               "quaternion_wxyz": [0.0, 0.0, 1.0, 0.0],
                               "gripper_closed": False},
                    "assessment": _ok_assessment(packet["step_id"] == 0)}
        return {"mode": "student", "reason": "student continues",
                "assessment": _ok_assessment(packet["step_id"] == 0)}


def _ok_assessment(first_control_step: bool):
    """gate 合法的默认通过响应：step 0 必须 not_started，其后禁止
    （gate.validate_assessment 的 (step==0) != (exec==not_started) 语义）。"""
    return {"task_progress": {"verified_completed": [],
                              "currently_attempting": "approach handle",
                              "remaining": ["open drawer"]},
            "current_subgoal": "approach handle",
            "execution_status": "not_started" if first_control_step
                                else "progressing",
            "execution_evidence": "robot moving toward handle",
            "expected_next_intent": "grasp handle",
            "predicted_next_intent": "grasp handle",
            "intent_status": "aligned",
            "intent_evidence": "trajectory toward handle"}


def run_direct_episode(case, *, gpt_client, session=None,
                       session_factory=None, audit_sink=None,
                       wait_steps: int = WAIT_STEPS):
    """Direct（gpt_only）：GPT 直接 authoring eef 目标（1–5 步/决策），
    无学生提案、无学生推理。响应契约 = libero_direct_response_schema
    （mode 枚举 ["eef"]）；gate 语义与 Hybrid 共用（接管正当性同源）。
    终局语义与 run_hybrid_episode 一致（done-only；stop 读终态原生判据）。"""
    t0_wall = time.time()
    record = {"schema": "gap_repro.lp_runner_episode.v1",
              "method": "direct",
              "case": {"campaign_id": case.campaign_id,
                       "episode_id": case.episode_id,
                       "benchmark": case.benchmark,
                       "condition": case.condition,
                       "task_index": case.task_index,
                       "task_name": case.task_name,
                       "init_index": case.init_index,
                       "max_steps": case.max_steps,
                       "instruction": case.instruction},
              "cycles": [], "chunks": [], "success": None,
              "done_step": None, "stop_requested": False,
              "final_is_success_check": None, "infra_error": None}
    try:
        if session is None:
            session = (session_factory(case) if session_factory is not None
                       else _lazy_session(case))
        obs = session.reset_to(case.init_index, wait_steps=wait_steps)
        record["reset_fingerprint_state"] = session.fingerprint_state(obs)
        record["reset_fingerprint_full"] = session.fingerprint_full(obs)
        record["n_init_states"] = int(len(session.init_states))
        executed_gripper_closed = None
        t = 0
        done = False
        while t < case.max_steps and not record["stop_requested"]:
            cycle_idx = len(record["cycles"])
            request_id = f"{case.campaign_id}|{case.episode_id}|d{cycle_idx}"
            cycle = {"request_id": request_id, "t_start": t,
                     "fingerprint_state_before": session.fingerprint_state(obs)}
            packet = build_observation_packet(
                step_id=t,
                images={"agentview": np.ascontiguousarray(
                            obs["agentview_image"][::-1, ::-1]),
                        "wrist": np.ascontiguousarray(
                            obs["robot0_eye_in_hand_image"][::-1, ::-1])},
                current_eef=_current_pose(obs),
                current_state=build_state(obs),
                instruction=case.instruction,
                executed_steps=t,
                executed_gripper_closed=executed_gripper_closed,
                source_detail={},
                request_id=request_id,
                remaining_steps=case.max_steps - t,
            )
            chunk = None
            feedback = None
            response = None
            trigger = None
            for attempt in range(ASSESS_MAX_ATTEMPTS):
                response = gpt_client.assess(packet, request_id,
                                             feedback=feedback)
                try:
                    validate_response_envelope(response, request_id)
                    # gate_policy='gpt_only'（upstream validation.py 分岔）：
                    # Direct 无学生基线可比对意图，不套接管门；仍校验
                    # 信箱/模式/steps/target 界（_target_chunk）。
                    if response.get("mode") != "eef":
                        raise ValueError(
                            "direct route requires mode 'eef' "
                            f"(got {response.get('mode')!r})")
                    trigger = "gpt_only"
                    chunk, info = direct_target_to_chunk(
                        response["target"], _current_pose(obs),
                        response["steps"])
                    break
                except ValueError as e:  # 含 ContractError
                    feedback = (f"response rejected by gate/contract: {e}; "
                                f"re-issue for the same request_id per schema")
                    record.setdefault("assess_retries", []).append(
                        {"request_id": request_id, "attempt": attempt,
                         "error": str(e)})
            else:
                raise RuntimeError(
                    f"GPT failed to correct response after "
                    f"{ASSESS_MAX_ATTEMPTS} attempts: {feedback}")
            # upstream direct schema 的 mode 枚举仅 ["eef"]（无 stop）：
            # Direct 的集终局 = env done 或预算耗尽；其他 mode 一律在
            # 重试循环内被拒并反馈修正（不可达 stop 分支已移除）。
            cycle["gpt_mode"] = "eef"
            cycle["gate_trigger"] = trigger
            cycle["takeover_info"] = info
            for row in chunk:
                obs, _, done, _ = session.step(
                    np.asarray(row, dtype=np.float64))
                t += 1
                if done:
                    break
            executed_gripper_closed = _last_gripper_closed(chunk)
            cycle["t_end"] = t
            cycle["steps_executed"] = t - cycle["t_start"]
            record["cycles"].append(cycle)
            record["chunks"].append({"origin": "direct", "t_start": cycle["t_start"],
                                     "t_end": t, "identity": {"gpt_mode": "eef"}})
            if done:
                break
        record["final_is_success_check"] = bool(session.is_success())
        record["success"] = bool(done) or (
            record["stop_requested"] and record["final_is_success_check"])
        record["done_step"] = t if done else \
            (record["cycles"][-1]["t_end"] if record["cycles"] else 0)
        status = VALID_SUCCESS if record["success"] else VALID_FAILURE
        native = 1.0 if record["success"] else 0.0
    except Exception as e:
        record["infra_error"] = f"{type(e).__name__}: {e}"
        status, native = INFRA_INCOMPLETE, None
    record["wall_s"] = round(time.time() - t0_wall, 2)
    if audit_sink is not None:
        audit_sink(record)
    n_chunks = len(record.get("chunks", []))
    result = EpisodeResult(
        campaign_id=case.campaign_id, case_id=case.episode_id,
        method="direct", route="file_npz", status=status,
        native_score=native, attempt_id=0,
        entered_policy_stage=(n_chunks > 0),
        reason=(record.get("infra_error")
                or f"done_step={record.get('done_step')} "
                   f"stop={record.get('stop_requested')} "
                   f"cycles={len(record.get('cycles', []))}"),
        init_fingerprint=record.get("reset_fingerprint_state"))
    return result, record
