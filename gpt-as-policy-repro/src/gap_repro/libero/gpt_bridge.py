"""L4：模型驱动 episode 桥（upstream run.py::Rollout/CodexPolicy 架构的
LIBERO 移植——整集一个 turn，模型经 libero_* 工具驱动，host 执行并 reply）。

与同步评估循环（agent_loop.py）的关系：agent_loop 用于管线测试与假传输
验收；本模块是正式采集的模型驱动协议（论文口径）。工具面 =
libero_tool_specs(method)：
- libero_start（两方法）：建 session + reset(init) → 观测包
- libero_act（direct）：GPT 的 eef 响应 → envelope/mode/target 界校验 →
  执行 → 新观测/终局
- pi05_infer（hybrid）：学生 chunk 推理（候选身份）
- libero_execute（hybrid）：GPT 响应 → envelope + gate（failure-or-intent）
  → student/edit/eef/stop 路由执行

输入错误协议（upstream InputError 语义）：可修正的输入错误不终止 turn——
拒绝 + 计数 + 在 reply 内容中反馈；host 不重复执行被拒调用。
观测包经 build_observation_packet 白名单（真值/答案/未来不可见）。
"""
import base64
import io
import json

import numpy as np

from gap_repro.agent.client import dispatch_tool_call
from gap_repro.agent.contract import GATE_PROMPT
from gap_repro.agent.gate import (
    build_observation_packet,
    validate_assessment,
)
from gap_repro.libero.actions import (
    LIBERO_DUMMY_ACTION,
    direct_target_to_chunk,
    edit_to_chunk,
    quat_xyzw_to_wxyz,
)
from gap_repro.libero.agent_contract import (
    libero_tool_specs,
    validate_response_envelope,
)
from gap_repro.libero.policy import (
    WAIT_STEPS,
    LiberoStudentPolicy,
    build_state,
)
from gap_repro.results import VALID_FAILURE, VALID_SUCCESS


class InputError(ValueError):
    """可修正的模型输入错误（upstream 同名语义）：拒绝但 turn 继续。"""


def _png_base64(img: np.ndarray) -> str:
    import imageio.v2 as imageio
    buf = io.BytesIO()
    imageio.imwrite(buf, np.ascontiguousarray(img), format="png")
    return base64.b64encode(buf.getvalue()).decode()


def _observation_packet(obs, case, step_id, executed_gripper_closed,
                        student_identity=None, request_id=None):
    return build_observation_packet(
        step_id=step_id,
        images={"agentview": np.ascontiguousarray(
                    obs["agentview_image"][::-1, ::-1]),
                "wrist": np.ascontiguousarray(
                    obs["robot0_eye_in_hand_image"][::-1, ::-1])},
        current_eef={"position": np.asarray(obs["robot0_eef_pos"],
                                            dtype=np.float64),
                     "quaternion_wxyz": quat_xyzw_to_wxyz(
                         np.asarray(obs["robot0_eef_quat"], dtype=np.float64))},
        current_state=build_state(obs),
        instruction=case.instruction,
        executed_steps=step_id,
        executed_gripper_closed=executed_gripper_closed,
        source_detail={"student_identity": student_identity},
        request_id=request_id,
        remaining_steps=case.max_steps - step_id,
    )


def _packet_with_images(packet):
    """工具结果内容项（upstream run.py::content_items 形状）：
    inputText + inputImage（data-URI）。联调时若 app-server 要求别的
    包络字段，在此一处修改。"""
    items = [{"type": "inputText",
              "text": json.dumps(
                  {k: v for k, v in packet.items() if k != "images"},
                  default=str, ensure_ascii=False)}]
    for name, img in packet.get("images", {}).items():
        items.append({"type": "inputImage",
                      "imageUrl": "data:image/png;base64,"
                                  + _png_base64(img)})
    return items


class LiberoRollout:
    """Host 侧单集状态机。"""

    def __init__(self, case, *, policy=None, session=None,
                 session_factory=None):
        self.case = case
        self.policy = policy
        self.session = session
        self.session_factory = session_factory
        self.step_id = 0            # 控制步（不含 wait）
        self.done = False
        self.success = None
        self.rejected_calls = 0
        self.stop_requested = False
        self.executed_gripper_closed = None
        self._action_plan = []
        self._active_identity = None
        self.record = {"schema": "gap_repro.lp_runner_episode.v1",
                       "method": None, "case": vars(case) if hasattr(
                           case, "__dict__") else case,
                       "chunks": [], "cycles": [], "assess_retries": []}

    # ---- 工具 handler ----
    def libero_start(self, task, output_dir=None):
        if task and task != self.case.instruction:
            raise InputError(
                f"task must match the episode instruction "
                f"{self.case.instruction!r}")
        if self.session is None:
            self.session = (self.session_factory(self.case)
                            if self.session_factory else _lazy(self.case))
        obs = self.session.reset_to(self.case.init_index,
                                    wait_steps=WAIT_STEPS)
        self.record["reset_fingerprint_state"] = \
            self.session.fingerprint_state(obs)
        self.record["n_init_states"] = int(len(self.session.init_states))
        self._obs = obs
        # P1-新2：模型可见面必须含下一响应应回显的 request_id
        # （前缀由 run_rollout 按方法注入：hybrid='h'，direct='d'）
        rid = f"{self.case.campaign_id}|{self.case.episode_id}|" \
              f"{getattr(self, '_rid_prefix', 'h')}0"
        return _observation_packet(obs, self.case, 0, None,
                                   request_id=rid)

    def _require_started(self):
        if not hasattr(self, "_obs"):
            raise InputError("libero_start must succeed before this tool")

    def pi05_infer(self, observation_path=None, output_dir=None):
        if self.policy is None:
            raise InputError("pi05_infer requires the hybrid method")
        self._require_started()
        chunk, identity = self.policy.infer_chunk(
            self._obs, self.case.instruction,
            f"{self.case.campaign_id}|{self.case.episode_id}|r{self.step_id}")
        self._pending_proposal = chunk
        self._active_identity = identity
        return {"chunk_sha256": identity["actions_sha256"],
                "full_steps": identity["full_steps"],
                "replan_steps": identity["replan_steps"],
                "note": "proposal stored host-side; call libero_execute "
                        "with your gate response"}

    def libero_execute(self, response=None, proposal_path=None,
                       output_dir=None):
        if response is None:
            raise InputError("response (libero response schema) required")
        self._require_started()
        if not hasattr(self, "_pending_proposal"):
            raise InputError("call pi05_infer before libero_execute")
        request_id = f"{self.case.campaign_id}|{self.case.episode_id}|" \
                     f"h{self.step_id}"
        try:
            validate_response_envelope(response, request_id)
            trigger = validate_assessment(response,
                                          {"step_id": self.step_id})
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            # P1-3：畸形响应（缺 mode 等）转可修正输入错误，不逃逸成 infra
            self.rejected_calls += 1
            raise InputError(str(e)) from e
        mode = response["mode"]
        if mode == "student":
            chunk = self._pending_proposal
            info = None
        elif mode == "edit":
            try:
                hold = None
                if response["edit"].get("gripper") == "keep":
                    hold = (float(self._pending_proposal[-1][6])
                            if hasattr(self, "_pending_proposal")
                            else self._gripper_action_now())
                chunk, info = edit_to_chunk(response["edit"],
                                            _pose(self._obs),
                                            response["steps"],
                                            gripper_hold=hold)
            except (ValueError, KeyError, TypeError, AttributeError) as e:
                self.rejected_calls += 1
                raise InputError(f"edit rejected: {e}") from e
        elif mode == "eef":
            try:
                chunk, info = direct_target_to_chunk(response["target"],
                                                     _pose(self._obs),
                                                     response["steps"])
            except (ValueError, KeyError, TypeError, AttributeError) as e:
                self.rejected_calls += 1
                raise InputError(f"eef target rejected: {e}") from e
        elif mode == "stop":
            self.stop_requested = True
            return self._terminal_packet()
        else:
            self.rejected_calls += 1
            raise InputError(f"unroutable mode {mode!r}")
        packet = self._execute_chunk(chunk, origin=mode, info=info)
        return packet

    def libero_act(self, observation_path=None, response=None,
                   output_dir=None):
        """direct：GPT 每决策 authoring 一个 eef 响应。"""
        if response is None:
            raise InputError("response (libero direct schema) required")
        self._require_started()
        request_id = f"{self.case.campaign_id}|{self.case.episode_id}|" \
                     f"d{self.step_id}"
        try:
            validate_response_envelope(response, request_id)
            if response.get("mode") != "eef":
                raise ValueError(
                    f"direct route requires mode 'eef' "
                    f"(got {response.get('mode')!r})")
            chunk, info = direct_target_to_chunk(response["target"],
                                                 _pose(self._obs),
                                                 response["steps"])
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            self.rejected_calls += 1
            raise InputError(str(e)) from e
        return self._execute_chunk(chunk, origin="direct", info=info)

    # ---- 执行与终局 ----
    def _gripper_action_now(self):
        """当前实测夹爪的归一化动作符号（keep 的回退 hold 值）。"""
        q = np.asarray(self._obs["robot0_gripper_qpos"],
                       dtype=np.float64).ravel()
        return 1.0 if float(q[0] - q[1]) < 0.01 else -1.0  # 开口小→合

    def _execute_chunk(self, chunk, origin, info=None):
        fp_before = self.session.fingerprint_state(self._obs)
        t_start = self.step_id
        next_rid = f"{self.case.campaign_id}|{self.case.episode_id}|" \
                   f"{getattr(self, '_rid_prefix', 'h')}{self.step_id}"
        for row in chunk:
            self._obs, _, done, _ = self.session.step(
                np.asarray(row, dtype=np.float64))
            self.step_id += 1
            if done:
                break
        self.executed_gripper_closed = bool(chunk[-1][6] > 0)
        # P2-8：候选身份入记录（审计链：学生 chunk sha256 / edit+takeover info）
        self.record["chunks"].append({
            "origin": origin, "t_start": t_start, "t_end": self.step_id,
            "fingerprint_state_before": fp_before,
            "student_identity": getattr(self, "_active_identity", None),
            "executed_info": info})
        if done or self.step_id >= self.case.max_steps:
            return self._terminal_packet(done=done)
        return _observation_packet(self._obs, self.case, self.step_id,
                                   self.executed_gripper_closed,
                                   request_id=next_rid)

    def _terminal_packet(self, done=None):
        self.done = True
        self.success = bool(done) if done is not None \
            else bool(self.session.is_success())
        if done is None and self.stop_requested:
            self.success = bool(self.session.is_success())
        self.record["final_is_success_check"] = \
            bool(self.session.is_success())
        self.record["success"] = self.success
        self.record["done_step"] = self.step_id
        return {"terminal": True, "success": self.success,
                "done_step": self.step_id,
                "instruction": self.case.instruction}


def _pose(obs):
    return {"position": np.asarray(obs["robot0_eef_pos"], dtype=np.float64),
            "quaternion_wxyz": quat_xyzw_to_wxyz(
                np.asarray(obs["robot0_eef_quat"], dtype=np.float64))}


def _lazy(case):
    from gap_repro.sim.libero_session import LiberoSession
    return LiberoSession(suite=case.benchmark, task_index=case.task_index)


def run_rollout(rollout: LiberoRollout, transport, *, method,
                turn_timeout=900.0, max_tool_calls=400, audit_sink=None):
    """泵一个整集 turn：turn/start → 工具事件分发 → reply → 终局。

    事件面（upstream）：item/tool/call 分发 handler 并 reply；
    turn/completed 在无终局时视为模型未完成任务（host 不代替决策）。
    返回 (success, record)。direct/hybrid 的 handler 表见 libero_tool_specs。
    """
    specs = libero_tool_specs(method)
    handlers = {"libero_start": rollout.libero_start}
    if method == "gpt_only":
        handlers["libero_act"] = rollout.libero_act
        rollout._rid_prefix = "d"
    else:
        handlers["pi05_infer"] = rollout.pi05_infer
        handlers["libero_execute"] = rollout.libero_execute
        rollout._rid_prefix = "h"
    # 提示结构（upstream：skill + gate prompt + 上下文）：GATE_PROMPT 逐字
    # （域中立，contract.py 钉定）；观测/工具说明来自本仓钉定契约。
    skill_text = (
        "Act as the autonomous policy agent for this single simulation "
        "rollout. Task instruction: " + rollout.case.instruction +
        ". Start with libero_start, then drive the episode with "
        + ("libero_act" if method == "gpt_only"
           else "pi05_infer + libero_execute") +
        " until the native terminal signal.\n"
        "Observations: two RGB images (agentview/wrist, 224px, rotated "
        "180 degrees to match training preprocessing), 8-D proprio state "
        "[eef_pos(3, m), axisangle(eef_quat)(3, rad), gripper dual-finger "
        "qpos(2)], the instruction, step_id (control steps executed), and "
        "remaining_steps (episode budget left). Pace decisions against "
        "the remaining budget.\n"
        "Actions are 7-D normalized deltas [dpos(3), drot(3), gripper(1)] "
        "within [-1, 1]. Measured controller transfer: about 0.011 m and "
        "0.085 rad per unit per step at full amplitude (setpoint-clamped; "
        "larger per-unit effect at smaller amplitudes; P0-1/v3 probe "
        "evidence). Delta rotations are interpreted in the WORLD frame "
        "(probe v3: commanded-x response axis purity 0.9975).\n"
        "Apparent visual completion is not success: the episode ends only "
        "on the environment's native terminal signal; a stop request "
        "reads the native predicate at the current state.\n"
        "Use English for every public explanation, assessment, progress "
        "update, and final report.")
    turn_text = skill_text + (
        "" if method == "gpt_only"
        else "\n\n# Unchanged baseline gate prompt\n\n" + GATE_PROMPT)
    transport.request("initialize", dict(
        clientInfo=dict(name="gap-repro-libero-policy", version="1"),
        capabilities=dict(experimentalApi=True)), turn_timeout)
    transport.notify("initialized", {})
    result = transport.request("thread/start", dict(
        model=transport.config.get("model"),
        modelProvider=transport.config.get("provider"),
        config=dict(model_reasoning_effort=transport.config.get("effort")),
        dynamicTools=specs, approvalPolicy="never"), turn_timeout)
    thread_id = result["thread"]["id"]
    turn_id = transport.request("turn/start", dict(
        threadId=thread_id,
        input=[dict(type="text", text=turn_text)]), turn_timeout)
    call_index = 0
    while not rollout.done and call_index < max_tool_calls:
        event = transport.next_message(turn_timeout)
        method_name, params = event.get("method"), event.get("params", {})
        if method_name in ("error", "turn/failed"):
            # P2-5：传输/模型错误立即报错（否则干等 900s 超时）
            raise RuntimeError(
                f"transport error event {method_name}: "
                f"{json.dumps(params)[:500]}")
        if method_name == "thread/tokenUsage/updated":
            rollout.record.setdefault("token_usage", []).append(
                params.get("tokenUsage", {}))
            continue
        if method_name == "item/tool/call":
            if params.get("threadId") not in (None, thread_id) or \
                    params.get("turnId") not in (None, turn_id):
                raise RuntimeError(
                    "Tool call belongs to another thread or turn")
            if audit_sink is not None:
                audit_sink({"call_index": call_index, "tool": params.get("tool"),
                            "arguments": params.get("arguments")})
            name, arguments = params.get("tool"), params.get("arguments")
            if arguments is None:
                arguments = {}
            elif not isinstance(arguments, dict):
                # P2-新1（upstream run.py 语义）：非 JSON-object 参数是
                # 可修正输入错误，不逃逸成 infra
                rollout.rejected_calls += 1
                packet = {"input_error": "Tool arguments must be a JSON object",
                          "rejected_calls": rollout.rejected_calls}
                transport.reply(event["id"], dict(success=False,
                                contentItems=_packet_with_images(packet)))
                call_index += 1
                continue
            try:
                packet = dispatch_tool_call(name, arguments, handlers,
                                            audit_sink=None,
                                            call_index=call_index)
                ok = True
            except InputError as error:
                rollout.rejected_calls += 1
                packet = {"input_error": str(error),
                          "rejected_calls": rollout.rejected_calls}
                ok = False
            except ValueError as error:
                if "Unknown host service" not in str(error):
                    raise  # host 侧 bug 不伪装成模型输入错误
                rollout.rejected_calls += 1
                packet = {"input_error": str(error),
                          "rejected_calls": rollout.rejected_calls}
                ok = False
            transport.reply(event["id"], dict(
                success=ok, contentItems=_packet_with_images(packet)))
            call_index += 1
        elif method_name == "turn/completed":
            break
    if not rollout.done and not rollout.stop_requested:
        raise RuntimeError(
            "turn/completed before native terminal: model abandoned the "
            "episode (infra, not a task failure)")
    if not rollout.done:
        rollout._terminal_packet(done=None)
    status = VALID_SUCCESS if rollout.record.get("success") else VALID_FAILURE
    rollout.record["status"] = status
    return status, rollout.record
