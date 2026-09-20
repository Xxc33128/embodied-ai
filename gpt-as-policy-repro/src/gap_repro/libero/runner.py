"""L3：LIBERO 批量 runner（学生路径）——单 episode 执行与审计记录。

协议口径（openpi examples/libero/main.py 逐字 + 计划 L3）：
- 沉降 = reset_to 内置 wait_steps=10（P1-1 修正：此前 runner 又走 10 步
  dummy = 双重 20 步，违背 wait10 锚且与 GPT 路径不对称）。
- 总循环 t < max_steps（控制步）；action_plan 空时重规划（每
  REPLAN_STEPS=5 步），popleft 执行。
- 终局：env done（原生 success，openpi 同源）→ VALID_SUCCESS(1.0)；
  预算耗尽未 done → VALID_FAILURE(0.0)（论文协议：固定预算截断是正常
  失败，不是 censored）；基础设施异常 → INFRA_INCOMPLETE（如实记录，
  禁止零填充/挑好重跑——attempt 语义在 results.select_attempt）。
  超时后的 is_success() 只作辅助记录（final_is_success_check），不翻转
  判定——与 openpi 客户端 done-only 语义逐字一致。
- 每 chunk 审计记录：request_id、候选身份（policy.identity）、执行区间
  [t_start, t_end)、chunk 前 state 指纹。学生请求失败一律升格
  INFRA_INCOMPLETE 终止本集，绝不静默重发（同 request_id 重 infer 会得到
  新采样——openpi 内部采样不可复现，重发=换候选，违反 §8.1.3）。
"""
import time

import numpy as np

from gap_repro.libero.policy import (
    REPLAN_STEPS,
    WAIT_STEPS,
    LiberoStudentPolicy,
)
from gap_repro.results import (
    INFRA_INCOMPLETE,
    VALID_FAILURE,
    VALID_SUCCESS,
    EpisodeResult,
)


def run_student_episode(case, *, policy: LiberoStudentPolicy,
                        session=None, session_factory=None,
                        audit_sink=None, wait_steps: int = WAIT_STEPS):
    """执行单集学生 episode，返回 (EpisodeResult, record dict)。

    session_factory(case) → LiberoSession（生产注入真构造；测试注入 fake）。
    audit_sink(record) 逐 chunk 回调（落盘/累积由调用方决定）。
    """
    t0_wall = time.time()
    record = {"schema": "gap_repro.lp_runner_episode.v1",
              "case": {"campaign_id": case.campaign_id,
                       "episode_id": case.episode_id,
                       "benchmark": case.benchmark,
                       "condition": case.condition,
                       "task_index": case.task_index,
                       "task_name": case.task_name,
                       "init_index": case.init_index,
                       "max_steps": case.max_steps,
                       "instruction": case.instruction,
                       "method": case.method},
              "chunks": [], "success": None, "done_step": None,
              "final_is_success_check": None, "infra_error": None}
    result = None
    try:
        if session is None:
            if session_factory is not None:
                session = session_factory(case)
            else:
                # 延迟导入：LiberoSession 依赖 libero 包（仅环境机可导入；
                # 测试路径经 session/session_factory 注入，不触达）
                from gap_repro.sim.libero_session import LiberoSession
                session = LiberoSession(suite=case.benchmark,
                                        task_index=case.task_index)
        obs = session.reset_to(case.init_index, wait_steps=wait_steps)
        record["reset_fingerprint_state"] = session.fingerprint_state(obs)
        record["reset_fingerprint_full"] = session.fingerprint_full(obs)
        record["n_init_states"] = int(len(session.init_states))

        prompt = case.instruction
        action_plan = []
        done = False
        t = 0
        # P1-1：沉降已由 reset_to(wait_steps=10) 完成（openpi wait10 逐字）；
        # 此处直接从控制步 0 开始，不再二次 wait（此前双重 20 步违背锚）。
        while t < case.max_steps:
            if not action_plan:
                request_id = (f"{case.campaign_id}|{case.episode_id}|"
                              f"r{len(record['chunks'])}")
                fp_before = session.fingerprint_state(obs)
                chunk, identity = policy.infer_chunk(obs, prompt, request_id)
                action_plan = [row for row in chunk]
                record["chunks"].append({
                    "request_id": request_id,
                    "t_start": t,
                    "fingerprint_state_before": fp_before,
                    "identity": identity,
                })
            action = action_plan.pop(0)
            obs, _, done, _ = session.step(np.asarray(action, dtype=np.float64))
            t += 1
            if done:
                record["chunks"][-1]["t_end"] = t
                record["chunks"][-1]["done_in_chunk"] = True
                break
            if not action_plan:
                record["chunks"][-1]["t_end"] = t
        if record["chunks"] and "t_end" not in record["chunks"][-1]:
            record["chunks"][-1]["t_end"] = t
        record["final_is_success_check"] = bool(session.is_success())
        record["success"] = bool(done)
        record["done_step"] = t if done else case.max_steps
        status = VALID_SUCCESS if done else VALID_FAILURE
        native = 1.0 if done else 0.0
    except Exception as e:  # 基础设施异常如实记录；绝不改判成功
        record["infra_error"] = f"{type(e).__name__}: {e}"
        status, native, done = INFRA_INCOMPLETE, None, None
    record["wall_s"] = round(time.time() - t0_wall, 2)
    if audit_sink is not None:
        audit_sink(record)
    n_chunks = len(record.get("chunks", []))
    reason = None
    if record.get("infra_error"):
        reason = record["infra_error"]
    else:
        reason = (f"done_step={record.get('done_step')} n_chunks={n_chunks} "
                  f"final_is_success_check={record.get('final_is_success_check')}")
    result = EpisodeResult(
        campaign_id=case.campaign_id, case_id=case.episode_id,
        method=case.method, route="file_npz", status=status,
        native_score=native, attempt_id=0,
        entered_policy_stage=(n_chunks > 0),
        reason=reason,
        init_fingerprint=record.get("reset_fingerprint_state"))
    return result, record
