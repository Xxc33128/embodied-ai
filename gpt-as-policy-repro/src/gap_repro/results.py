"""W10：结果核算器与 campaign 冻结（计划 §8 语义）。

状态分类（§8.2）：valid_success / valid_failure / invalid_native_layout /
budget_censored / infrastructure_incomplete。
分母（§8.3）：全解决才输出主口径 successes/N；已解决集率 S/(N-U)；
固定 panel 界 [S/N, (S+U)/N]（缺失上下界，非置信区间）。
配对：仅双方均已解决的 case 进入配对；缺失清单必须报告。
native_score：缺失保留 null；全量可用才计算均值；不补零。
attempt（§8.1）：全部保留；最终取首个进入策略阶段的 attempt；
基础设施工具初始化失败最多重试 2 次（共 3 次）。
"""

from __future__ import annotations

import hashlib
import json as _json
from dataclasses import dataclass, field

VALID_SUCCESS = "valid_success"
VALID_FAILURE = "valid_failure"
INVALID_LAYOUT = "invalid_native_layout"
BUDGET_CENSORED = "budget_censored"
INFRA_INCOMPLETE = "infrastructure_incomplete"

RESOLVED = {VALID_SUCCESS, VALID_FAILURE}
UNRESOLVED = {INVALID_LAYOUT, BUDGET_CENSORED, INFRA_INCOMPLETE}


@dataclass
class EpisodeResult:
    campaign_id: str
    case_id: str
    method: str                 # direct / hybrid / student_only
    route: str                  # app_server / api_gateway
    status: str                 # 五态之一
    native_score: float | None = None
    attempt_id: int = 0
    entered_policy_stage: bool = True
    reason: str | None = None
    init_fingerprint: str | None = None  # §8.3 初始 RGB/本体/EEF 指纹

    def __post_init__(self):
        if self.status not in RESOLVED | UNRESOLVED:
            raise ValueError(f"unknown status: {self.status}")
        if self.status == VALID_SUCCESS and self.native_score is None:
            raise ValueError("valid_success requires native_score")
        if self.status == VALID_FAILURE and self.native_score is None:
            # §8.2：valid_failure 带"原 score"（idle/RPC 超时属 infrastructure_incomplete，
            # 不落在本态——作者式不补分情形由五态外的独立裁定承担）
            raise ValueError("valid_failure requires native_score (author-style "
                             "timeout-without-score maps to infrastructure_incomplete)")


MAX_INIT_RETRIES = 2  # §8.1.2：基础设施初始化额外重试最多 2 次（共 3 次 attempt）


def select_attempt(attempts: list[EpisodeResult]) -> EpisodeResult | None:
    """§8.1.2 语义：
    - 首个进入策略阶段的 attempt 为最终结果（§8.1.3 不挑好结果）；
    - 未进入策略阶段的 attempts 视为基础设施初始化失败，总数超过
      1+MAX_INIT_RETRIES 时违规抛错（invalid 布局不适用重试规则，由调用方
      以 INVALID_LAYOUT 状态另行登记，不进入本函数的 attempts）。"""
    if not attempts:
        return None
    entered = [a for a in attempts if a.entered_policy_stage]
    if entered:
        if len(attempts) - len(entered) > MAX_INIT_RETRIES:
            raise ValueError(
                f"init retries {len(attempts) - len(entered)} exceed "
                f"1+{MAX_INIT_RETRIES} allowed (§8.1.2)")
        return entered[0]
    if len(attempts) > 1 + MAX_INIT_RETRIES:
        raise ValueError(
            f"all {len(attempts)} attempts never entered policy stage; "
            f"retries exceed §8.1.2 limit")
    return attempts[0]


def summarize_route(results: list[EpisodeResult], *,
                    expected_case_ids: list[str]) -> dict:
    """单路线核算（T1，audit F01）：分母来自冻结 case 清单，非调用方口算。

    - expected_case_ids 必须唯一；结果引用清单外 case / 混合 campaign 拒绝；
    - 未登记的冻结 case 记 missing_log，计入 unresolved（缺失不消失）；
    - attempt 按 attempt_id 排序取首个进入策略阶段者（不依赖日志拼接次序，
      不挑好结果）；同 (case, attempt_id) 幂等重日志去重、冲突重日志拒绝；
    - 敏感性"infra 按失败计"只在 valid+infra 全覆盖时给完整口径
      （=S/(S+F+infra)），invalid/budget/缺失未决时给部分口径并显式标注；
    - per_case 保留 init_fingerprint 供配对资格检查。
    """
    expected = list(expected_case_ids)
    if len(set(expected)) != len(expected):
        raise ValueError("duplicate case ids in frozen panel")
    expected_set = set(expected)
    n = len(expected)

    campaigns = {r.campaign_id for r in results}
    if len(campaigns) > 1:
        raise ValueError(f"mixed campaigns in one route summary: {campaigns}")
    methods_seen, routes_seen = set(), set()
    by_case: dict[str, list[EpisodeResult]] = {c: [] for c in expected}
    for r in results:
        if r.case_id not in expected_set:
            raise ValueError(f"case {r.case_id!r} outside frozen panel")
        methods_seen.add(r.method)
        routes_seen.add(r.route)
        by_case[r.case_id].append(r)
    if len(methods_seen) > 1 or len(routes_seen) > 1:
        raise ValueError(f"mixed methods/routes in one route summary: "
                         f"{methods_seen} {routes_seen}")

    resolved: dict[str, EpisodeResult] = {}
    unresolved: dict[str, str] = {}
    attempt_counts: dict[str, int] = {}
    for case_id in expected:
        raw = by_case[case_id]
        # 幂等重日志去重（同 attempt_id 同内容），冲突拒绝；attempt_id 排序
        seen: dict[int, EpisodeResult] = {}
        for a in sorted(raw, key=lambda x: x.attempt_id):
            prev = seen.get(a.attempt_id)
            if prev is None:
                seen[a.attempt_id] = a
                continue
            same = (prev.status, prev.native_score, prev.entered_policy_stage,
                    prev.reason, prev.init_fingerprint) == \
                   (a.status, a.native_score, a.entered_policy_stage,
                    a.reason, a.init_fingerprint)
            if not same:
                raise ValueError(
                    f"case {case_id!r} attempt_id {a.attempt_id} conflicting re-log")
        attempts = list(seen.values())
        attempt_counts[case_id] = len(attempts)
        sel = select_attempt(attempts)
        if sel is None:
            unresolved[case_id] = "missing_log"
        elif sel.status in RESOLVED:
            resolved[case_id] = sel
        else:
            unresolved[case_id] = sel.status

    S = len(resolved)
    U = len(unresolved)
    success_count = sum(1 for r in resolved.values() if r.status == VALID_SUCCESS)
    scores = [r.native_score for r in resolved.values()
              if r.native_score is not None]
    full_score_available = len(scores) == n and U == 0
    missing = sorted(c for c, st in unresolved.items() if st == "missing_log")
    infra_cases = sorted(c for c, st in unresolved.items()
                         if st == INFRA_INCOMPLETE)
    other_unresolved = {c: st for c, st in unresolved.items()
                        if st not in (INFRA_INCOMPLETE, "missing_log")}

    # 敏感性：infra 按失败计 = S/(S+F+infra)，仅 valid+infra 全覆盖时给完整口径
    valid_failures = sum(1 for r in resolved.values() if r.status == VALID_FAILURE)
    full_scope = not other_unresolved and not missing
    if full_scope:
        denom = success_count + valid_failures + len(infra_cases)
        sens = (success_count / denom) if denom else None
        sens_detail = {"scope": "full", "numerator": success_count,
                       "denominator": denom, "infra_cases": infra_cases,
                       "remaining_unresolved": {}}
    else:
        sens = None  # 部分口径不给单一数字，防误读为完整成功率
        sens_detail = {"scope": "partial", "numerator": success_count,
                       "denominator": success_count + valid_failures + len(infra_cases),
                       "infra_cases": infra_cases,
                       "remaining_unresolved": {**other_unresolved,
                                                **{c: "missing_log" for c in missing}}}

    out = {
        "panel_size": n,
        "expected_case_ids": expected,
        "missing_cases": missing,
        "registered_cases": len({r.case_id for r in results}),
        "attempt_counts": attempt_counts,  # §8.1.1 全部 attempt 保留的证据
        "resolved": S,
        "unresolved": U,
        "success_count": success_count,
        "headline_success_rate": (success_count / n) if U == 0 else None,
        "resolved_only_rate": (success_count / (n - U)) if (n - U) > 0 else None,
        "bounds": [success_count / n, (success_count + U) / n],
        "score_mean_full_panel": (sum(scores) / len(scores)) if full_score_available else None,
        "score_available_mean": (sum(scores) / len(scores)) if scores else None,
        "n_scored": len(scores),
        "unresolved_detail": unresolved,
        "sensitivity_infra_as_failure": sens,
        "sensitivity_detail": sens_detail,
        "per_case": {cid: {"status": r.status, "native_score": r.native_score,
                           "reason": r.reason, "attempt_id": r.attempt_id,
                           "init_fingerprint": r.init_fingerprint}
                     for cid, r in resolved.items()},
    }
    return out


def pair_routes(route_a: dict, route_b: dict) -> dict:
    """§8.3 配对：双方均已解决的 case 才入配对；缺失清单单独报告。

    T1（audit F02）：per_case 的 init_fingerprint 进入配对记录；指纹不一致的
    配对显式列入 fingerprint_mismatch_pairs 并单计严格同指纹子集——不同布局
    的同 case 不得无声计入"严格同布局配对"。
    """
    a_cases = set(route_a["per_case"])
    b_cases = set(route_b["per_case"])
    common = sorted(a_cases & b_cases)
    paired = []
    fingerprint_mismatch = []
    strict = 0
    for cid in common:
        a, b = route_a["per_case"][cid], route_b["per_case"][cid]
        a_win = (a["status"] == VALID_SUCCESS) - (b["status"] == VALID_SUCCESS)
        a_fp, b_fp = a.get("init_fingerprint"), b.get("init_fingerprint")
        entry = {"case_id": cid, "a_success": a["status"] == VALID_SUCCESS,
                 "b_success": b["status"] == VALID_SUCCESS, "direction": a_win,
                 "a_fingerprint": a_fp, "b_fingerprint": b_fp}
        paired.append(entry)
        if a_fp is not None and b_fp is not None:
            if a_fp == b_fp:
                strict += 1
            else:
                fingerprint_mismatch.append(
                    {"case_id": cid, "a": a_fp, "b": b_fp})
    missing_from_pairing = {
        "a_resolved_b_missing": sorted(a_cases - b_cases),
        "b_resolved_a_missing": sorted(b_cases - a_cases),
    }
    both_success = sum(1 for p in paired if p["a_success"] and p["b_success"])
    both_fail = sum(1 for p in paired if not p["a_success"] and not p["b_success"])
    a_only = sum(1 for p in paired if p["a_success"] and not p["b_success"])
    b_only = len(paired) - both_success - both_fail - a_only
    return {
        "paired_count": len(paired),
        "paired_fingerprint_strict_count": strict,
        "fingerprint_mismatch_pairs": fingerprint_mismatch,
        "both_success": both_success, "both_fail": both_fail,
        "a_only": a_only, "b_only": b_only,
        "paired": paired,
        "missing_from_pairing": missing_from_pairing,
        "note": "配对仅含双方已解决 case；缺失清单原样报告，不补零不替换；"
                "初始指纹不一致的配对显式列出，严格同布局口径单计",
    }


def map_session_status(session_status: str, reason: str | None,
                       native_score: float | None) -> str:
    """原 session status/reason → 五态适配（§8.2）。原样保留原字段。"""
    table = {
        "native_completed": None,        # 由 reward 结果细分 success/failure
        "invalid_native_layout": INVALID_LAYOUT,
        "budget_censored": BUDGET_CENSORED,
        "incomplete": INFRA_INCOMPLETE,
    }
    mapped = table.get(session_status)
    if mapped is None:
        if session_status == "native_completed":
            return VALID_SUCCESS if native_score == 1.0 else VALID_FAILURE
        raise ValueError(f"unknown session status: {session_status}")
    return mapped


def freeze_campaign(campaign: dict, input_paths: dict[str, str]) -> dict:
    """campaign 冻结：对每个输入文件计算 sha256 并写入 manifest；
    冻结后再改动任何输入都会导致哈希不一致（可检测）。"""
    import datetime
    from pathlib import Path

    frozen = dict(campaign)
    frozen["frozen_inputs"] = {}
    for name, path in input_paths.items():
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(1 << 20):
                h.update(chunk)
        frozen["frozen_inputs"][name] = {"path": path, "sha256": h.hexdigest()}
    import datetime
    import json as _json
    campaign_core = _json.dumps(
        {k: v for k, v in campaign.items()}, sort_keys=True, ensure_ascii=False)
    frozen["campaign_content_sha256"] = hashlib.sha256(
        campaign_core.encode("utf-8")).hexdigest()
    frozen["frozen_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    frozen["frozen"] = True
    return frozen


def verify_frozen_campaign(manifest: dict) -> dict:
    """加载端复算：campaign 内容哈希 + 各输入文件哈希（篡改可检测）。"""
    problems = []
    core = _json.dumps({k: v for k, v in manifest.items()
                        if k not in ("frozen_inputs", "frozen_at", "frozen",
                                     "campaign_content_sha256")},
                       sort_keys=True, ensure_ascii=False)
    if hashlib.sha256(core.encode("utf-8")).hexdigest() != manifest.get("campaign_content_sha256"):
        problems.append("campaign content hash mismatch")
    from pathlib import Path
    for name, ent in manifest.get("frozen_inputs", {}).items():
        p = Path(ent["path"])
        if not p.is_file():
            problems.append(f"{name}: file missing")
            continue
        h = hashlib.sha256()
        with open(p, "rb") as f:
            while chunk := f.read(1 << 20):
                h.update(chunk)
        if h.hexdigest() != ent["sha256"]:
            problems.append(f"{name}: input hash mismatch")
    return {"ok": not problems, "problems": problems}
