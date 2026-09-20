#!/usr/bin/env python3
"""L3：LIBERO 批量 campaign CLI（学生路径；gap-sim/gap-repro 协同执行）。

用法（gap-sim 容器内）：
  python3 scripts/run_lp5_campaign.py \
    --inventory /workspace/data/l1_case_inventory.json \
    --campaign-id dev_smoke --mode development \
    --episodes-per-case 1 --conditions Ori,Env \
    --req-dir /workspace/data/lp5/req --resp-dir /workspace/data/lp5/resp \
    --out /workspace/data/lp5/dev_smoke_results.json

模式语义（计划 L3）：
- development：管线覆盖/开发，结果不入正式 n；无需冻结 manifest。
- formal：必须已存在冻结 manifest（--freeze-manifest），启动时
  verify_frozen_campaign 校验 campaign 内容与全部输入哈希；任何不一致
  拒绝启动（audit F05：禁止看到结果再挑案例）。

核算（T1）：分母 = 冻结 case 清单（expected_case_ids）；输出按
benchmark（=suite×condition）分组 summarize_route + 全局 missing_log。
增量续跑：--out 已有的 episode 按 (episode_id) 跳过。
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/repo/src")

from gap_repro.libero.policy import (  # noqa: E402
    CaseSpec,
    FileNpzTransport,
    LiberoStudentPolicy,
)
from gap_repro.libero.runner import run_student_episode  # noqa: E402
from gap_repro.results import (  # noqa: E402
    EpisodeResult,
    freeze_campaign,
    summarize_route,
    verify_frozen_campaign,
)

CONDITIONS = ("Ori", "Obj", "Pos", "Sem", "Task", "Env")


def select_cases(inventory: dict, *, conditions, suites, episodes_per_case,
                 campaign_id):
    """冻结 case 池展开：inventory 完整 case × init 索引 → CaseSpec 列表。

    episode_id = f"{benchmark}#{task_index}#i{init_index}"（正式配对的
    身份原子；GPT 侧同 id 配对）。缺失资产 case 拒绝进入（不抽件补新）。
    """
    specs = []
    for case in inventory["cases"]:
        if case["condition"] not in conditions:
            continue
        suite = case["benchmark"]
        if not _suite_match(suite, suites):
            continue
        if not (case.get("bddl_exists") and case.get("init_exists")):
            raise RuntimeError(
                f"incomplete case in pool: {case['benchmark']}#{case['task_index']}"
                f"（禁止抽掉失败 case 补新 case）")
        for i in range(episodes_per_case):
            episode_id = f"{case['benchmark']}#{case['task_index']}#i{i}"
            specs.append(CaseSpec.from_inventory(
                case, campaign_id=campaign_id, episode_id=episode_id,
                init_index=i))
    return specs


def _suite_match(benchmark: str, suites) -> bool:
    for s in suites:
        if benchmark == s:
            return True
        if benchmark.startswith(s + "_"):
            rest = benchmark[len(s) + 1:]
            if rest in ("object", "swap", "lan", "task", "env"):
                return True
    return False


def _atomic_json_dump(obj, path: Path):
    """P2-2：句柄写 + os.rename（dump 中途崩溃不得截断 results 文件）。"""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    os.rename(tmp, path)


def load_done(out_path: Path):
    if out_path.exists():
        return {r["case_id"] for r in json.load(open(out_path))["results"]}
    return set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--campaign-id", required=True)
    ap.add_argument("--mode", choices=["development", "formal"],
                    default="development")
    ap.add_argument("--freeze-manifest", default=None,
                    help="formal 模式必需：冻结 manifest 路径")
    ap.add_argument("--episodes-per-case", type=int, default=50)
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--suites", default="libero_goal,libero_spatial,"
                                        "libero_10,libero_object")
    ap.add_argument("--req-dir", default="/workspace/data/lp5/req")
    ap.add_argument("--resp-dir", default="/workspace/data/lp5/resp")
    ap.add_argument("--server-timeout-s", type=float, default=180.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    conditions = tuple(c for c in args.conditions.split(",") if c)
    suites = [s for s in args.suites.split(",") if s]
    inventory = json.load(open(args.inventory))
    specs = select_cases(inventory, conditions=conditions, suites=suites,
                         episodes_per_case=args.episodes_per_case,
                         campaign_id=args.campaign_id)
    if not specs:
        raise SystemExit("no cases selected")
    expected_ids = [s.episode_id for s in specs]
    assert len(set(expected_ids)) == len(expected_ids), "episode_id 冲突"

    if args.mode == "formal":
        if not args.freeze_manifest:
            raise SystemExit("formal 模式需要 --freeze-manifest")
        manifest = json.load(open(args.freeze_manifest))
        verdict = verify_frozen_campaign(manifest)
        if not verdict["ok"]:
            raise SystemExit(f"冻结校验失败，拒绝启动: {verdict}")
        frozen_ids = manifest["expected_case_ids"]  # freeze_campaign 扁平合并
        if sorted(frozen_ids) != sorted(expected_ids):
            raise SystemExit("case 池与冻结 manifest 不一致，拒绝启动")
        print(f"FORMAL_FREEZE_VERIFIED cases={len(expected_ids)}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path)
    if out_path.exists():
        state = json.load(open(out_path))
    else:
        state = {"schema": "gap_repro.lp5_campaign.v1",
                 "mode": args.mode, "campaign_id": args.campaign_id,
                 "expected_case_ids": expected_ids, "results": []}

    transport = FileNpzTransport(args.req_dir, args.resp_dir,
                                 timeout_s=args.server_timeout_s)
    policy = LiberoStudentPolicy(transport)

    n_done = 0
    for spec in specs:
        if spec.episode_id in done:
            continue
        result, record = run_student_episode(spec, policy=policy)
        entry = {"case_id": spec.episode_id,
                 "status": result.status,
                 "native_score": result.native_score,
                 "method": result.method,
                 "route": result.route,
                 "attempt_id": result.attempt_id,
                 "entered_policy_stage": result.entered_policy_stage,
                 "reason": result.reason,
                 "init_fingerprint": result.init_fingerprint,
                 "record": record}
        state["results"].append(entry)
        n_done += 1
        _atomic_json_dump(state, out_path)
        print(f"[{result.status}] {spec.episode_id} "
              f"done_step={record.get('done_step')}", flush=True)

    # 核算：分母 = 冻结清单（T1）；按 benchmark 分组 + 全局
    results = [EpisodeResult(
        campaign_id=args.campaign_id, case_id=e["case_id"],
        method=e["method"], route=e["route"], status=e["status"],
        native_score=e["native_score"], attempt_id=e["attempt_id"],
        entered_policy_stage=e["entered_policy_stage"],
        reason=e["reason"], init_fingerprint=e["init_fingerprint"])
        for e in state["results"]]
    by_benchmark = {}
    for s in specs:
        by_benchmark.setdefault(s.benchmark, []).append(s.episode_id)
    summary = {b: summarize_route(
        [r for r in results if r.case_id in ids],
        expected_case_ids=ids)
        for b, ids in sorted(by_benchmark.items())}
    state["summary"] = summary
    _atomic_json_dump(state, out_path)
    n_resolved = sum(s["resolved"] for s in summary.values())
    n_success = sum(s["success_count"] for s in summary.values())
    print(f"CAMPAIGN_DONE selected={len(expected_ids)} "
          f"executed_now={n_done} resolved={n_resolved} "
          f"success={n_success}", flush=True)


if __name__ == "__main__":
    main()
