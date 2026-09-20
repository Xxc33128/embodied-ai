#!/usr/bin/env python3
"""L4：GPT 联调 CLI——单集 Direct/Hybrid 真实 episode（用户提供接入后运行）。

用法（Mac 或 gap-sim；凭据只经环境变量）：
  python3 scripts/run_lp5_gpt_episode.py \
    --method direct|hybrid --route app_server|api_gateway \
    --config configs/gpt_access.example.json \
    --benchmark libero_goal --condition Ori --task-index 0 --init-index 0 \
    --inventory /workspace/data/l1_case_inventory.json \
    --out /workspace/data/lp5/gpt_dev/goal_Ori_t0_i0.json

config（route A）：{"app_server_cmd": "...", "model": "...", "provider": "...",
"effort": "xhigh", "workspace": "..."}（workspace 可省，默认 /tmp）。
config（route B）：{"endpoint": "...", "api_key_env": "MY_KEY_ENV",
"model": "...", "effort": "...", "rate_limit_s": 1.0}。
缺失项 → ConfigMissingError 打印所需清单（即"待用户提供"的可执行清单）。
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/repo/src")

from gap_repro.agent.client import ApiGatewayTransport, AppServerTransport  # noqa: E402
from gap_repro.libero.gpt_bridge import LiberoRollout, run_rollout  # noqa: E402
from gap_repro.libero.policy import CaseSpec, LiberoStudentPolicy, FileNpzTransport  # noqa: E402


def load_case(inventory_path, benchmark, condition, task_index, init_index,
              campaign_id="gpt_dev", method="student_only"):
    inventory = json.load(open(inventory_path))
    for case in inventory["cases"]:
        if (case["benchmark"] == benchmark and case["condition"] == condition
                and case["task_index"] == task_index):
            if not (case.get("bddl_exists") and case.get("init_exists")):
                raise SystemExit(f"case 资产不完整: {case}")
            episode_id = f"{benchmark}#{task_index}#i{init_index}"
            return CaseSpec.from_inventory(
                case, campaign_id=campaign_id, episode_id=episode_id,
                init_index=init_index, method=method)
    raise SystemExit(f"case 不在清单: {benchmark}/{condition}#{task_index}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["direct", "hybrid"], required=True)
    ap.add_argument("--route", choices=["app_server", "api_gateway"],
                    required=True)
    ap.add_argument("--config", required=True, help="接入配置 JSON（凭据只存环境变量名）")
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--benchmark", default="libero_goal")
    ap.add_argument("--condition", default="Ori")
    ap.add_argument("--task-index", type=int, default=0)
    ap.add_argument("--init-index", type=int, default=0)
    ap.add_argument("--req-dir", default="/workspace/data/lp5/req")
    ap.add_argument("--resp-dir", default="/workspace/data/lp5/resp")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    config = json.load(open(args.config))
    transport = (AppServerTransport(config) if args.route == "app_server"
                 else ApiGatewayTransport(config))
    try:
        transport.connect()
    except Exception as e:
        raise SystemExit(f"接入配置不完整/不可用：{e}\n"
                         f"（补齐后重跑；凭据经环境变量，不写入任何文件）")

    method = "gpt_only" if args.method == "direct" else "pi05_plus_gpt"
    case = load_case(args.inventory, args.benchmark, args.condition,
                     args.task_index, args.init_index, method=args.method)
    policy = None
    if args.method == "hybrid":
        policy = LiberoStudentPolicy(FileNpzTransport(
            args.req_dir, args.resp_dir))
    rollout = LiberoRollout(case, policy=policy)
    status, record = None, None
    try:
        status, record = run_rollout(rollout, transport, method=method)
    except Exception as e:
        # P2-4：infra（弃驶/超时/传输错误）也必须落盘完整记录
        status = "infrastructure_incomplete"
        record = dict(rollout.record)
        record["infra_error"] = f"{type(e).__name__}: {e}"
    finally:
        transport.close()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"status": status, "record": record}, f,
                  indent=1, ensure_ascii=False)
    print(f"GPT_EPISODE_DONE method={args.method} status={status} "
          f"success={record.get('success')} out={out}", flush=True)


if __name__ == "__main__":
    main()
