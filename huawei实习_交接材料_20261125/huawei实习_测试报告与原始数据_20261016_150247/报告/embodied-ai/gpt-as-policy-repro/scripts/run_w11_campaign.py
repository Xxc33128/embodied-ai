#!/usr/bin/env python3
"""W11 正式运行 runner（脚手架）。

按 campaign 冻结的 50 case × {direct, hybrid} × 路线逐 episode 执行：
environment(MuJoCo) + pi05 NPU 服务 + check_once 判据 + results 核算。
每 episode 独立 reset（seed=layout_id）；输出逐 case EpisodeResult JSON。
用法：python3 scripts/run_w11_campaign.py --campaign configs/campaigns/w11_v1.json --dry-run
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_campaign(path):
    c = json.load(open(path))
    assert c.get("frozen") is True, "campaign 未冻结（freeze_campaign）"
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--methods", default="direct,hybrid")
    args = ap.parse_args()
    campaign = load_campaign(ROOT / args.campaign if not Path(args.campaign).is_absolute() else Path(args.campaign))
    if len(campaign["cases"]) != 50:
        raise SystemExit(f"campaign cases != 50: {len(campaign['cases'])}")
    cases = campaign["cases"]
    plan = [{"case_id": c, "method": m} for c in cases for m in args.methods.split(",")]
    print(json.dumps({"planned_episodes": len(plan),
                      "note": "每 episode 独立 reset；全部 attempt 保留；"
                              "没有进入策略阶段的 episode 按 §8.1.2 最多重试 2 次",
                      "plan_head": plan[:3]}, ensure_ascii=False, indent=1))
    if args.dry_run:
        return
    raise SystemExit("正式执行需要：W3 渲染（视觉观测）+ 场景对象转换 + 服务编排；"
                     "当前 W11 脚手架仅规划与核算（如实标注，不虚构完成）")


if __name__ == "__main__":
    main()
