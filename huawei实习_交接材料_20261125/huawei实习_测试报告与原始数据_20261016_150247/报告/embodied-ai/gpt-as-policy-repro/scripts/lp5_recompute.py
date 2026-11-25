#!/usr/bin/env python3
"""L5：LP5 每格样本量精确重算（审计 F11 修正版）。

对 2026-09-19 冻结提案的三处修正：
1. 精度公式：Wald ⌈z²p(1−p)/w²⌉ → 精确 Wilson 整数搜索
   （gap_repro.stats.wilson.min_n_for_halfwidth，at-p 口径：观测 p̂=p 处
   半宽 ≤ w）。审计数字复核：p=0.01/w=0.05 精确 N=89（Wald 给 16，
   实际半宽 0.1045——严重低估）。
2. 池上限：每格 case 池 = 10 任务 × 50 init = **500 集**。凡 N>500 的格
   标 infeasible@w（须放宽 w 或接受截断精度）——提案未查此约束
   （goal·Pos@0.04=566、goal/spatial/10·Env@0.04≈597 均 >500）。
3. 零先验格（p=0，goal·Task）：at-p 语义退化为"观测 0 成功时上界 ≤ w"
   的检测口径（提案保底 n=20 时上界 0.1611——如实并列为检测能力）。

p 先验来源：2026-09-19 提案文档的论文锚定表（π₀.₅ 公开值）。
**来源缺口登记**：提案引用 configs/libero_pro.lock.json::pi05_table，
该键在现行 lock 中不存在——须在正式冻结前补录 lock（本脚本输出
prior_source 字段显式标注）。

输出：docs/acceptance/lp5-recompute.json + stdout 汇总表。
"""
import json
import sys

sys.path.insert(0, "src")
from gap_repro.stats.wilson import (  # noqa: E402
    wilson_half_width,
    wilson_interval,
    min_n_for_halfwidth,
)

POOL_CAP = 500  # 10 tasks × 50 inits

# 论文锚定（提案 §2 表，π₀.₅ 行）
PRIORS = {
    ("libero_goal", "Ori"): 0.97, ("libero_goal", "Obj"): 0.97,
    ("libero_goal", "Pos"): 0.38, ("libero_goal", "Sem"): 0.97,
    ("libero_goal", "Task"): 0.00, ("libero_goal", "Env"): 0.46,
    ("libero_spatial", "Ori"): 0.98, ("libero_spatial", "Obj"): 0.97,
    ("libero_spatial", "Pos"): 0.20, ("libero_spatial", "Sem"): 0.97,
    ("libero_spatial", "Task"): 0.01, ("libero_spatial", "Env"): 0.46,
    ("libero_10", "Ori"): 0.93, ("libero_10", "Obj"): 0.92,
    ("libero_10", "Pos"): 0.08, ("libero_10", "Sem"): 0.93,
    ("libero_10", "Task"): 0.01, ("libero_10", "Env"): 0.46,
    ("libero_object", "Ori"): 0.98, ("libero_object", "Obj"): 0.98,
    ("libero_object", "Pos"): 0.17, ("libero_object", "Sem"): 0.96,
    ("libero_object", "Task"): 0.01, ("libero_object", "Env"): 0.73,
}
MAX_STEPS = {"libero_goal": 300, "libero_spatial": 220,
             "libero_10": 520, "libero_object": 280}
# 提案分层（B 档口径）：A=Task×4 + goal/10 的 Ori+Sem @0.05；
# B=其余 Ori/Obj/Pos/Sem @0.07；C=Env×4 @0.07（备选 0.10）
TIER_W = {"A": 0.05, "B": 0.07, "C": 0.07}


def tier_of(suite, cond):
    if cond == "Task":
        return "A"
    if cond == "Env":
        return "C"
    if cond in ("Ori", "Sem") and suite in ("libero_goal", "libero_10"):
        return "A"
    return "B"


def main():
    out = {"schema": "gap_repro.lp5_recompute.v2",
           "date": "2026-09-20",
           "method": "exact Wilson integer search (at-p), audit F11",
           "pool_cap": POOL_CAP,
           "prior_source": "2026-09-19 提案文档论文锚定表；lock::pi05_table "
                           "缺失待补录（来源缺口）",
           "cells": {}, "wald_legacy_note": {}}
    total = {"A": 0, "B": 0, "C": 0}
    infeasible = []
    rows = []
    for (suite, cond), p in sorted(PRIORS.items()):
        w = TIER_W[tier_of(suite, cond)]
        r = min_n_for_halfwidth(p, w, criterion="at-p")
        n = r["n"]
        # 零/低 p 格的检测口径补充：观测 0 成功时的上界
        k0_upper = wilson_interval(0, n)[1] if p <= 0.01 else None
        # Wald 对照（提案旧数）
        z2 = 1.959963984540054 ** 2
        wald = -(-int(z2 * p * (1 - p) * 10000) // (int(w * 100) ** 2)) \
            if p not in (0.0,) else None
        cap_flag = n > POOL_CAP
        if cap_flag:
            infeasible.append({"cell": f"{suite}·{cond}", "n": n, "w": w})
        alloc = {"per_task_floor": n // 10, "remainder": n % 10}
        cell = {"prior_p": p, "tier": tier_of(suite, cond), "w": w,
                "n_exact_wilson": n, "n_wald_legacy": wald,
                "feasible_within_pool": not cap_flag,
                "k0_upper_at_n": round(k0_upper, 4) if k0_upper is not None
                                 else None,
                "allocation": alloc,
                "npu_upper_chunks": n * (MAX_STEPS[suite] // 5),
                "env_hours_upper": round(
                    n * MAX_STEPS[suite] * 0.82 / 3600 / 32, 2)}
        out["cells"][f"{suite}·{cond}"] = cell
        total[cell["tier"]] += min(n, POOL_CAP)
        rows.append((f"{suite}·{cond}", p, w, n, wald,
                     "OVER" if cap_flag else "ok"))

    out["totals"] = {
        "A": total["A"], "B": total["B"], "C_at_0.07": total["C"],
        "grand_A_B_C007": total["A"] + total["B"] + total["C"],
        "note": "超池格按 500 计入总量并标 infeasible；正式冻结须放宽 w "
                "或声明截断精度",
        "infeasible_cells": infeasible,
    }
    json.dump(out, open("docs/acceptance/lp5-recompute.json", "w"),
              indent=1, ensure_ascii=False)

    print(f"{'cell':<22}{'p':>5}{'w':>6}{'N_exact':>9}{'N_wald':>8}  pool")
    for r in rows:
        print(f"{r[0]:<22}{r[1]:>5}{r[2]:>6}{r[3]:>9}{str(r[4]):>8}  {r[5]}")
    print("\nTOTALS:", json.dumps(out["totals"], ensure_ascii=False)[:500])


if __name__ == "__main__":
    main()
