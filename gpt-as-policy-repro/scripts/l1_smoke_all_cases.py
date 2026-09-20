#!/usr/bin/env python3
"""L1 收尾门：24 变体 × 10 任务逐 case 冒烟（gap-sim 执行）。

计划 L1 验收项（repro-next-execution.md §L1）：每个任务级配置完成 reset、观测、
原成功判据读取和少量 dummy step；记录缺失与异常，不能抽掉失败 case 补新 case。

每 case 记录：
- reset_to(0) 成功 + 三层指纹（state 硬门 / fixture digest / full）
- 3 个 dummy step 后观测图像差（渲染响应 sanity）
- is_success()（LIBERO 原生判据）reset 后读数
- init 张量形状（应为 (50, n)，对齐论文每任务 50 集）
- 指令双重来源：task.language（benchmark 属性 = evaluate.py 实际馈送）vs
  BDDL :language 原文；二者差异逐 case 落盘（Sem=Ori 裁决的证据基础）

幂等：results.json 已有的 ok case 跳过（--fresh 全量重跑）。逐 case flush，
崩溃后重跑只补缺。输出 /workspace/data/l1_smoke_results.json。
"""
import argparse
import gc
import hashlib
import json
import os
import re
import sys
import time

sys.path.insert(0, "/workspace/repo/src")
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("LIBERO_CONFIG_PATH", "/workspace/data/libero_config")
from gap_repro.sim.libero_session import _ensure_runtime_env  # noqa: E402
_ensure_runtime_env()

import numpy as np  # noqa: E402

PRO = "/workspace/repo/upstream/LIBERO-PRO"
RESULTS = "/workspace/data/l1_smoke_results.json"

BASE = ["libero_goal", "libero_spatial", "libero_10", "libero_object"]
VARIANTS = {"Ori": "", "Obj": "_object", "Pos": "_swap", "Sem": "_lan",
            "Task": "_task", "Env": "_env"}


def short(h: str) -> str:
    return h[:16] if h else None


def load_results():
    if os.path.exists(RESULTS):
        return json.load(open(RESULTS))
    return {"schema": "gap_repro.lp_l1_smoke.v1",
            "cases": {}, "skipped_missing": []}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    if args.fresh and os.path.exists(RESULTS):
        os.rename(RESULTS, RESULTS + ".prev")

    out = load_results()
    done = {k for k, v in out["cases"].items() if v.get("ok")}

    from gap_repro.sim.libero_session import LiberoSession
    import torch

    for b in BASE:
        for cond, suf in VARIANTS.items():
            name = b + suf
            for i in range(10):
                key = f"{name}#{i}"
                if key in done:
                    continue
                rec = {"benchmark": name, "suite": b, "condition": cond,
                       "task_index": i}
                t0 = time.time()
                try:
                    sess = LiberoSession(suite=name, task_index=i)
                    task = sess.task
                    rec["task_name"] = task.name
                    # init 张量形状（直接读盘，独立于 benchmark 加载路径）
                    init_path = (f"{PRO}/libero/libero/init_files/"
                                 f"{name}/{task.name}.pruned_init")
                    if not os.path.isfile(init_path):
                        raise FileNotFoundError(f"init missing: {init_path}")
                    try:
                        blob = torch.load(init_path, map_location="cpu",
                                          weights_only=False)
                    except TypeError:
                        # 容器 torch 较老，无 weights_only kwarg
                        blob = torch.load(init_path, map_location="cpu")
                    # .pruned_init 内是 pickled numpy 数组（generate_init_states
                    # 的 np.array + pickle）；torch.load 旧版直接还原为 ndarray
                    if hasattr(blob, "shape") and hasattr(blob, "ndim"):
                        rec["init_count"] = int(blob.shape[0])
                        rec["init_second_dim"] = (int(blob.shape[1])
                                                  if blob.ndim == 2 else None)
                    elif hasattr(blob, "shape"):
                        rec["init_count"] = int(blob.shape[0])
                        rec["init_second_dim"] = None
                    else:
                        rec["init_count"] = len(blob)
                        rec["init_second_dim"] = None
                    # 指令双重来源
                    content = open(sess.bddl_file).read()
                    m = re.search(r"\(:language\s+(.*?)\)", content)
                    rec["instruction_benchmark"] = task.language
                    rec["instruction_bddl"] = m.group(1).strip() if m else None
                    rec["instructions_differ"] = (
                        rec["instruction_bddl"] is not None
                        and rec["instruction_bddl"] != rec["instruction_benchmark"])
                    rec["bddl_sha256"] = hashlib.sha256(
                        content.encode()).hexdigest()[:16]
                    # reset + 三层指纹
                    obs = sess.reset_to(0, wait_steps=10)
                    rec["fingerprint_state"] = short(
                        LiberoSession.fingerprint_state(obs))
                    rec["fixture_digest"] = short(sess.fixture_pose_digest())
                    rec["fingerprint_full"] = short(sess.fingerprint_full(obs))
                    rec["is_success_after_reset"] = sess.is_success()
                    # dummy steps + 渲染响应
                    dummy = np.array([0.0] * 6 + [-1.0])
                    obs2 = obs
                    for _ in range(3):
                        obs2, _, _, _ = sess.step(dummy)
                    diffs = LiberoSession.image_diff_stats(obs, obs2)
                    # image_diff_stats 返回扁平 {mean_abs, frac_gt0,
                    # frac_gt30, max_sum}（审查 P1-1：此前读不存在的嵌套
                    # "mean" 键，240 case 全 NaN——死检查已修正）
                    rec["agentview_diff_mean_abs_after_dummy"] = round(
                        float(diffs.get("mean_abs", float("nan"))), 5)
                    rec["ok"] = True
                    del sess
                except Exception as e:  # 失败如实记录，不抽 case
                    rec["ok"] = False
                    rec["error"] = f"{type(e).__name__}: {e}"
                rec["elapsed_s"] = round(time.time() - t0, 1)
                out["cases"][key] = rec
                json.dump(out, open(RESULTS, "w"), indent=1,
                          ensure_ascii=False)
                mark = "OK " if rec["ok"] else "FAIL"
                print(f"[{mark}] {key} {rec.get('error', '')}",
                      flush=True)
                gc.collect()

    # 缺失资产清点（不应有：Env 生成完成后）
    for b in BASE:
        for cond, suf in VARIANTS.items():
            name = b + suf
            for i in range(10):
                key = f"{name}#{i}"
                if key not in out["cases"]:
                    out["skipped_missing"].append(key)

    ok = sum(1 for v in out["cases"].values() if v.get("ok"))
    fail = len(out["cases"]) - ok
    per_cond = {}
    for k, v in out["cases"].items():
        c = v["condition"]
        d = per_cond.setdefault(c, {"ok": 0, "fail": 0})
        d["ok" if v.get("ok") else "fail"] += 1
    out["summary"] = {"total": len(out["cases"]), "ok": ok, "fail": fail,
                      "per_condition": per_cond}
    json.dump(out, open(RESULTS, "w"), indent=1, ensure_ascii=False)
    print(f"SMOKE_DONE ok={ok} fail={fail} "
          f"skipped_missing={len(out['skipped_missing'])}", flush=True)


if __name__ == "__main__":
    main()
