#!/usr/bin/env python3
"""LP-A2 配对比较：torch_cpu vs torch_npu 闭环 episode 对账（本地或 gap-sim 均可跑）。

按 (arm, init) 配对：首块动作差（pinned 臂对阈值 0.03/0.005 判定）、
块 sha 相等数、发散步（首个 ||Δexecuted||_inf > 0.1）、成功率相等性。
输出 JSON（预注册见 configs/lp_a2_validation.json）。
用法: lp_a2_compare.py <ep_dir> <out_json>
"""
import glob
import json
import pathlib
import sys

import numpy as np

EP_DIR = pathlib.Path(sys.argv[1])
OUT = sys.argv[2]
DIVERGENCE_TOL = 0.1  # 描述性发散判据（预注册于 validation.json paper_faithful_arm 条目）
PIN_MAX, PIN_MEAN = 0.03, 0.005  # scene_pinned 首块阈值（继承 LP-A0 带）


def load_ep(arm, device, init):
    p = EP_DIR / f"ep_{arm}_{device}_init{init}.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    d["executed"] = np.load(EP_DIR / f"ep_{arm}_{device}_init{init}.npz")["executed"]
    return d


def first_chunk(d):
    return np.asarray(d["chunks"][0]["actions"], dtype=np.float32)


def diff_stats(a, b):
    d = np.abs(a - b)
    return {"max_abs": round(float(d.max()), 6),
            "mean_abs": round(float(d.mean()), 6)}


def divergence_step(ca, cb):
    n = min(len(ca), len(cb))
    for t in range(n):
        if float(np.abs(ca[t] - cb[t]).max()) > DIVERGENCE_TOL:
            return t
    return None


pairs = []
for arm in ("scene_pinned", "paper_faithful"):
    inits = sorted({int(p.split("_init")[1][0]) for p in
                    glob.glob(str(EP_DIR / f"ep_{arm}_*_init*.json"))})
    for init in inits:
        ca, cb = load_ep(arm, "torch_cpu", init), load_ep(arm, "torch_npu", init)
        if ca is None or cb is None:
            continue
        entry = {
            "arm": arm, "init": init,
            "success_cpu": ca["success"], "success_npu": cb["success"],
            "done_step_cpu": ca["done_step"], "done_step_npu": cb["done_step"],
            "success_equal": ca["success"] == cb["success"],
            "first_chunk_diff": diff_stats(first_chunk(ca), first_chunk(cb)),
            "chunk_sha_equal_count": sum(
                c["server"]["actions_sha"] == d["server"]["actions_sha"]
                for c, d in zip(ca["chunks"], cb["chunks"])),
            "chunk_count_cpu": ca["n_replans"], "chunk_count_npu": cb["n_replans"],
            "divergence_step": divergence_step(ca["executed"], cb["executed"]),
            "scene_fingerprint_cpu": ca["start_identity"]["fingerprint_full"],
            "scene_fingerprint_npu": cb["start_identity"]["fingerprint_full"],
            "movables_fingerprint_equal":
                ca["start_identity"]["fingerprint_state"]
                == cb["start_identity"]["fingerprint_state"],
        }
        if arm == "scene_pinned":
            fd = entry["first_chunk_diff"]
            entry["threshold_verdict"] = (
                "PASS" if fd["max_abs"] <= PIN_MAX and fd["mean_abs"] <= PIN_MEAN
                else "FAIL")
        pairs.append(entry)

out = {
    "schema": "gap_repro.lp_a2_compare.v1",
    "pairs": pairs,
    "thresholds": {"scene_pinned_first_chunk": {"max_abs": PIN_MAX, "mean_abs": PIN_MEAN},
                   "divergence_tol_inf": DIVERGENCE_TOL},
    "verdict": {
        "pinned_first_chunks": all(p.get("threshold_verdict") == "PASS" for p in pairs
                                   if p["arm"] == "scene_pinned"),
        "success_flip_cases": [f"{p['arm']} init{p['init']}"
                               for p in pairs if not p["success_equal"]],
    },
}
json.dump(out, open(OUT, "w"), indent=1)
print(f"COMPARE_DONE pairs={len(pairs)} "
      f"pinned_pass={out['verdict']['pinned_first_chunks']} "
      f"flips={out['verdict']['success_flip_cases']}", flush=True)
for p in pairs:
    print(json.dumps(p, ensure_ascii=False), flush=True)
