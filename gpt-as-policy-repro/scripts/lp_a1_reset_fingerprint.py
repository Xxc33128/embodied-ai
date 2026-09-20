#!/usr/bin/env python3
"""LP-A1 验收（v1）：reset 指纹一致性（服务器 gap-sim 执行，产出 JSON 证据）。

协议：LIBERO Goal task0 × init states {0,1} × 每组 3 次独立 reset
（每次 reset 前穿插 3 个 dummy 步，模拟"任意历史后重置"）。
判据：同 (task, init) 的全部指纹一致；不同 init 指纹不同。

v1 存档（md5 580812e5…）后取代说明：本脚本的状态硬门只覆盖 robot+movables，
对 fixture 漂移盲（审查 P1）；P1 修复后的完整协议见 lp_a1_fixture_drift_probe.py
（fixture 位姿入指纹 + 同状态双渲染归因）。本脚本的图像容差推导：
mean≤5.0 ≈ 观测极值 3.41×1.5、frac_gt30≤0.10 ≈ 观测极值 6.98%×1.4
（n=4 开发诊断样本，非正式统计门）。
"""
import json
import sys

sys.path.insert(0, "/workspace/repo/src")

import numpy as np

from gap_repro.sim.libero_session import LiberoSession

session = LiberoSession(suite="libero_goal", task_index=0, seed=0)
records = []
for init in (0, 1):
    sts, img_stats = [], []
    for rep in range(3):
        obs = session.reset_to(init)
        for _ in range(3):
            session.step(np.array([0.0] * 6 + [-1.0]))
        sts.append(session.fingerprint_state(obs))
        if rep >= 1:
            img_stats.append(session.image_diff_stats(obs_prev, obs))
        obs_prev = obs
    consistent = len(set(sts)) == 1
    records.append({"init": init, "state_fingerprints": sts,
                    "consistent": consistent,
                    "image_diff_stats": img_stats})
    print(f"init={init} consistent={consistent} st={sts[0][:16]}", flush=True)

cross = records[0]["state_fingerprints"][0] != records[1]["state_fingerprints"][0]
img_ok = all(s["mean_abs"] <= 5.0 and s["frac_gt30"] <= 0.10
             for r in records for s in r["image_diff_stats"])
out = {"schema": "gap_repro.lp_a1_fingerprint.v1", "suite": "libero_goal",
       "task_index": 0, "records": records,
       "reset_consistency": all(r["consistent"] for r in records),
       "init_pairing_discriminates": cross,
       "render_image_tolerance_ok": img_ok,
       "note": "状态指纹=硬门（逐位）；图像 OSMesa 逐次微小非确定性以容差记录"}
json.dump(out, open("/workspace/data/lp_a1_fingerprint.json", "w"), indent=1)
print(f"FINGERPRINT_DONE consistency={out['reset_consistency']} "
      f"pairing_discriminates={cross} img_tol_ok={img_ok}", flush=True)
