#!/usr/bin/env python3
"""LP-A1 P1 解除实验：fixture 漂移量化 + 图像差异归因隔离（gap-sim 执行）。

审查（2026-09-18-LP-A1对抗性审查.md P1-1）要求的一次性只读对照：
  (a) 多次 reset 各记录 fixture 根 body 位姿 → 证明"每次 reset 重采样"并量化幅度；
  (b) 同一物理状态连续渲染 2 次做 image_diff（两个相机）→ 纯渲染器噪声基线；
  (c) 跨 reset 同 init 图像 diff → 渲染器+fixture 漂移的混合项。
归因判据（本脚本预注册，诊断性阈值，非正式统计门）：
  同状态双渲染全部满足 mean_abs ≤ 1.0 且 frac_gt30 ≤ 0.01 → 判"渲染器逐位稳定"，
  跨 reset 图像差主要归因 fixture 漂移；违反则归因改写为渲染器逐次非确定性。
协议：与 v1 证据（lp_a1_reset_fingerprint.py）同构——libero_goal task0 × init{0,1}
× 3 次独立 reset，组内穿插 3 个 dummy 步。movables 硬门指纹算法未变，
本实验同时复现 v1 的 7013dce1…/9fe5fae0… 作为交叉锚。
"""
import json
import sys

sys.path.insert(0, "/workspace/repo/src")

import numpy as np

from gap_repro.sim.libero_session import LiberoSession

CAMERA_KEYS = ("agentview_image", "robot0_eye_in_hand_image")
# 预注册（诊断性）：同状态双渲染视为"渲染器逐位稳定"的阈值
RENDERER_STABLE_MEAN_ABS = 1.0
RENDERER_STABLE_FRAC_GT30 = 0.01

session = LiberoSession(suite="libero_goal", task_index=0, seed=0)

regions = session.env.env.parsed_problem.get("regions", {})
records = []
cross_by_init = {0: {k: [] for k in CAMERA_KEYS},
                 1: {k: [] for k in CAMERA_KEYS}}
obs_prev = None
for init in (0, 1):
    for rep in range(3):
        obs = session.reset_to(init)
        # (b) 同一物理状态连续重渲染（不 step、且在穿插 dummy 步之前捕获）：
        # 纯渲染器项。obs 与 obs_rerender 必须取自同一物理状态，
        # 否则混入 dummy 步的物理/视觉变化（首版探针曾犯此错，已修正）。
        obs_rerender = session.env.env._get_observations(force_update=True)
        state_fp = session.fingerprint_state(obs)
        poses = session.fixture_poses()
        scene_fp = session.fixture_pose_digest()
        cams = [k for k in CAMERA_KEYS if k in obs and k in obs_rerender]
        same_state_diff = {k: session.image_diff_stats(obs, obs_rerender, key=k)
                           for k in cams}
        # 穿插 3 个 dummy 步（与 v1 协议同构：模拟"任意历史后重置"；
        # 不影响本 rep 已捕获的各项，只推进随机流与物理历史）
        for _ in range(3):
            session.step(np.array([0.0] * 6 + [-1.0]))
        if rep >= 1:
            for k in cams:
                if k in obs_prev:
                    cross_by_init[init][k].append(
                        session.image_diff_stats(obs_prev, obs, key=k))
        obs_prev = obs
        records.append({
            "init": init, "rep": rep,
            "state_fingerprint": state_fp,
            "fixture_poses": poses,
            "scene_fingerprint": scene_fp,
            "fingerprint_full": session.fingerprint_full(obs),
            "same_state_render_diff": same_state_diff,
        })
        print(f"init={init} rep={rep} st={state_fp[:12]} scene={scene_fp[:12]} "
              f"same_render_mean={ {k: v['mean_abs'] for k, v in same_state_diff.items()} }",
              flush=True)


# ---- 聚合与判定 ----
def diff_agg(diffs):
    if not diffs:
        return None
    return {"n": len(diffs),
            "mean_abs_max": max(d["mean_abs"] for d in diffs),
            "frac_gt0_max": max(d["frac_gt0"] for d in diffs),
            "frac_gt30_max": max(d["frac_gt30"] for d in diffs),
            "mean_abs_mean": round(float(np.mean([d["mean_abs"] for d in diffs])), 4),
            "frac_gt0_mean": round(float(np.mean([d["frac_gt0"] for d in diffs])), 4),
            "frac_gt30_mean": round(float(np.mean([d["frac_gt30"] for d in diffs])), 4)}


all_same = [r["same_state_render_diff"][k] for r in records
            for k in r["same_state_render_diff"]]
renderer_stable = all(d["mean_abs"] <= RENDERER_STABLE_MEAN_ABS
                      and d["frac_gt30"] <= RENDERER_STABLE_FRAC_GT30
                      for d in all_same)

per_init = []
for init in (0, 1):
    rs = [r for r in records if r["init"] == init]
    state_fps = {r["state_fingerprint"] for r in rs}
    scene_fps = {r["scene_fingerprint"] for r in rs}
    # fixture 漂移幅度：各 fixture 相对 rep0 的逐轴最大差
    drift = {}
    for name in rs[0]["fixture_poses"]:
        base = np.array(rs[0]["fixture_poses"][name]["body_pos"])
        deltas = np.array([np.abs(np.array(r["fixture_poses"][name]["body_pos"]) - base)
                           for r in rs[1:]])
        drift[name] = {"max_abs_delta_pos_per_axis":
                       [round(float(v), 6) for v in deltas.max(axis=0)]}
    per_init.append({
        "init": init,
        "state_fingerprints_distinct": len(state_fps),
        "movables_consistent": len(state_fps) == 1,
        "state_fingerprint": sorted(state_fps)[0],
        "scene_fingerprints_distinct": len(scene_fps),
        "scene_fingerprints": sorted(scene_fps),
        "fixture_drift_vs_rep0": drift,
        "same_state_render_diff_agg": {k: diff_agg(
            [r["same_state_render_diff"][k] for r in rs if k in r["same_state_render_diff"]])
            for k in CAMERA_KEYS},
        "cross_reset_render_diff_agg": {k: diff_agg(cross_by_init[init][k])
                                        for k in CAMERA_KEYS},
    })

movables_ok = all(a["movables_consistent"] for a in per_init)
scene_drifts = all(a["scene_fingerprints_distinct"] == 3 for a in per_init)
out = {
    "schema": "gap_repro.lp_a1_fixture_drift.v1",
    "suite": "libero_goal", "task_index": 0, "seed": 0,
    "preregistered_renderer_stable_threshold": {
        "mean_abs": RENDERER_STABLE_MEAN_ABS, "frac_gt30": RENDERER_STABLE_FRAC_GT30},
    "bddl_regions": regions,
    "verdicts": {
        "movables_hard_gate_consistent": movables_ok,
        "scene_fingerprint_distinct_per_reset": scene_drifts,
        "renderer_bit_stable_same_state": renderer_stable,
        "attribution": ("scene_resampling_dominates" if renderer_stable else
                        "renderer_nondeterminism_present"),
    },
    "records": records,
    "aggregates": {"per_init": per_init},
    "note": "scene 指纹 reset 间不同=协议内在重采样（BDDL region 内均匀采样，"
            "幅度上界=region 边长）；图像差归因由同状态双渲染(纯渲染器)与"
            "跨 reset(渲染器+fixture) 两组 diff 分离",
}
json.dump(out, open("/workspace/data/lp_a1_fixture_drift.json", "w"), indent=1)
print(f"PROBE_DONE movables_ok={movables_ok} scene_drifts={scene_drifts} "
      f"renderer_stable={renderer_stable} attribution={out['verdicts']['attribution']}",
      flush=True)
