#!/usr/bin/env python3
"""LP-A2 自测：reset_to 的 fixture 钉扎路径（scene_pinned 臂上线前干跑）。"""
import sys

sys.path.insert(0, "/workspace/repo/src")

from gap_repro.sim.libero_session import LiberoSession

s = LiberoSession(suite="libero_goal", task_index=0, seed=0)
obs = s.reset_to(0)
poses = s.fixture_poses()
# 给 stove 一个明显偏移作为钉扎目标，验证 reset 后真的被钉住
pinned = {k: {"body_pos": list(v["body_pos"])} for k, v in poses.items()}
pinned["flat_stove_1"]["body_pos"][0] += 0.005
obs2 = s.reset_to(0, fixture_poses=pinned)
after = s.fixture_poses()
delta = after["flat_stove_1"]["body_pos"][0] - pinned["flat_stove_1"]["body_pos"][0]
assert abs(delta) < 1e-9, f"钉扎未生效: delta={delta}"
others = all(after[k]["body_pos"] == pinned[k]["body_pos"] for k in pinned)
assert others, "其余 fixture 钉扎不一致"
print("PIN_SELFTEST_OK delta=%.6f digest=%s" % (delta, s.fixture_pose_digest()[:12]))
s.env.close()
