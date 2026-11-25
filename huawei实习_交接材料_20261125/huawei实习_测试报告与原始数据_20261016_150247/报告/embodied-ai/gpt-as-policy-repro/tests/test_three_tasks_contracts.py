"""三任务判据 fixture：应成功与应失败序列（W5 扩展）。

对象几何为构造值；谓词语义已按 func_parser 原文实现（E1）。
"""

from __future__ import annotations

import numpy as np

from gap_repro.tasks import three_simple_tasks as tst
from gap_repro.tasks.state_adapter import FuncEvaluator, ObjectState, RobotState, StateProvider

Q_ID = np.array([1.0, 0, 0, 0])
IDENT = np.array([1.0, 0, 0, 0])


def _box(cx, cy, z, hx, hy, hz):
    """局部盒（原点居中，min/max 两角）；中心位置由 ObjectState.position 给出。"""
    return np.array([[-hx, -hy, -hz], [hx, hy, hz]])


def _robot(home=(0.0, 0.0)):
    return RobotState(
        arm_joints={"left": [0.0] * 6, "right": [0.0] * 6},
        gripper_openings={"left": 1.0, "right": 1.0},
        ee_positions={"left": np.array([-0.3 + home[0], -0.45, 0.9]),
                      "right": np.array([0.3 + home[0], -0.45, 0.9])},
        ee_quats_wxyz={"left": IDENT.copy(), "right": IDENT.copy()},
    )


def dustbin_provider(ok=True, home=True):
    z = 0.78
    objects = {"dustbin": ObjectState("dustbin", np.array([0.0, -0.2, z]), Q_ID.copy(),
                                      _box(0.0, -0.2, z + 0.1, 0.15, 0.1, 0.1))}
    for i in range(4):
        inside = ok or i >= 2  # ok=False 时两瓶留在桌远处
        # 瓶在桶内：z 落入桶体 z 区间（is_A_cover_B 要求 b_zmax < a_zmax）
        pos = (np.array([0.0 + 0.05 * (i % 2), -0.2 + 0.04 * (i // 2), z + 0.04])
               if inside else np.array([-0.45 + 0.05 * i, 0.35, z + 0.005]))
        objects[f"bottle_{i}"] = ObjectState(f"bottle_{i}", pos, Q_ID.copy(),
                                             _box(pos[0], pos[1], pos[2], 0.03, 0.03, 0.04))
    r = _robot((0, 0) if home else (0.5, 0))
    prov = StateProvider(objects, r)
    prov.capture_episode_state()
    return FuncEvaluator(prov)


def classify_provider(ok=True):
    z = 0.78
    objects = {}
    for b in range(3):
        objects[f"basket{b}"] = ObjectState(f"basket{b}", np.array([0.3 * (b - 1), 0.3, z]),
                                            Q_ID.copy(), _box(0.3 * (b - 1), 0.3, z, 0.12, 0.08, 0.06))
    for c in range(3):
        for j in range(2):
            label = f"cat{c}_obj{j}"
            pos = np.array([0.3 * (c - 1), 0.3, z + 0.05])
            if not ok and (c, j) == (0, 1):
                pos = np.array([0.0, 0.3, z + 0.05])  # cat0 的一个物体错入 basket1
            objects[label] = ObjectState(label, pos, Q_ID.copy(),
                                         _box(pos[0], pos[1], pos[2], 0.03, 0.03, 0.03))
    prov = StateProvider(objects, _robot())
    prov.capture_episode_state()
    return FuncEvaluator(prov)


def arrange_provider(ok=True):
    z = 0.78
    objects = {}
    digit_values = [3, 1, 4, 2] if ok else [1, 3, 4, 2]  # ok=False：digit_0 错位
    cat_map = {f"digit_{i}": dv for i, dv in enumerate(digit_values)}
    # 名次（按数值降序）决定应上哪个 mat：rank0 → mat_0
    order = sorted(range(4), key=lambda i: digit_values[i], reverse=True)
    for i, dv in enumerate(digit_values):
        label = f"digit_{i}"
        rank = order.index(i)
        pos = (np.array([0.1 * rank, -0.2, z + 0.005]) if ok or i != 0
               else np.array([0.35, -0.2, z + 0.005]))  # ok=False：digit_0 不在 mat_3
        objects[label] = ObjectState(label, pos, Q_ID.copy(),
                                     _box(pos[0], pos[1], pos[2], 0.03, 0.03, 0.005))
    for m in range(4):
        objects[f"mat_{m}"] = ObjectState(f"mat_{m}", np.array([0.1 * m, -0.2, z]),
                                          Q_ID.copy(), _box(0.1 * m, -0.2, z, 0.04, 0.04, 0.003))
    prov = StateProvider(objects, _robot())
    prov.label_cat_index = cat_map
    prov.capture_episode_state()
    return FuncEvaluator(prov)


def _flat(values):
    out = []
    for v in values:
        if isinstance(v, (list, tuple)):
            out.extend(_flat(v))
        else:
            out.append(float(v))
    return out


def test_dustbin_all_in_bin_succeeds():
    ev = dustbin_provider(ok=True)
    assert all(c == 1.0 for c in _flat(tst.dustbin_run_reward(ev)))


def test_dustbin_two_bottles_out_fails():
    ev = dustbin_provider(ok=False)
    assert 0.0 in _flat(tst.dustbin_run_reward(ev))


def test_dustbin_ladder_scores():
    ev = dustbin_provider(ok=True)
    assert [s for _, s in tst.dustbin_score_ladder(ev)] == [[10], [25], [40], [100]]


def test_classify_all_sorted_succeeds():
    # 原文注册的是 3×3 判据网格：对角（cat_i 全在 basket_i）为真、非对角为假。
    # 组内 success 规则属 reward_manager 内部逻辑，待 W5b 核对（Y5 遗留）。
    ev = classify_provider(ok=True)
    grid = [row[:] for row in [list(map(float, r)) for r in
            [grp[:3] for grp in [tst.classify_run_reward(ev)[0:3]]]][0]] if False else None
    raw = tst.classify_run_reward(ev)
    diag_ok = (raw[0][0] == 1.0 and raw[1][1] == 1.0 and raw[2][2] == 1.0)
    assert diag_ok
    assert all(v == 1.0 for v in _flat(raw[3:]))  # settled + back


def test_classify_wrong_basket_fails():
    ev = classify_provider(ok=False)
    raw = tst.classify_run_reward(ev)
    diag_ok = (raw[0][0] == 1.0 and raw[1][1] == 1.0 and raw[2][2] == 1.0)
    assert not diag_ok


def test_arrange_sorted_desc_succeeds():
    ev = arrange_provider(ok=True)
    vals = _flat(tst.arrange_run_reward(ev))
    assert all(v == 1.0 for v in vals), vals


def test_arrange_wrong_order_fails():
    ev = arrange_provider(ok=False)
    assert 0.0 in _flat(tst.arrange_run_reward(ev))


def test_arrange_ladder_structure():
    ev = arrange_provider(ok=True)
    ladder4 = tst.arrange_score_ladder(ev, n_digits=4)
    assert [s for _, s in ladder4] == [[5], [15], [30], [100]]
    ladder5 = tst.arrange_score_ladder(ev, n_digits=5)
    assert [s for _, s in ladder5] == [[5], [15], [25], [40], [100]]
