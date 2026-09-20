"""剩余任务判据 fixture：build_tower / imitate_sorting / make_kong / fold_clothes。"""

from __future__ import annotations

import numpy as np
import pytest
import pytest

from gap_repro.tasks import remaining_tasks as rt
from gap_repro.tasks.state_adapter import FuncEvaluator, ObjectState, RobotState, StateProvider

Q_ID = np.array([1.0, 0, 0, 0])
IDENT = np.array([1.0, 0, 0, 0])


def _box(cx, cy, z, hx, hy, hz):
    return np.array([[-hx, -hy, -hz], [hx, hy, hz]])


def tower_provider(ok=True):
    z0 = 0.78
    objects = {}
    # 塔的几何：底三块在 z0，block7 在中段，block3/block4 顶部；ok=False 顶部缺失
    positions = {
        "block0": (0.0, 0.0, z0), "block1": (-0.03, 0.0, z0 + 0.03),
        "block2": (0.03, 0.0, z0 + 0.03), "block5": (0.03, 0.0, z0 + 0.03),
        "block6": (-0.03, 0.0, z0 + 0.03), "block7": (0.0, 0.0, z0 + 0.06),
        "block3": (0.0, 0.0, z0 + 0.09), "block4": (0.0, 0.0, z0 + 0.115),
    }
    for i in range(8):
        label = f"block{i}"
        px, py, pz = positions[label]
        if not ok and label in ("block3", "block4"):
            px, py, pz = px + 0.3, py, pz  # 顶部两块放偏
        # block4 判据用局部 y 轴向天（5°）→ 绕 x 转 90°（审查 R1 同源语义）
        quat = np.array([0.70710678, 0.70710678, 0.0, 0.0]) if label == "block4" else Q_ID.copy()
        objects[label] = ObjectState(label, np.array([px, py, pz]), quat,
                                     _box(px, py, pz, 0.024, 0.012, 0.012))
    prov = StateProvider(objects, RobotState(
        arm_joints={"left": [0.0] * 6, "right": [0.0] * 6},
        gripper_openings={"left": 1.0, "right": 1.0},
        ee_positions={"left": np.array([-0.3, -0.45, 0.9]),
                      "right": np.array([0.3, -0.45, 0.9])},
        ee_quats_wxyz={"left": IDENT.copy(), "right": IDENT.copy()}))
    prov.capture_episode_state()
    return rt.extend_evaluator(FuncEvaluator(prov))


def _tower_layer_fixture(z_table, z_block0, z_support, z_block73, z_block4):
    """分层布置（W5b：base 与 middle 对 block0/block1 上下互斥 → 原判据隐含
    同名多实例；分层 fixture 在单层内验证语义，不虚构实例）。"""
    z = 0.78
    layout = {"block0": z_block0, "block1": z_support, "block2": z_support,
              "block5": z_support, "block6": z_support, "block7": z_block73,
              "block3": z_block73 + 0.02, "block4": z_block4}
    objects = {}
    for i in range(8):
        label = f"block{i}"
        quat = (np.array([0.70710678, 0.70710678, 0.0, 0.0]) if label == "block4"
                else Q_ID.copy())
        pz = layout[label]
        hx, hy, hz = ((0.024, 0.012, 0.012) if label != "block4"
                      else (0.012, 0.024, 0.012))  # block4 局部 y 向上 → 盒沿 y 短
        objects[label] = ObjectState(label, np.array([0.0, 0.0, pz]), quat,
                                     _box(0.0, 0.0, pz, hx, hy, hz))
    prov = StateProvider(objects, RobotState(
        arm_joints={"left": [0.0] * 6, "right": [0.0] * 6},
        gripper_openings={"left": 1.0, "right": 1.0},
        ee_positions={"left": np.array([-0.3, -0.45, 0.9]),
                      "right": np.array([0.3, -0.45, 0.9])},
        ee_quats_wxyz={"left": IDENT.copy(), "right": IDENT.copy()}))
    prov.capture_episode_state()
    return rt.extend_evaluator(FuncEvaluator(prov))


@pytest.mark.xfail(reason="W5b: tower base/middle 同名多实例上下关系待布局数据定案（镜像已转节点制）", strict=False)
def test_tower_complete_succeeds():
    # W5b：base 与 middle 语义在单实例下互斥 → 分别 fixture 验证各层语义；
    # 实例多义性待布局数据核对。
    ev_base = _tower_layer_fixture(0.78, 0.82, 0.78, 0.9, 0.93)   # block0 高于托块
    assert rt.eval_checks(rt.tower_base_structure_checks(ev_base)) == 1.0
    ev_mid = _tower_layer_fixture(0.78, 0.78, 0.815, 0.85, 0.88)  # 托块高于 block0
    assert rt.eval_checks(rt.tower_middle_structure_checks(ev_mid)) == 1.0
    ev_top = _tower_layer_fixture(0.78, 0.78, 0.815, 0.85, 0.9)  # top 相对 block7/block3
    assert rt.eval_checks(rt.tower_top_structure_checks(ev_top)) == 1.0


@pytest.mark.xfail(reason="W5b: tower base/middle 同名多实例上下关系待布局数据定案（镜像已转节点制）", strict=False)
def test_tower_top_missing_fails():
    ev = tower_provider(ok=False)
    def flat(v):
        if isinstance(v, (list, tuple)):
            return [x for s in v for x in flat(s)]
        return [float(v)]
    assert rt.eval_checks(rt.tower_run_reward(ev)) == 0.0


@pytest.mark.xfail(reason="W5b: tower base/middle 同名多实例上下关系待布局数据定案（镜像已转节点制）", strict=False)
def test_tower_score_stages_structure():
    ev = _tower_layer_fixture(0.78, 0.81, 0.85, 0.88, 0.91)
    stages = rt.tower_score_stages(ev)
    assert [s for _, s in stages] == [[10], [30], [100]]


def sorting_provider(stage_done=5, home=True):
    z = 0.78
    objects = {}
    for i in range(5):
        inside = i < stage_done
        pos = (np.array([0.0, 0.3, z + 0.02]) if inside
               else np.array([-0.4 + 0.05 * i, -0.3, z + 0.01]))
        objects[f"target_{i}"] = ObjectState(f"target_{i}", pos, Q_ID.copy(),
                                             _box(pos[0], pos[1], pos[2], 0.02, 0.02, 0.01))
    for i in range(5):
        pos = np.array([0.1, 0.3, z + 0.02])
        objects[f"aim{i}"] = ObjectState(f"aim{i}", pos, Q_ID.copy(),
                                         _box(pos[0], pos[1], pos[2], 0.02, 0.02, 0.01))
    objects["basket0"] = ObjectState("basket0", np.array([0.0, 0.3, z]), Q_ID.copy(),
                                     _box(0.0, 0.3, z, 0.1, 0.08, 0.05))
    objects["basket1"] = ObjectState("basket1", np.array([0.1, 0.3, z]), Q_ID.copy(),
                                     _box(0.1, 0.3, z, 0.1, 0.08, 0.05))
    prov = StateProvider(objects, RobotState(
        arm_joints={"left": [0.0] * 6, "right": [0.0] * 6},
        gripper_openings={"left": 1.0, "right": 1.0},
        ee_positions={"left": np.array([-0.3, -0.45, 0.9]),
                      "right": np.array([0.3, -0.45, 0.9])},
        ee_quats_wxyz={"left": IDENT.copy(), "right": IDENT.copy()}))
    prov.capture_episode_state()  # 基准在归位位姿捕获
    if not home:
        prov.robot.ee_positions["left"] = prov.robot.ee_positions["left"] + np.array([0.5, 0, 0])
    return rt.extend_evaluator(FuncEvaluator(prov))


def test_sorting_full_stage_succeeds():
    ev = sorting_provider(stage_done=5, home=True)
    stages = rt.sorting_run_reward(ev, [f"target_{i}" for i in range(5)],
                                   [f"aim{i}" for i in range(5)])
    assert all(c == 1.0 for c in stages[-1])


def test_sorting_partial_stage_boundary():
    ev = sorting_provider(stage_done=2, home=True)
    stages = rt.sorting_run_reward(ev, [f"target_{i}" for i in range(5)],
                                   [f"aim{i}" for i in range(5)])
    assert all(c == 1.0 for c in stages[1])  # stage1 完成
    assert any(c == 0.0 for c in stages[4])  # stage4 未完成


def test_sorting_not_home_blocks_final_stage():
    ev = sorting_provider(stage_done=5, home=False)
    stages = rt.sorting_run_reward(ev, [f"target_{i}" for i in range(5)],
                                   [f"aim{i}" for i in range(5)])
    assert stages[-1][-1] == 0.0  # all_robot_back_to_origin


def test_sorting_ladder():
    ev = sorting_provider(stage_done=5, home=True)
    ladder = rt.sorting_score_ladder(ev, [f"target_{i}" for i in range(5)],
                                     [f"aim{i}" for i in range(5)])
    assert [s for _, s in ladder] == [5, 15, 30, 50, 100]


def kong_provider(ok=True):
    z = 0.78
    objects = {}
    pos_map = {}
    n = 0
    for push, tiles in rt.MAKE_KONG_TARGET_MAP.items():
        for t in tiles:
            pos_map[t] = (0.2 + 0.02 * n, 0.1, z)
            n += 1
    y_up = np.array([0.70710678, 0.70710678, 0.0, 0.0])  # 局部 y 轴向天（其他牌立起 7° 检查）
    for label, (px, py, pz) in pos_map.items():
        is_target = label.startswith("mahjong0")
        tilted = (not ok) and is_target and label == "mahjong0_1"
        quat = Q_ID.copy() if (is_target and not tilted) else (Q_ID.copy() if not is_target else y_up)
        if not is_target:
            quat = y_up  # 非目标牌立起：local y → world z（7° 检查）
        if tilted:
            quat = Q_ID.copy() * 0 + np.array([0.70710678, 0.70710678, 0.0, 0.0])  # 目标牌歪倒
        objects[label] = ObjectState(label, np.array([px, py, pz]), quat,
                                     _box(px, py, pz, 0.012, 0.01, 0.018))
    pos9 = np.array([0.319, -0.15, z + 0.02]) if ok else np.array([0.35, -0.15, z + 0.02])
    objects["mahjong9_0"] = ObjectState(
        "mahjong9_0", pos9, np.array([0.0, 0.70710678, 0.70710678, 0.0]),
        _box(pos9[0], pos9[1], pos9[2], 0.012, 0.01, 0.018))
    prov = StateProvider(objects, RobotState(
        arm_joints={"left": [0.0] * 6, "right": [0.0] * 6},
        gripper_openings={"left": 1.0, "right": 1.0},
        ee_positions={"left": np.array([-0.3, -0.45, 0.9]),
                      "right": np.array([0.3, -0.45, 0.9])},
        ee_quats_wxyz={"left": IDENT.copy(), "right": IDENT.copy()}))
    prov.capture_episode_state()
    return rt.extend_evaluator(FuncEvaluator(prov))


def test_kong_checks_pass_when_arranged():
    ev = kong_provider(ok=True)
    checks = rt.make_kong_run_reward(ev, push="mahjong5_0")
    assert all(rt.eval_checks(g) == 1.0 for g in [checks[0][:12], checks[1][:12]])
    # W5b：原文 mahjong9 的 qpos 目标 [0,0.707,0.707,0] 与 axis_up(y,7°) 对同实例
    # 不可同真（该四元数下 y 轴指向 +x）——待原环境核对语义
    pytest.xfail("mahjong9 qpos/axis-up 语义冲突待 W5b 核对")


def test_kong_tilted_other_tile_fails():
    ev = kong_provider(ok=False)
    checks = rt.make_kong_run_reward(ev, push="mahjong5_0")
    assert any(rt.eval_checks(group) == 0.0 for group in checks)


def test_fold_clothes_explicitly_blocked():
    ev = sorting_provider(stage_done=1)
    with pytest.raises(NotImplementedError, match="W3"):
        rt.fold_clothes_run_reward(ev)
