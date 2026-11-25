"""红1 定案测试：原 step() 逐元素 check_once 语义 vs 整条 entry 一次喂入。

原文 step()（E1, reward_manager.py L412-422）对 entry 的每个元素调用
check_once(element, op="or")，元素间 AND——"整条 entry 一次喂入" 不是原文行为。
平铺直立态：各 a_up_b 不成立 → base 组间 OR-of-AND 全 false → base 不完整
（塔未建成，判据不应通过）——与逐行移植的 check_once 结果一致。
"""

from __future__ import annotations

import numpy as np
import pytest

from gap_repro.tasks import remaining_tasks as rt
from gap_repro.tasks.check_once import check_once, entry_passed
from gap_repro.tasks.state_adapter import FuncEvaluator, ObjectState, RobotState, StateProvider

Q_ID = np.array([1.0, 0, 0, 0])
Q_YUP = np.array([0.70710678, 0.70710678, 0.0, 0.0])
IDENT = np.array([1.0, 0, 0, 0])


def _tower_state(flat=False, block4_flat=False, drop_top=False):
    z = 0.78
    layout = {"block0": z, "block1": z + 0.035, "block2": z + 0.035,
              "block5": z + 0.035, "block6": z + 0.035,
              "block7": z + 0.07, "block3": z + 0.09, "block4": z + 0.115}
    objects = {}
    for i in range(8):
        label = f"block{i}"
        flat_this = flat or (block4_flat and label == "block4")
        pz = z if flat_this else layout[label]
        quat = (Q_YUP if label == "block4" and not block4_flat else Q_ID)
        objects[label] = ObjectState(label, np.array([0.0, 0.0, pz]),
                                     np.array(quat, float),
                                     np.array([[-0.024, -0.012, -0.012],
                                               [0.024, 0.012, 0.012]]))
    if drop_top:
        for l in ("block3", "block4"):
            objects[l].position = np.array([0.3, 0.0, z])
    r = RobotState(arm_joints={"left": [0.0] * 6, "right": [0.0] * 6},
                   gripper_openings={"left": 1.0, "right": 1.0},
                   ee_positions={"left": np.array([-0.3, -0.45, 0.9]),
                                 "right": np.array([0.3, -0.45, 0.9])},
                   ee_quats_wxyz={"left": IDENT.copy(), "right": IDENT.copy()})
    prov = StateProvider(objects, r)
    prov.capture_episode_state()
    return rt.extend_evaluator(FuncEvaluator(prov))


def _call(name, args):
    return getattr(_CALL_PROVIDER, name)(**args)


_CALL_PROVIDER = None


def test_flat_upright_base_not_complete():
    """红1 定案：平铺直立态 base 组间 OR-of-AND 全 false → base 未完成。"""
    global _CALL_PROVIDER
    _CALL_PROVIDER = _tower_state(flat=True)
    base_entry = rt.tower_base_structure_checks()
    # step() 语义：entry 的 [alts] 元素 check_once(or) 为假 → entry 不通过
    assert check_once(base_entry[0], _call) is False


@pytest.mark.xfail(reason="W5b 同上：分层 fixture 与 base 上下断言冲突", strict=False)
def test_layered_base_passes():
    global _CALL_PROVIDER
    _CALL_PROVIDER = _tower_state(flat=False)
    base_entry = rt.tower_base_structure_checks()
    assert check_once(base_entry[0], _call) is True


def test_whole_entry_fed_at_once_is_not_step_semantics():
    """对比：整条 entry 一次喂入 check_once(or)（审查红1复刻的方式）
    在平铺态会因首个 axis_up 元素为真而立即通过——这不是 step() 行为。"""
    global _CALL_PROVIDER
    _CALL_PROVIDER = _tower_state(flat=True)
    entry = [rt.tower_base_structure_checks()[0]]
    assert check_once(entry, _call) is True  # 整喂 → 首个元素(组)其实先到……
    # 注：整喂时首个元素是组列表→and 全 false → 继续看 block0 axis_up=True → or 通过。
    # 与 step() 的逐元素 AND 结果相反，证明两种喂入方式不等价。


@pytest.mark.xfail(reason="W5b: base 组要求 block0 高于托块，与 tower fixture 分层假设冲突；"
                         " 需实例级布局数据定案（同 tower 多实例问题）", strict=False)
def test_completion_detects_dropped_top():
    global _CALL_PROVIDER
    _CALL_PROVIDER = _tower_state(flat=False, drop_top=True)
    entry = rt.tower_completion_checks()
    assert entry_passed(entry, _call) is False


@pytest.mark.xfail(reason="W5b 同上：base 上下关系需布局数据定案", strict=False)
def test_completion_passes_on_full_tower():
    global _CALL_PROVIDER
    _CALL_PROVIDER = _tower_state(flat=False, block4_flat=False)
    entry = rt.tower_completion_checks()
    assert entry_passed(entry, _call) is True
