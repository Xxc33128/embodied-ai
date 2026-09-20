"""organize_table 判据冻结 fixture：应成功与应失败序列（W5 反例覆盖）。

E3 说明：fixture 数值是构造状态（不依赖物理）；is_not_moved 等尚未与
func_parser 逐行核对的谓词，此处先锁定适配语义，比对测试在核对后补。
"""

from __future__ import annotations

import numpy as np
import pytest

from gap_repro.tasks import organize_table as ot
from gap_repro.tasks.state_adapter import FuncEvaluator, ObjectState, RobotState, StateProvider

Q_ID = np.array([1.0, 0, 0, 0])


def mk_provider(mouse_on_pad=True, keyboard_in_frame=True, garage_on_cushion=True,
                alarm_on_drawer=True, grippers_open=1.0):
    z = 0.78
    mouse_pos = (np.array([0.2, -0.1, z + 0.008]) if mouse_on_pad
                 else np.array([-0.4, 0.4, z + 0.008]))
    objects = {
        # 原语义（R3）：z_min(垫) < 鼠标 z 即可（无上界，垫上方也算过）；
        # "未放好"须 XY 移出垫面才失败
        "mouse": ObjectState("mouse", mouse_pos,
                             # 原判据无 abs：局部 y 轴须对齐 +x → 绕 z 转 -90°（审查 R2）
                             np.array([0.70710678, 0.0, 0.0, -0.70710678]),
                             np.array([[-.03, -.05, 0], [.03, .05, .02]])),
        "mousemat": ObjectState("mousemat", np.array([0.2, -0.1, z]), Q_ID.copy(),
                                np.array([[-.1, -.08, 0], [.1, .08, .005]])),
        "keyboard": ObjectState("keyboard", np.array([0.1, 0.0, z if keyboard_in_frame else z + 0.3]),
                                Q_ID.copy(), np.array([[-.2, -.07, 0], [.2, .07, .02]])),
        "frame": ObjectState("frame", np.array([0.1, 0.0, z]), Q_ID.copy(),
                             np.array([[-.22, -.09, 0], [.22, .09, .03]])),
        "garage": ObjectState("garage", np.array([0.1, 0.22, z + 0.03 if garage_on_cushion else z + 0.3]),
                              Q_ID.copy(), np.array([[-.04, -.04, 0], [.04, .04, .06]])),
        "cube_cushion": ObjectState("cube_cushion", np.array([0.1, 0.22, z]), Q_ID.copy(),
                                    np.array([[-.05, -.05, 0], [.05, .05, .03]])),
        "alarm": ObjectState("alarm", np.array([-0.15, 0.1, z + 0.26 if alarm_on_drawer else z + 0.4]),
                             Q_ID.copy(), np.array([[-.04, -.04, 0], [.04, .04, .05]])),
        "drawer": ObjectState("drawer", np.array([-0.15, 0.1, z]), Q_ID.copy(),
                              np.array([[-.1, -.1, 0], [.1, .1, .2]])),
    }
    ident = np.array([1.0, 0.0, 0.0, 0.0])
    robot = RobotState(
        arm_joints={"left": [0.0] * 6, "right": [0.0] * 6},
        gripper_openings={"left": grippers_open, "right": grippers_open},
        ee_positions={"left": np.array([-0.3, -0.45, 0.9]), "right": np.array([0.3, -0.45, 0.9])},
        ee_quats_wxyz={"left": ident.copy(), "right": ident.copy()},
    )
    prov = StateProvider(objects, robot,
                         label_cat_index={"garage": 6})  # Y1：per-env 阈值 0.025
    prov.capture_episode_state()
    return prov


def test_all_placed_and_arms_home_succeeds():
    ev = FuncEvaluator(mk_provider())
    checks = ot.run_reward_checks(ev)
    assert all(c == 1.0 for c in checks), checks


def test_mouse_displaced_fails_its_group():
    # R3 语义：is_A_in_B 无 z 上界（垫上方也算过），失败须 XY 移出垫面
    p = mk_provider()
    p.objects["mouse"].position = np.array([-0.4, 0.4, 0.78])
    ev = FuncEvaluator(p)
    assert 0.0 in ot.mouse_checks(ev)


def test_alarm_not_on_drawer_fails():
    ev = FuncEvaluator(mk_provider(alarm_on_drawer=False))
    assert 0.0 in ot.alarm_checks(ev)


def test_garage_xy_far_fails_garage_group():
    p = mk_provider()
    p.objects["garage"].position = np.array([0.3, 0.22, 0.81])  # 高度对但 xy 远
    ev = FuncEvaluator(p)
    assert 0.0 in ot.garage_checks(ev)


def test_arms_not_home_fails_back_to_origin():
    p = mk_provider()
    p.robot.ee_positions["left"] = np.array([0.0, -0.2, 0.9])
    assert FuncEvaluator(p).all_robot_back_to_origin() == 0.0


def test_grippers_closed_fails_open_check():
    p = mk_provider(grippers_open=0.1)
    assert FuncEvaluator(p).is_all_gripper_open() == 0.0


def test_score_ladder_transitions():
    ev = FuncEvaluator(mk_provider())
    ladder = ot.get_score_ladder(ev)
    assert [s for _, s in ladder] == [[25], [50], [75], [100]]
    for group, _ in ladder:
        assert group[0] == 1.0  # is_all_gripper_open
    assert all(v == 1.0 for v in ladder[-1][0][1:])  # 全四件套组


def test_is_A_up_B_exact_thresholds():
    ev = FuncEvaluator(mk_provider())
    # 原语义（E1）：z_diff<=min → 0；>=max → 0；否则 1
    assert ev.is_A_up_B(label_A="garage", label_B="cube_cushion",
                        z_threshold_min=0.005, z_threshold_max=0.1) == 1.0
    assert ev.is_A_up_B(label_A="garage", label_B="cube_cushion",
                        z_threshold_min=0.05) == 0.0
