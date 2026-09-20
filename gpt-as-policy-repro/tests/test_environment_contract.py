"""W5 环境契约测试：session 表面在 MuJoCo 上的行为（容器内运行）。

对象状态经 register_layout_objects 注入冻结 fixture；机器人侧真实物理。
无资产环境显式 BLOCKED（不得静默 skip）。
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

URDF = Path(os.environ.get("GAP_REPRO_DATA", "")) / \
    "hf_cache/Assets/Robots/x5/X5A.urdf"


def _require_env():
    import mujoco  # noqa: F401
    if not URDF.is_file():
        pytest.fail(f"BLOCKED: X5A.urdf 不存在（{URDF}），需资产环境", pytrace=False)


@pytest.fixture(scope="module")
def env():
    _require_env()
    from gap_repro.sim.environment import MuJoCoEnvironment
    from test_task_contracts import mk_provider
    e = MuJoCoEnvironment(URDF, task="organize_table")
    prov = mk_provider()  # 成功态对象布置
    e.register_layout_objects(prov.objects)
    e.instruction = "organize the table"
    e.reset()
    return e


def test_fail_closed_empty_criteria(env):
    from gap_repro.sim.environment import RewardManagerView
    rm = RewardManagerView(1)
    with pytest.raises(RuntimeError, match="vacuous success"):
        rm.require_non_empty()


def test_reset_clears_counters_and_seed_layout(env):
    env.seed_manager.add_case(0, "Assets/Eval_Layout/RoboDojo/arx_x5/0/organize_table_0.json")
    env.reset(seed=0)
    assert env.take_action_cnt[0] == 0
    assert env.end_flag[0] is False
    assert env.step_id == 0
    assert env.seed_manager.seed_info[0]["scene_layout"].endswith("organize_table_0.json")


def test_take_action_cnt_exactly_once(env):
    env.reset()
    before = env.take_action_cnt[0]
    cmd = {}
    for arm in ("left", "right"):
        cmd[f"{arm}_arm_joint_state"] = {"position": [0.0] * 6}
        cmd[f"{arm}_ee_joint_state"] = {"position": [1.0]}
    env.take_action(cmd)
    assert env.take_action_cnt[0] == before + 1


def _home_cmd(opening=1.0):
    cmd = {}
    for arm in ("left", "right"):
        cmd[f"{arm}_arm_joint_state"] = {"position": [0.0] * 6}
        cmd[f"{arm}_ee_joint_state"] = {"position": [opening]}
    return cmd


def test_take_action_validates_contract(env):
    env.reset()
    bad_len = _home_cmd()
    bad_len["left_arm_joint_state"]["position"] = [0.0] * 5
    with pytest.raises(ValueError, match="6 joint"):
        env.take_action(bad_len)
    bad_open = _home_cmd()
    bad_open["left_ee_joint_state"]["position"] = [1.5]
    with pytest.raises(ValueError, match="out of"):
        env.take_action(bad_open)


def test_success_reached_and_score_100(env):
    env.reset()
    cmd = _home_cmd(1.0)
    steps = 0
    while not env.end_flag[0] and steps < env.step_lim:
        env.take_action(cmd)
        steps += 1
    assert env.success[0] is True
    assert env.end_flag[0] is True
    env.get_score()
    assert env.get_score()[0] == 100.0
    assert steps < env.step_lim


def test_truncation_at_step_lim(env):
    # success 由注册判据决定（夹爪开合只在计分阶梯）；截断需用失败 fixture 阻止成功
    from test_task_contracts import mk_provider
    env.register_layout_objects(mk_provider(mouse_on_pad=False).objects)
    env.reset()
    env.step_lim = 3
    cmd = _home_cmd(1.0)
    for _ in range(3):
        env.take_action(cmd)
    assert env.end_flag[0] is True
    assert env.success[0] is False
    env.step_lim = 1000
    env.register_layout_objects(mk_provider().objects)  # 还原成功态供后续测试


def test_obs_contract(env):
    env.reset()
    obs = env.get_obs()
    assert obs["states"].shape == (14,) and np.isfinite(obs["states"]).all()
    assert obs["instruction"] == "organize the table"
    assert obs["vision"] == {}
    assert obs["remaining_steps"] == env.step_lim
    # T4d：视觉已接。渲染后端可用时返回三相机契约；不可用时显式 BLOCKED
    # （fail-closed，不给空图冒充）。完整三相机断言见 tests/test_cameras.py。
    try:
        vision = env.get_obs(include_vision=True)["vision"]
    except RuntimeError as e:
        assert "BLOCKED" in str(e)
    else:
        assert set(vision) == {"cam_head", "cam_left_wrist", "cam_right_wrist"}
        for v in vision.values():
            assert v["color"].dtype == np.uint8 and v["color"].shape[2] == 4


def test_obs_positions_match_fixture(env):
    """成功态对象注入后，判据应看到 fixture 里的物体位姿。"""
    env.reset()
    obs = env.get_obs()
    env.run_reward()
    entry = env.reward_manager.check_list[0][0] if env.reward_manager.check_list[0] else []
    assert all(env._call(n[1], {**n[2], "env_idx": 0}) == 1.0 for n in entry)


def test_reset_clears_velocity_and_time(env):
    """T3（audit F03）：reset 必须清零速度与仿真时间——上一集物理残留禁止。"""
    env.data.qvel[:] = 1.0
    env.data.time = 12.0
    env.reset()
    assert np.all(env.data.qvel == 0.0)
    assert env.data.time == 0.0


def test_reset_clears_warmstart_ctrl_act(env):
    """T3：warmstart/ctrl/act 计数同为跨集残留源。"""
    env.data.qacc_warmstart[:] = 0.5
    env.data.ctrl[:] = 0.3
    env.data.act[:] = 0.7
    env.reset()
    assert np.all(env.data.qacc_warmstart == 0.0)
    assert np.all(env.data.ctrl == 0.0)
    assert np.all(env.data.act == 0.0)


def test_mark_unstable_and_reset(env):
    """T3（audit F03）：mark_env_unstable 不再 AttributeError，且 reset 清除。"""
    env.mark_env_unstable(0)
    assert 0 in env.unstable_envs
    env.reset()
    assert 0 not in env.unstable_envs
