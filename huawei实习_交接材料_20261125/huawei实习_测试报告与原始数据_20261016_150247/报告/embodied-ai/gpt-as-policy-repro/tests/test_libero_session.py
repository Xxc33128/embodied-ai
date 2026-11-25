"""LP-A1：LIBERO session 契约测试（需 mujoco+libero；无则整组 skip——Mac 开发机
默认 skip，正式执行在 gap-sim 容器）。"""
import numpy as np
import pytest

libero_session = pytest.importorskip(
    "gap_repro.sim.libero_session", reason="requires mujoco+libero stack")

DUMMY = np.array([0.0] * 6 + [-1.0])


@pytest.fixture(scope="module")
def session():
    s = libero_session.LiberoSession(suite="libero_goal", task_index=0, seed=0)
    yield s
    s.env.close()


def test_bddl_resolved(session):
    import pathlib
    assert pathlib.Path(session.bddl_file).exists()
    assert session.task_description  # 语言指令非空


def test_reset_movables_hard_gate(session):
    # 硬门（robot+movables）跨 reset 逐位一致：恰为 set_init_state 恢复集上的
    # 构造性保证；场景 fixtures 漂移不在此门内（见 fixture 相关用例）
    obs1 = session.reset_to(0)  # wait_steps=10 对齐 openpi 客户端协议
    st1 = session.fingerprint_state(obs1)
    for _ in range(3):
        session.step(DUMMY)
    obs2 = session.reset_to(0)
    st2 = session.fingerprint_state(obs2)
    assert st1 == st2, "同 case 两次 reset 的 movables 指纹必须逐位一致"


def test_fingerprint_full_scene_sensitivity(session):
    # LP-A1 审查 P1 反盲回归：fixture 根 body 的位姿微扰（平移 1mm + 旋转 1e-3）
    # 必须改变 scene digest 与 fingerprint_full（证明配对身份对场景布局不再盲）
    obs = session.reset_to(0)
    scene0 = session.fixture_pose_digest()
    full0 = session.fingerprint_full(obs)
    domain = session.env.env
    name = sorted(domain.fixtures_dict.keys())[0]
    bid = domain.sim.model.body_name2id(domain.fixtures_dict[name].root_body)
    orig_pos = domain.sim.model.body_pos[bid].copy()
    orig_quat = domain.sim.model.body_quat[bid].copy()
    domain.sim.model.body_pos[bid] = orig_pos + np.array([1e-3, 0.0, 0.0])
    try:
        scene_pos = session.fixture_pose_digest()
    finally:
        domain.sim.model.body_pos[bid] = orig_pos
    pert = np.array([1e-3, 0.0, 0.0, 1.0])
    domain.sim.model.body_quat[bid] = (orig_quat + pert) / np.linalg.norm(orig_quat + pert)
    try:
        scene_quat = session.fixture_pose_digest()
    finally:
        domain.sim.model.body_quat[bid] = orig_quat
    assert scene_pos != scene0, "fixture 平移 1mm 扰动必须改变 scene digest"
    assert scene_quat != scene0, "fixture 旋转微扰必须改变 scene digest"
    assert session.fixture_pose_digest() == scene0, "扰动位姿未恢复原状"
    assert session.fingerprint_full(obs) == full0, "恢复后配对身份应回到原值"


def test_scene_digest_deterministic_within_episode(session):
    # 同一 episode 内 scene digest 确定（模型场读取无逐次抖动）
    session.reset_to(0)
    assert session.fixture_pose_digest() == session.fixture_pose_digest()


def test_fixture_drift_within_region_envelope(session):
    # 跨 reset fixture 漂移有界（审查 P3-3 收紧）：x/y 门取该 fixture 专用
    # MultiRegionRandomSampler（名 f"{fixture}_sampler"）的 x_ranges/y_ranges
    # 精确跨度；无 sampler 的 fixture 退回全局 region 最大边长。z 由固定
    # z_offset 决定，门=2e-3（实测 z 漂移恒 0）。超界说明采样器/协议异常。
    session.reset_to(0)
    poses_a = session.fixture_poses()
    session.reset_to(0)
    poses_b = session.fixture_poses()
    samplers = session.env.env.placement_initializer.samplers
    regions = session.env.env.parsed_problem.get("regions", {})
    global_extent = max(
        (max(r[2] - r[0], r[3] - r[1]) for spec in regions.values()
         for r in spec.get("ranges", [])), default=0.0)
    assert global_extent > 0, "BDDL regions 未解析到（任务族无 region？）"
    for name in poses_a:
        s = samplers.get(f"{name}_sampler")
        if s is not None and hasattr(s, "x_ranges"):
            bound_xy = max(hi - lo for lo, hi in
                           list(s.x_ranges) + list(s.y_ranges)) + 2e-3
        else:
            bound_xy = global_extent + 2e-3
        for axis in (0, 1):
            delta = abs(poses_a[name]["body_pos"][axis]
                        - poses_b[name]["body_pos"][axis])
            assert delta <= bound_xy, (
                f"fixture {name} axis{axis} 漂移 {delta:.4f} 超出包络 {bound_xy:.4f}")
        delta_z = abs(poses_a[name]["body_pos"][2] - poses_b[name]["body_pos"][2])
        assert delta_z <= 2e-3, f"fixture {name} z 漂移 {delta_z:.4f} 超出 2e-3"


def test_same_state_double_render(session):
    # 图像差归因哨兵（LP-A1 审查 P1 对照实验的测试级固化）：
    # (1) 渲染器逐位稳定判据（探针预注册）：同物理状态连续重渲染
    #     mean_abs≤1.0 且 frac_gt30≤0.01——实测 agentview 0.0019、腕部
    #     0.04–0.09，余量充分。注意同状态观测必须取自同一物理状态
    #     （不得隔 dummy 步比较：3 个 dummy 步在腕部视野即有 ~1.0 的
    #     真实视觉变化，首版探针曾混淆，见 lp_a1_fixture_drift_probe.py）。
    # (2) 主导关系：同状态 diff 的 3 倍须仍小于跨 reset diff（实测
    #     agentview 2.9 / 腕部 5.0，fixture 重采样主导 30×/50×+）；
    #     若失败=fixture 漂移消失或渲染器噪声爆发，归因模型须改写。
    obs1 = session.reset_to(0)
    obs1b = session.env.env._get_observations(force_update=True)
    obs2 = session.reset_to(0)
    for cam in ("agentview_image", "robot0_eye_in_hand_image"):
        same = session.image_diff_stats(obs1, obs1b, key=cam)
        cross = session.image_diff_stats(obs1, obs2, key=cam)
        assert same["mean_abs"] <= 1.0 and same["frac_gt30"] <= 0.01, same
        assert same["mean_abs"] * 3 <= cross["mean_abs"], (
            f"{cam}: 同状态 diff {same} 未被跨 reset diff {cross} 主控（≥3×），归因失效")


def test_pairing_init_states_differ(session):
    obs0 = session.reset_to(0)
    st0 = session.fingerprint_state(obs0)
    obs1 = session.reset_to(1)
    st1 = session.fingerprint_state(obs1)
    assert st0 != st1, "不同 init state 必须产生不同状态（配对错位检测）"


def test_success_flag_available(session):
    obs = session.reset_to(0)
    assert isinstance(session.is_success(), bool)
    assert session.is_success() is False, "近初态 dummy 状态不应判定成功"
    assert session.fingerprint_state(obs)  # 状态指纹可计算


def test_fingerprint_keys_contract():
    # 契约：指纹键集双向锁定（fingerprint_state 循环体即用此常量，
    # 改键集必改常量、必触发本测试）
    assert "agentview_image" in libero_session.FINGERPRINT_IMAGE_KEYS
    assert "robot0_eye_in_hand_image" in libero_session.FINGERPRINT_IMAGE_KEYS
    assert set(libero_session.FINGERPRINT_STATE_KEYS) == {
        "robot0_joint_pos", "robot0_eef_pos", "robot0_gripper_qpos", "object-state"}
