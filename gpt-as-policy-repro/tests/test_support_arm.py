"""支持臂轨迹回放测试：队列生命周期、drain 语义、query 一次性注册。"""

from __future__ import annotations

from gap_repro.sim.support_arm import SupportArmReplay


def _mk(n, pos_fn):
    return [{"position": [pos_fn(i)] * 6} for i in range(n)]


def _entry_seq(n):
    return ([{"position": [0.1 * i] * 6} for i in range(n)],
            [{"position": [0.01 * i]} for i in range(n)])


def test_load_and_drain():
    sr = SupportArmReplay(0)
    traj_arm, traj_eef = _entry_seq(5)
    sr.load_support_arm_traj(traj_arm, traj_eef)
    assert sr.pending == 5
    out = sr.drain(3)
    assert len(out) == 3 and sr.pending == 2
    out2 = sr.drain(10)
    assert len(out2) == 2 and sr.pending == 0
    assert sr.drain(1) == []


def test_query_only_registers_once():
    sr = SupportArmReplay(0)
    ta, te = _entry_seq(1)
    sr.load_support_arm_traj(ta, te)
    assert sr.query() is True  # 首次允许
    assert sr.query() is False  # 二次跳过（防重复注册）


def test_queue_exhausted_is_exhausted():
    sr = SupportArmReplay(0)
    ta, te = _entry_seq(1)
    sr.load_support_arm_traj(ta, te)
    sr.query()  # 原 query_support_arm_traj 递增 query_support_times
    assert not sr.exhausted
    sr.drain(10)
    assert sr.exhausted


def test_exhausted_requires_query_and_empty_queue():
    sr = SupportArmReplay(0)
    ta, te = _entry_seq(1)
    sr.load_support_arm_traj(ta, te)
    sr.query()
    sr.drain(10)
    assert sr.exhausted  # query 过 + 队列空 = exhausted


def test_multi_env_isolation():
    sr = SupportArmReplay()
    ta, te = _entry_seq(1)
    sr.load_support_arm_traj(ta, te)
    sr.env_idx = 1
    assert sr.pending == 0  # env 1 无数据
    sr.env_idx = 0
    assert sr.pending == 1
