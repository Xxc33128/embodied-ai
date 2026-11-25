"""W4：25Hz ACK → 10×250Hz 子步控制序列 + 渐进夹爪（复刻原语义）。

原实现（E1）：
- src/eval_client/eval_env.py：interpolation_nums=collect_interval（=10，250/25Hz）；
  臂/夹爪前 floor(n*0.8)=8 子步线性插值 alpha=(i+1)/(interp_count+1) 即 1/9…8/9，
  其后 2 子步保持目标；夹爪每个插值值 clip 到 gripper_scale 后配 mimic；
  支持臂队列每子步最多消费 1 项、耗尽即断、剩余跨步保留。
- env/robot_manager/control_manager.py：process_gripper_val 渐进开合——
  目标与当前差 < 行程的 20% 时直达，否则单步移动行程的 20%。

分层（R7 修订）：渐进夹爪在 250Hz 子步内以实测关节值应用
（environment.take_action 的子步循环调 apply_progressive_gripper），
25Hz 目标层与插值层均不做 20% 限幅，全程仅此一处。
"""

from __future__ import annotations

import copy
from collections import deque

import numpy as np

GRIPPER_KEY_SUFFIX = "ee_joint_state"


def apply_progressive_gripper(
    target: float, current: float, scale: tuple[float, float], eps: float = 0.2
) -> float:
    """原 process_gripper_val：单步最多移动 (hi-lo)*eps，接近目标则直达。"""
    lo, hi = scale
    target, current = float(target), float(current)
    percentage = abs(target - current) / (hi - lo)
    if percentage < eps:
        return target
    if target > current:
        return current + (hi - lo) * eps
    return current - (hi - lo) * eps


class SubstepScheduler:
    """一个 25Hz 控制步 → n 个 250Hz 子步控制 dict。支持臂队列跨步持久。"""

    def __init__(
        self,
        interpolation_nums: int = 10,
        gripper_scale: tuple[float, float] = (-0.01, 0.044),
        mimic: tuple[float, float] = (1.0, 0.0),
    ):
        if interpolation_nums < 1:
            raise ValueError("interpolation_nums must be >= 1")
        self.n = int(interpolation_nums)
        self.interp_count = int(np.floor(self.n * 0.8))
        self.gripper_scale = gripper_scale
        self.mimic = mimic
        self.support_queue: deque = deque()

    def push_support_actions(self, entries: list[dict]) -> None:
        """支持臂（非 target）控制 dict 入队，每项为一个子步的完整控制。"""
        self.support_queue.extend(entries)

    def make_sequence(
        self,
        control_info: dict,
        current_arm: dict[str, list[float]],
        current_gripper: dict[str, float],
    ) -> list[dict]:
        """control_info 为 25Hz 目标（gripper 的 position 为 [标量]）；
        current_* 为 ACK 时刻实测状态。返回 n 个独立子步 dict。"""
        seq = [copy.deepcopy(control_info) for _ in range(self.n)]
        # 支持臂先消费（原实现顺序），每子步至多一项，队列空即止
        for i in range(self.n):
            if not self.support_queue:
                break
            seq[i].update(self.support_queue.popleft())
        for key, ctrl in control_info.items():
            if key.endswith(GRIPPER_KEY_SUFFIX):
                self._interp_gripper(seq, key, ctrl, current_gripper[key])
            else:
                self._interp_arm(seq, key, ctrl, current_arm[key])
        return seq

    def _interp_arm(self, seq, key, ctrl, current):
        cur = np.asarray(current, float)
        target = np.asarray(ctrl["position"], float)
        if cur.shape != target.shape:
            raise ValueError(f"{key}: current/target shape mismatch")
        for i in range(self.interp_count):
            alpha = (i + 1) / (self.interp_count + 1)
            seq[i][key]["position"] = ((1 - alpha) * cur + alpha * target).tolist()
        for i in range(self.interp_count, self.n):
            seq[i][key]["position"] = target.tolist()

    def _interp_gripper(self, seq, key, ctrl, current):
        lo, hi = self.gripper_scale
        m_a, m_b = self.mimic
        raw_target = float(ctrl["position"][0])
        target = float(np.clip(raw_target, lo, hi))
        for i in range(self.interp_count):
            alpha = (i + 1) / (self.interp_count + 1)
            v = float(np.clip((1 - alpha) * current + alpha * raw_target, lo, hi))
            seq[i][key]["position"] = [v, v * m_a + m_b]
        for i in range(self.interp_count, self.n):
            seq[i][key]["position"] = [target, target * m_a + m_b]


def clamp_substep_target(current: float, target: float, dt: float = 0.004,
                         velocity_limit: float = 5.0) -> float:
    """velocity_limit_sim=5.0 的执行层等效：单子步目标位移 ≤ v*dt（E3 工程等效，
    Isaac 为执行器速度上限；MJCF position 执行器无对应原语）。"""
    max_delta = velocity_limit * dt
    delta = float(target) - float(current)
    if abs(delta) <= max_delta:
        return float(target)
    return float(current) + np.sign(delta) * max_delta
