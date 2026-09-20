"""W5b：支持臂轨迹回放——原 query_support_arm_traj + queue 语义（E1）。

原实现（imitate_sorting_sequence.py L126-145 + make_kong.py L95-170）：
- query_support_arm_traj(env_idx) 在 episode 开始时调用一次，从 self.traj
  （{arm: [...], eef: [...]}) 构造 control_info dict 列表追加到
  support_arm_action[env_idx] 队列；
- 每个 25Hz 控制步的插值阶段从队列 pop 最多 interpolation_nums 条
  （每子步最多 1 条），队列空即断；
- query_support_times 防止重复注册。

本模块管理队列生命周期，队列消费由 SubstepScheduler 完成。
Franka MJCF + pkl 轨迹解析为 W5b 后续（当前用合成轨迹验证队列机制）。
"""

from __future__ import annotations

from collections import deque
from typing import Any


class SupportArmReplay:
    """管理一个支持臂的轨迹回放队列（与 SubstepScheduler 配合）。"""

    def __init__(self, env_idx: int = 0):
        self.env_idx = env_idx
        self.queues: dict[int, deque] = {}  # env_idx → deque[dict]
        self.query_times: dict[int, int] = {env_idx: 0}

    def load_support_arm_traj(self, arm_traj: list[dict], eef_traj: list[dict]):
        """原 load_support_arm_traj：把 arm/eef 轨迹合成 control_info dict 入队。"""
        q = self.queues.setdefault(self.env_idx, deque())
        for step_arm, step_eef in zip(arm_traj, eef_traj):
            ci = {}
            ci.update(step_arm)
            ci.update(step_eef)
            q.append(ci)

    def query(self) -> bool:
        """原 query_support_arm_traj 防重语义：首次 True，后续 False。"""
        if self.query_times.get(self.env_idx, 0) > 0:
            return False
        self.query_times[self.env_idx] = 1
        return True

    def drain(self, n: int) -> list[dict]:
        """弹出至多 n 条（SubstepScheduler 在插值阶段调用）。"""
        q = self.queues.get(self.env_idx)
        if not q:
            return []
        out = []
        for _ in range(min(n, len(q))):
            out.append(q.popleft())
        return out

    @property
    def pending(self) -> int:
        return len(self.queues.get(self.env_idx, deque()))

    @property
    def exhausted(self) -> bool:
        return self.pending == 0 and self.query_times.get(self.env_idx, 0) > 0
