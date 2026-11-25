"""harness 主链路零费冒烟（新口径 §4.2 第二步的接线验证）：
5 维 eef_abs_pose 悬停策略 × robosuite-bowl，验证 eval()/reset(seed)/step/
scorer/日志全链路。悬停策略不解题（success 恒 0 为预期）。

  .venv-robosuite/bin/python verify_robosuite_harness.py
"""
import numpy as np
from inspect_robots import eval
from inspect_robots.scene import Scene
from inspect_robots.scorer import episode_length, success_at_end
from inspect_robots.task import Epochs, Task
from inspect_robots.types import Action, ActionChunk
from inspect_robots.policy import PolicyConfig, PolicyInfo
from inspect_robots.spaces import ActionSemantics, Box

from robosuite_bowl import RobosuiteBowlEmbodiment


class HoverPolicy:
    """5 维 eef_abs_pose 悬停策略：仅验证 harness 接线，不解题。"""

    def __init__(self):
        lo = np.array([-0.30, -0.30, 0.83, -np.pi, 0.0])
        hi = np.array([0.35, 0.30, 1.30, np.pi, 1.0])
        self.info = PolicyInfo(name="hover", action_space=Box(
            shape=(5,), low=lo, high=hi,
            semantics=ActionSemantics(control_mode="eef_abs_pose", rotation_repr="none",
                                      gripper="continuous", frame="world",
                                      dim_labels=("x", "y", "z", "yaw", "gripper"))))
        self.config = PolicyConfig(action_horizon=1)
        self.num_inferences = 0

    def reset(self, scene):
        self.num_inferences = 0

    def act(self, observation):
        self.num_inferences += 1
        return ActionChunk(actions=[Action(data=np.zeros(5))])


def main():
    task = Task(
        name="robosuite-bowl-noop-smoke",
        scenes=[Scene(id=f"seed-{i}", instruction="pick the red cube into the bowl",
                      init_seed=3000 + i) for i in range(2)],
        scorer=[success_at_end(), episode_length()],
        max_steps=60,
        epochs=Epochs(count=1, reducer="mean"),
    )
    emb = RobosuiteBowlEmbodiment(cameras=False)
    (log,) = eval(task, HoverPolicy(), emb, log_dir="logs-rsbowl-noop-smoke")
    print("status:", log.status)
    print("metrics:", {k: round(v, 4) for k, v in log.results.metrics.items()})


if __name__ == "__main__":
    main()
