"""W5：十任务注册表（元数据 + 判据构造锚点）。

step_lim 来自各任务源码（E1，task/RoboDojo/tasks/<task>.py 的 self.step_lim）；
organize_table 的判据构造在 tasks/organize_table.py 中按原文件逐条镜像。
判据求值语义由 state_adapter 对照 func_parser 实现。
"""

from __future__ import annotations

from dataclasses import dataclass

# 计划 §5 的控制步预算（E2，原 panel）与任务源码 step_lim（E1）对应一致。
STEP_LIM = {
    "organize_table": 1000,
    "classify_objects_by_language": 1100,
    "imitate_sorting_sequence": 1600,
    "arrange_largest_number": 1050,
    "pack_objects_into_box": 1300,
    "classify_objects": 1100,
    "build_tower": 1050,
    "make_kong": 600,
    "fold_clothes": 500,
    "put_bottles_into_dustbin": 700,
}

# 支持臂交互任务（需要 query_support_arm_traj 与 Franka）。
INTERACT_TASKS = ("imitate_sorting_sequence", "make_kong")


@dataclass(frozen=True)
class TaskSpec:
    name: str
    step_lim: int
    interact: bool


def get_task_spec(name: str) -> TaskSpec:
    if name not in STEP_LIM:
        raise KeyError(f"unknown task: {name}")
    return TaskSpec(name=name, step_lim=STEP_LIM[name],
                    interact=name in INTERACT_TASKS)


def all_task_specs() -> list[TaskSpec]:
    return [get_task_spec(n) for n in STEP_LIM]
