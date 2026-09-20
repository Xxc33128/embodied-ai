#!/usr/bin/env python3
"""LP1 验收冒烟：LIBERO Goal 首任务 env 创建 + reset/init + 10 步 dummy 闭环。

验证点：libero 导入链、bddl 加载、robosuite 1.4.0 + mujoco 3.2.3 在 aarch64
CPU 上的 env 创建/渲染（OSMesa）/step、观测结构与图像有效性、耗时基线。
不作为任何策略评测依据。
"""
import os
import os.path as osp
import sys
import time

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("LIBERO_CONFIG_PATH", "/workspace/data/libero_config")

sys.path.insert(0, "/workspace/repo/upstream/LIBERO-PRO")

from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

import numpy as np  # noqa: E402

t0 = time.time()
task_suite = benchmark.get_benchmark_dict()["libero_goal"]()
task = task_suite.get_task(0)
task_description = task.language
bddl_file = osp.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl_file, camera_heights=256, camera_widths=256)
env.seed(0)
t_create = time.time() - t0

t1 = time.time()
env.reset()
init_states = task_suite.get_task_init_states(0)
obs = env.set_init_state(init_states[0])
t_reset = time.time() - t1

t2 = time.time()
dummy = np.array([0.0] * 6 + [-1.0])
for _ in range(10):
    obs, reward, done, info = env.step(dummy)
t_step = time.time() - t2

img = obs["agentview_image"]
wrist = obs["robot0_eye_in_hand_image"]
assert img.shape == (256, 256, 3), img.shape
assert img.std() > 0 and wrist.std() > 0, "degenerate render"
assert len(obs["robot0_joint_pos"]) == 7 and len(obs["robot0_eef_pos"]) == 3
assert len(obs["robot0_gripper_qpos"]) == 2 and "object-state" in obs
assert not done, "dummy action must not terminate"

print(f"LIBERO_SMOKE_OK task={task_description!r}")
print(f"  create={t_create:.1f}s reset={t_reset:.1f}s 10steps={t_step:.2f}s")
print(f"  agentview={img.shape} std={img.std():.1f} wrist={wrist.shape} std={wrist.std():.1f}")
print(f"  joints={obs['robot0_joint_pos'].round(3).tolist()}")
env.close()
