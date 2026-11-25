#!/usr/bin/env python3
"""LP-A0 真实帧采集（gap-sim 容器）：LIBERO Goal task0 的 6 个 init state 首帧。

输出 /workspace/data/lp_a0_frames.npz：
  agentview (6,256,256,3) uint8、wrist (6,256,256,3) uint8、state (6,8) float32
state = robot0_joint_pos(7) + robot0_gripper_qpos[:1]（8 维，与 LiberoInputs 契约一致）。
"""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("LIBERO_CONFIG_PATH", "/workspace/data/libero_config")
sys.path.insert(0, "/workspace/repo/upstream/LIBERO-PRO")

from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

import numpy as np  # noqa: E402
import pathlib  # noqa: E402

task_suite = benchmark.get_benchmark_dict()["libero_goal"]()
task = task_suite.get_task(0)
bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=224, camera_widths=224)
env.seed(0)
env.reset()
init_states = task_suite.get_task_init_states(0)

av, wr, st = [], [], []
for i in range(6):
    obs = env.set_init_state(init_states[i])
    for _ in range(3):
        obs, _, _, _ = env.step(np.array([0.0] * 6 + [-1.0]))
    av.append(obs["agentview_image"].copy())
    wr.append(obs["robot0_eye_in_hand_image"].copy())
    st.append(np.concatenate([obs["robot0_joint_pos"], obs["robot0_gripper_qpos"][:1]]).astype(np.float32))
    print(f"frame {i} joints={st[-1].round(3).tolist()}", flush=True)
env.close()

np.savez("/workspace/data/lp_a0_frames.npz",
         agentview=np.stack(av), wrist=np.stack(wr), state=np.stack(st))
print("FRAMES_SAVED", np.stack(av).shape, np.stack(st).shape)
