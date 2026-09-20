# W5：原 RoboDojoSession → env 调用清单（E1）

来源：`GPT-as-Policy@8f3d362b hybrid_rollout/robodojo/robodojo_server/session.py`（231 行）
与 `register_native_evaluation`（S25）。`sim/environment.py` 必须精确提供以下表面，
不模拟其余 Isaac API。

## register_native_evaluation(env)

| 调用 | 语义 |
|---|---|
| `env.run_reward()` | 注册当前进度判据（native reset 会清空） |
| `env.get_score()`（如有） | 初始分查询 |
| `env.num_envs` | 环境数（评测=1） |
| `env.reward_manager.check_list / final_check_list / trigger_check_list` | 每 env 的判据组；**任一 env 三者总数为 0 → 拒绝（空判据 fail-closed）** |
| `env.interact`（如有）+ `env.query_support_arm_traj(env_idx)` | 支持臂演示轨迹查询必须注册（排序/麻将） |
| `env.get_running_env_idx_list()` | 运行中 env 索引 |

## reset 路径

| 调用 | 语义 |
|---|---|
| `env.seed_manager.seed_info[seed]['scene_layout']` | reset 前按 seed 取实际布局 |
| `env.reset(seed=[seed])` | native reset |

## 观测（session._observe）

`raw = env.get_obs()`，session 要求：
- `raw['state'][f'{arm}_arm_joint_state']`（6）与 `f'{arm}_ee_joint_state'`（1）拼接为 14 维且有限；
- `raw['state'][f'{arm}_ee_pose']`（7，位置+wxyz 四元数）；
- `raw['vision'][<相机>]['color']` HxWx4 uint8，别名集：cam_high∈{cam_high,cam_head,head_camera,top_camera}，cam_left_wrist∈{cam_left_wrist,left_camera}，cam_right_wrist∈{cam_right_wrist,right_camera}；
- `raw['instruction']`（str）。

## 执行（session.chunk_step）

- 校验 1..15 行 ×14、夹爪列 6/13 ∈[0,1]；
- 每行构造 command（left/right × arm_joint_state[6] + ee_joint_state[1]）后调 `env.take_action(command)`；
- **`env.take_action_cnt[0]` 必须恰好 +1**（"Native action was not executed exactly once"）；
- 之后读 `env.end_flag[0]`、`env.success[0]`、`env.step_lim`、`self.step_id` 判 terminated/truncated。

## 终局与分数

- `env.reward_manager.get_score()[0]`：score∈[0,100]，native_score = score/100；
- success=1.0 时直接记 1.0；
- 结果字段（evaluation_outcome.json）：`complete / valid_for_success_rate / native_success / native_score / status∈{invalid_native_layout, native_completed, budget_censored, incomplete} / reason / native_control_steps / native_step_limit`。

## 判据构造（任务侧，十个任务共用的查询词汇表）

统计自 `task/RoboDojo/tasks/*.py` 的 `reward_manager.<func>(...)` 调用：
`check×77, all_robot_back_to_origin×39, is_axis_up×34, is_axis_aligned×34, is_lift×29,
query×16, update_object_state×11, is_joint_position_above_ratio×10,
is_robot_not_back_to_origin×9, is_not_moved×8, is_joint_position_below_ratio×7,
is_all_gripper_open×6, is_stacked×5, is_qpos_close×5,
is_functional_point_lower_than_root_point×5, is_joint_position_ratio_change_from_above_to_below×4,
is_moved×3, trigger_check×2, repeat×2, is_not_lift×2, is_labels_axis_difference_in_range×2,
is_functional_point_not_moved×2, is_in_line×1, check_single_env×3, call_func_parser×1`

这 26 个函数即 `tasks/state_adapter.py` 需对照 `env/reward_manager/func_parser.py`
（E1，112KB）逐一实现语义的状态查询面。
