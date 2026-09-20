"""W5：MuJoCo 环境的 session 调用表面。

对照 docs/acceptance/w5-session-call-inventory.md（E1）逐项实现：
register_native_evaluation 需要的 run_reward/reward_manager 三清单（空判据
fail-closed）、seed_manager.seed_info、reset(seed)、get_obs 状态契约、
take_action（take_action_cnt 恰好 +1）、end_flag/success/step_lim、get_score。

对象状态经 register_layout_objects 注入（冻结 fixture 或未来场景转换产物）；
机器人侧为真实 MuJoCo 物理步进。视觉渲染后端未就绪（镜像 mujoco 无
Renderer、无 PyOpenGL）→ include_vision=True 显式抛 BLOCKED（W3 待办）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gap_repro.sim import control, robots, scene
from gap_repro.tasks import registry
from gap_repro.tasks import task_entries as te
from gap_repro.tasks.check_once import entry_passed
from gap_repro.tasks.state_adapter import (
    FuncEvaluator,
    ObjectState,
    RobotState,
    StateProvider,
)

ARM_KEYS = ("left_arm_joint_state", "right_arm_joint_state")
GRIP_KEYS = ("left_ee_joint_state", "right_ee_joint_state")


class RewardManagerView:
    """session 依赖的 reward_manager 只读面（清单容器 + get_score）。"""

    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self._score = 0.0
        self.check_list = [[] for _ in range(num_envs)]
        self.final_check_list = [[] for _ in range(num_envs)]
        self.trigger_check_list = [[] for _ in range(num_envs)]
        self.query_list = [[] for _ in range(num_envs)]

    def reset(self):
        self.check_list = [[] for _ in range(self.num_envs)]
        self.final_check_list = [[] for _ in range(self.num_envs)]
        self.trigger_check_list = [[] for _ in range(self.num_envs)]

    def step_criteria(self, call) -> bool:
        """原文 reward_manager.step 语义（E1）：首条 entry 经 check_once 逐元素
        求值（AND 跨元素、OR/AND 交替递归），全真则 pop。
        返回三清单是否已全空（= get_reward 的 1.0 条件）。"""
        if self.check_list and self.check_list[0]:
            if entry_passed(self.check_list[0][0], call):
                self.check_list[0].pop(0)
        return (len(self.check_list[0]) == 0
                and len(self.final_check_list[0]) == 0
                and len(self.trigger_check_list[0]) == 0)

    def register_query(self, query_list, aim_num):
        """原 reward_manager.query：判据满足时计数 +1，超过 aim_num 判败。"""
        for env_idx in range(self.num_envs):
            self.query_list[env_idx].append([query_list, aim_num, 0])

    def step_queries(self, call, env_idx=0):
        """每控制步推进 query 计数；返回 (不超限?, 计数==aim?)。
        current_num > aim → _mark_env_failed；current_num != aim → reward 0。"""
        results = []
        for query in self.query_list[env_idx]:
            query_checks, aim_num, counter = query
            ok = all(float(call(c[0], {**c[1], "env_idx": env_idx})) == 1.0
                     for c in query_checks)
            if ok:
                query[2] = counter + 1
            within = query[2] <= aim_num
            exact = query[2] == aim_num
            results.append((within, exact))
        return results

    def require_non_empty(self):
        """原 register_native_evaluation 的 fail-closed：空判据拒绝。"""
        counts = {n: [len(g) for g in getattr(self, n)]
                  for n in ("check_list", "final_check_list", "trigger_check_list")}
        ok = all(sum(counts[n][i] for n in counts) > 0 for i in range(self.num_envs))
        if not ok:
            raise RuntimeError("Native task completion conditions are empty; "
                               "refusing vacuous success")
        return counts

    def get_running_env_idx_list(self):
        return [0] if not self.end_flag[0] else []

    def query_support_arm_traj(self, env_idx=0):
        if not self.interact:
            return []
        raise NotImplementedError("支持臂演示轨迹回放待 W5b（imitate_sorting/make_kong）")

    def get_score(self):
        return [self._score]


class SeedManager:
    """session.reset 前按 seed 取布局：seed_info[seed]['scene_layout']。"""

    def __init__(self, seed_info: dict | None = None):
        self.seed_info = seed_info or {}

    def add_case(self, reset_seed: int, layout_path: str):
        self.seed_info.setdefault(reset_seed, {})["scene_layout"] = layout_path


class MuJoCoEnvironment:
    def __init__(self, urdf: Path, task: str = "organize_table",
                 seed_info: dict | None = None, layout_name: str | None = None,
                 obj_root: str | None = None, layout_dir: str | None = None):
        """layout_name 给定时加载真实场景（T4）：同一 model/data 承载机器人、
        桌面、自由物体；否则仅机器人（既有 fixture 注入路径）。"""
        import mujoco

        self.scene_manifest = None
        if layout_name:
            self.model, self.data, self.scene_manifest = scene.build_scene_mjcf(
                urdf, layout_name, obj_root or scene.OBJ_ROOT,
                layout_dir=layout_dir)
            self.meta = None
        else:
            self.model, self.meta = robots.compile_dual_x5(urdf)
            self.data = mujoco.MjData(self.model)
        self._mujoco = mujoco
        self.task_spec = registry.get_task_spec(task)
        self.task = task
        self.num_envs = 1
        self.interact = self.task_spec.interact
        self.step_lim = self.task_spec.step_lim
        self.take_action_cnt = [0]
        self.end_flag = [False]
        self.success = [False]
        self.step_id = 0
        self.dt = 0.004
        # 每环境物理失稳标志。初值必须为空列表：上游 session.py:179 用
        # `0 in unstable_envs` 成员测试（False==0，[False] 会让 0 号环境恒失稳）。
        # 接通仿真后按需 append/置位，仅以 `i in unstable_envs` 表达"环境 i 失稳"。
        self.unstable_envs = []
        self.scheduler = control.SubstepScheduler(
            interpolation_nums=10, gripper_scale=robots.GRIPPER_SCALE,
            mimic=robots.MIMIC[1:])
        self.seed_manager = SeedManager(seed_info)
        self.reward_manager = RewardManagerView(self.num_envs)
        self.provider = StateProvider(objects={}, robot=self._robot_state_from_mj())
        if self.scene_manifest:
            self.provider.label_cat_index = {
                label: e["category_idx"]
                for label, e in self.scene_manifest["objects"].items()}
            self._write_default_object_poses()
            mujoco.mj_forward(self.model, self.data)
            self._sync_provider_objects()
        self._episode_registered = False
        self._entries = None
        self._score_stages = None
        self._score_count = 0
        self.evaluator = FuncEvaluator(self.provider)
        self.instruction = ""
        self._vision = None  # T4d：惰性创建 VisionRenderer
        mujoco.mj_forward(self.model, self.data)

    # --- 状态注入与同步 ---
    def register_layout_objects(self, objects: dict[str, ObjectState]):
        self.provider.objects = dict(objects)

    def _write_default_object_poses(self):
        """把 manifest 的冻结初态写回自由关节（mj_resetData 会把 freejoint 清零）。"""
        m, d = self.model, self.data
        for label, e in self.scene_manifest["objects"].items():
            if not e["dynamic"]:
                continue
            bid = int(m.body(e["body"]).id)
            adr = int(m.body_jntadr[bid])
            if adr < 0 or int(m.body_jntnum[bid]) != 1:
                raise RuntimeError(f"{label}: dynamic object without single freejoint")
            if int(m.jnt_type[adr]) != int(self._mujoco.mjtJoint.mjJNT_FREE):
                raise RuntimeError(f"{label}: joint is not free")
            qadr = int(m.jnt_qposadr[adr])
            d.qpos[qadr:qadr + 3] = e["default_pos"]
            d.qpos[qadr + 3:qadr + 7] = e["default_quat_wxyz"]
            vadr = int(m.jnt_dofadr[adr])
            d.qvel[vadr:vadr + 6] = 0.0

    def _sync_provider_objects(self):
        """从同一 MjData 同步物体位姿与局部 bbox（T4：评分只见真实物理状态）。"""
        if not self.scene_manifest:
            return
        for label, e in self.scene_manifest["objects"].items():
            bid = int(e["body_id"])
            self.provider.objects[label] = ObjectState(
                label=label,
                position=self.data.xpos[bid].copy(),
                quat_wxyz=self.data.xquat[bid].copy(),
                bbox_vertices_local=np.array(
                    [e["local_bbox_min"], e["local_bbox_max"]], dtype=float))

    def _robot_state_from_mj(self) -> RobotState:
        d, m = self.data, self.model
        arm, grip, ee, quat = {}, {}, {}, {}
        for side in ("left", "right"):
            q6 = [d.qpos[m.joint(f"{side}_{j}").qposadr[0]] for j in robots.ARM_JOINTS]
            j7 = d.qpos[m.joint(f"{side}_joint7").qposadr[0]]
            lo, hi = robots.GRIPPER_SCALE
            arm[side] = list(q6)
            grip[side] = (j7 - lo) / (hi - lo)
            bid = m.body(f"{side}_link6").id
            ee[side] = d.xpos[bid].copy()
            quat[side] = d.xquat[bid].copy()
        return RobotState(arm_joints=arm, gripper_openings=grip, ee_positions=ee,
                          ee_quats_wxyz=quat)

    def _call(self, name, args):
        return getattr(self.evaluator, name)(**args)

    def _sync_provider_robot(self):
        prev = self.provider.robot
        rs = self._robot_state_from_mj()
        rs.origin_endpose = dict(prev.origin_endpose)  # 归位基准跨步保留
        self.provider.robot = rs

    # --- session 表面 ---
    def run_reward(self):
        """按任务构建节点判据 entries 并登记（原文 run_eval 前言语义，一次）。"""
        if self._entries is None:
            pass  # 首次构建（下方按任务分支）
        elif not self.reward_manager.check_list or len(self.reward_manager.check_list[0]) == 0:
            for e in self._entries:  # reset 后队列被清空 → 幂等重挂
                self.reward_manager.check_list[0].append(e)
            return self.reward_manager.require_non_empty()
        else:
            return self.reward_manager.require_non_empty()
        if self.task == "organize_table":
            lim = self.provider.label_cat_index.get("garage")
            garage_dis = {5: 0.045, 6: 0.025, 7: 0.045, 8: 0.02, 11: 0.032}.get(lim, 0.02)
            self._entries = [te.organize_entry(garage_dis)]
            self._score_stages = te.organize_score_stages()
        elif self.task == "put_bottles_into_dustbin":
            self._entries = te.dustbin_entries()
            self._score_stages = te.dustbin_score_stages()
        elif self.task == "classify_objects":
            self._entries = te.classify_entries()
            self._score_stages = te.classify_score_stages()
        else:
            raise NotImplementedError(f"criteria entries for {self.task} pending (W5b)")
        for e in self._entries:
            self.reward_manager.check_list[0].append(e)
        self._episode_registered = True
        return self.reward_manager.require_non_empty()

    def reset(self, seed=None):
        """native reset（T3，audit F03）：MuJoCo 完整 data reset 后恢复冻结初态。

        mj_resetData 清 qpos/qvel/ctrl/act/time/warmstart/计数器（上一集物理
        残留全清），再写冻结初始 qpos（双臂零位 + 夹爪 0.044 开度）。
        注：seed→布局映射在布局接线（T4）后生效，本步保持初态与既有语义一致。
        """
        import mujoco
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = 0.0
        for side in ("left", "right"):
            self.data.qpos[self.model.joint(f"{side}_joint7").qposadr[0]] = 0.044
            self.data.qpos[self.model.joint(f"{side}_joint8").qposadr[0]] = 0.044
        if self.scene_manifest:
            self._write_default_object_poses()
        mujoco.mj_forward(self.model, self.data)
        self.take_action_cnt = [0]
        self.end_flag = [False]
        self.success = [False]
        self.step_id = 0
        self.unstable_envs = []  # 失稳标记跨 episode 不泄漏
        self.reward_manager.reset()  # 含 query_list 清零（审查红2）
        self.reward_manager._score = 0.0  # Y4：transition 分跨 episode 不泄漏
        self._score_count = 0             # 红2：档位计数跨 episode 不泄漏
        self._episode_registered = False
        self._sync_provider_robot()
        self._sync_provider_objects()
        self.provider.capture_episode_state()

    def get_obs(self, include_vision: bool = False):
        """原 session._observe 契约（E1）：raw['state'] 各臂 arm/ee_joint_state +
        ee_pose(7D wxyz)、raw['vision']（含 color）、raw['instruction']。
        另附 states(14D)/eef 便捷字段。include_vision=True 时经 VisionRenderer
        输出三相机 HxWx4 uint8；后端不可用则显式 BLOCKED（fail-closed）。"""
        self._sync_provider_robot()
        r = self.provider.robot
        m, d = self.model, self.data
        state = {}
        poses = {}
        for a in ("left", "right"):
            q6 = [float(d.qpos[m.joint(f"{a}_{j}").qposadr[0]]) for j in robots.ARM_JOINTS]
            lo, hi = robots.GRIPPER_SCALE
            j7 = float(d.qpos[m.joint(f"{a}_joint7").qposadr[0]])
            opening = (j7 - lo) / (hi - lo)  # 归一化 0闭/1开（审查红4：原 j7 角尺度错误）
            bid = m.body(f"{a}_link6").id
            state[f"{a}_arm_joint_state"] = np.array(q6, np.float32)
            state[f"{a}_ee_joint_state"] = np.array([opening], np.float32)
            pose7 = np.r_[d.xpos[bid], d.xquat[bid]].astype(np.float32)
            state[f"{a}_ee_pose"] = pose7
            poses[a] = pose7
        states = np.concatenate([np.r_[state[f"{a}_arm_joint_state"],
                                       state[f"{a}_ee_joint_state"]]
                                 for a in ("left", "right")]).astype(np.float32)
        if states.shape != (14,) or not np.isfinite(states).all():
            raise RuntimeError("Invalid native dual-arm state")
        vision = {}
        if include_vision:
            if self._vision is None:
                from gap_repro.sim.observations import VisionRenderer

                self._vision = VisionRenderer(self.model)
            vision = self._vision.render(self.data)
        obs = dict(state=state, vision=vision, instruction=self.instruction,
                   states=states, eef_positions=np.stack([poses["left"][:3],
                                                          poses["right"][:3]]),
                   remaining_steps=max(0, self.step_lim - self.step_id))
        return obs

    def take_action(self, command: dict):
        if self.end_flag[0]:
            raise RuntimeError("episode already ended")
        control_info, current_arm, current_grip = {}, {}, {}
        for arm, akey, gkey in (("left", ARM_KEYS[0], GRIP_KEYS[0]),
                                ("right", ARM_KEYS[1], GRIP_KEYS[1])):
            pos = list(command[akey]["position"])
            if len(pos) != 6:
                raise ValueError(f"{akey}: expected 6 joint targets")
            opening = float(command[gkey]["position"][0])
            if not 0.0 <= opening <= 1.0:
                raise ValueError(f"{gkey}: opening out of [0,1]")
            j7 = robots.opening_to_joint(opening)
            m = self.model
            cur6 = [self.data.qpos[m.joint(f"{arm}_{j}").qposadr[0]]
                    for j in robots.ARM_JOINTS]
            control_info[akey] = {"position": pos}
            control_info[gkey] = {"position": [j7]}
            current_arm[akey] = cur6
            current_grip[gkey] = self.data.qpos[self.model.joint(f"{arm}_joint7").qposadr[0]]
        seq = self.scheduler.make_sequence(control_info, current_arm, current_grip)
        for sub in seq:
            for arm in ("left", "right"):
                akey, gkey = (f"{arm}_arm_joint_state", f"{arm}_ee_joint_state")
                for i, j in enumerate(robots.ARM_JOINTS):
                    act = self.model.actuator(f"{arm}_{j}_act").id
                    cur = self.data.qpos[self.model.joint(f"{arm}_{j}").qposadr[0]]
                    tgt = control.clamp_substep_target(cur, sub[akey]["position"][i])
                    self.data.ctrl[act] = tgt
                # R7：渐进夹爪按原文 MetaControl 语义在每 250Hz 子步以实测值应用
                act = self.model.actuator(f"{arm}_gripper_act").id
                cur7 = self.data.qpos[self.model.joint(f"{arm}_joint7").qposadr[0]]
                j7 = control.apply_progressive_gripper(
                    sub[gkey]["position"][0], cur7, robots.GRIPPER_SCALE)
                self.data.ctrl[act] = j7
            self._mujoco.mj_step(self.model, self.data)
        self.take_action_cnt[0] += 1
        self.step_id += 1
        self._sync_provider_robot()
        self._sync_provider_objects()  # T4：评分前从 MjData 取真实物体位姿
        # 原文语义（E1 reward_manager.step）：判据队列逐 entry AND-pop，
        # 三清单全空 → get_reward=1 → is_episode_end 置 success=True。
        self.run_reward()  # 首次注册 entries；此后仅刷新 require_non_empty 状态
        empty = self.reward_manager.step_criteria(self._call)
        queries_ok = all(w and x for w, x in
                         self.reward_manager.step_queries(self._call))
        if empty and queries_ok:
            self.success[0] = True
        if self.success[0] or self.step_id >= self.step_lim:
            self.end_flag[0] = True

    def mark_env_unstable(self, env_idx=0, reason=""):
        """失稳标记（T3 修复）：容器为 list，按成员语义去重追加；
        上游 `0 in unstable_envs` 兼容（成员测试，非真值测试）。"""
        if env_idx not in self.unstable_envs:
            self.unstable_envs.append(env_idx)

    def get_running_env_idx_list(self):
        return [0] if not self.end_flag[0] else []

    def query_support_arm_traj(self, env_idx=0):
        if not self.interact:
            return []
        raise NotImplementedError("支持臂演示轨迹回放待 W5b（imitate_sorting/make_kong）")

    def get_score(self):
        """原文 transition 语义（E1 _evaluate_score_entries）：从当前状态档
        起逐档尝试推进，每步至多前进一档；单调不回落。"""
        self.run_reward()
        if self._score_stages is None:
            return [0.0]
        while self._score_count < len(self._score_stages):
            checks, s_val = self._score_stages[self._score_count]
            if entry_passed(checks, self._call):
                self._score_count += 1
            else:
                break
        score = (self._score_stages[self._score_count - 1][1][0]
                 if self._score_count > 0 else 0.0)
        self.reward_manager._score = float(score)
        return [float(score)]
