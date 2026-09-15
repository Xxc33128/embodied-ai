"""robosuite bowl 的 inspect-robots 适配器（新口径 §4.2 模型闭环）。

职责仅为协议翻译，不含任何控制代码。独立复核（2026-09-15）后的三项修复：
  ① 夹爪为二值协议：底层 PandaGripper.format_action 按 sign 累加到 ±1 位置
     限位（+1=闭、−1=开，E0 标定），不支持连续开度——协议维 0=闭/1=开 按
     0.5 阈值映射，docs 披露二值差异；
  ② robosuite 观测四元数为 xyzw（robots/robot.py:438），yaw 解析按 xyzw，
     eef_quat 对外统一转 wxyz 并在 docs 声明；
  ③ 碗体碰撞 geom 在渲染 group 0（默认隐藏），已补 group 1 视觉副本；
     离屏帧为 OpenGL 上下颠倒，翻转后交给模型，docs 声明方向。

物体真值只进 info 供评分，不进模型观测（口径 §4.2）。成功判定统一用底层
env._check_success()（含速度条件），phase/phase_max 供 0–4 最高阶段评分。

运行环境：.venv-robosuite（robosuite 1.5.2 + inspect-robots 0.58.0 同 venv）。
"""
from __future__ import annotations

import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_TABLE_TOP = 0.8
_BOWL_INNER_R = 0.049


def _wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def _yaw_from_quat_xyzw(q) -> float:
    """robosuite 观测四元数为 (x,y,z,w)（E1：robots/robot.py:438）。"""
    x, y, z, w = (float(v) for v in q)
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


class RobosuiteBowlEmbodiment:
    """红块入碗的 robosuite embodiment（Panda + OSC_POSE + 漏斗碗）。

    参数
    ----
    execution_noise:
        每控制步 EEF 目标的世界系高斯执行噪声 σ（x/y m, z m, yaw rad）；
        默认 (0.0015, 0.001, 0.017)，全 0 关闭（确定性回归）。噪声源在
        reset(seed) 时按 seed 重置，保证同 seed 轨迹可重放。
    cameras:
        False 时关闭离屏渲染与相机观测（契约/物理测试提速用）。
    """

    def __init__(self, *, img_h: int = 224, img_w: int = 224,
                 execution_noise: tuple[float, float, float] = (0.0015, 0.001, 0.017),
                 cameras: bool = True):
        from robosuite.controllers import load_composite_controller_config
        from inspect_robots.embodiment import (
            AUTO_RESET,
            PRIVILEGED_SUCCESS,
            RENDERABLE,
            RESETTABLE,
            SEEDABLE,
            EmbodimentInfo,
        )

        self._cameras_on = bool(cameras)
        self._cc = load_composite_controller_config(controller="BASIC")
        self._cc["body_parts"]["right"]["input_ref_frame"] = "world"
        self._img_h, self._img_w = img_h, img_w
        self._noise = np.asarray(execution_noise, dtype=np.float64)
        self._rng = np.random.RandomState(0)
        self._env = None
        self._phase_max = 0
        self._make_env_kwargs = dict(
            robots="Panda",
            controller_configs=self._cc,
            has_renderer=False,
            has_offscreen_renderer=self._cameras_on,
            use_camera_obs=self._cameras_on,
            camera_names=(["frontview", "sideview", "robot0_eye_in_hand"]
                          if self._cameras_on else []),
            camera_heights=img_h,
            camera_widths=img_w,
            control_freq=20,
            reward_shaping=False,
            horizon=100000,  # 终止由 harness 的 max_steps 控制
        )

        self.info = EmbodimentInfo(
            name="robosuite-bowl",
            action_space=self._box(),
            observation_space=self._obs_space(),
            control_hz=20.0,
            is_simulated=True,
            capabilities=frozenset(
                {SEEDABLE, RESETTABLE, AUTO_RESET, PRIVILEGED_SUCCESS, RENDERABLE}
            ),
            docs=self._docs(),
        )
        self._instruction: str | None = None

    # ---------------------------------------------------------------- #
    def _box(self):
        from inspect_robots.spaces import ActionSemantics, Box

        lo = np.array([-0.30, -0.30, 0.80, -np.pi, 0.0])
        hi = np.array([0.35, 0.30, 1.30, np.pi, 1.0])
        return Box(
            shape=(5,), low=lo, high=hi,
            semantics=ActionSemantics(
                control_mode="eef_abs_pose",
                rotation_repr="none",
                gripper="binary",  # 底层 PandaGripper 仅支持开/闭二值（E1）
                frame="world",
                dim_labels=("x", "y", "z", "yaw", "gripper"),
                max_step=(0.04, 0.04, 0.04, 0.35, None),
            ),
        )

    def _obs_space(self):
        from inspect_robots import StateField
        from inspect_robots.spaces import CameraSpec as CS, ObservationSpace, StateSpec

        cams = (
            (CS(name="front", height=self._img_h, width=self._img_w, channels=3),
             CS(name="side", height=self._img_h, width=self._img_w, channels=3),
             CS(name="wrist", height=self._img_h, width=self._img_w, channels=3))
            if self._cameras_on else ()
        )
        return ObservationSpace(
            cameras=cams,
            state=StateSpec(fields=(
                StateField("eef_pos", (3,), "m"),
                StateField("eef_quat", (4,), "unit_quat"),
                StateField("eef_yaw", (1,), "rad"),
                StateField("gripper", (1,), "normalized"),
                StateField("eef_state", (5,), "mixed"),
                StateField("joint_pos", (7,), "rad"),
                StateField("joint_vel", (7,), "rad/s"),
                StateField("joint_eff", (7,), "N·m"),
            )),
        )

    def _docs(self) -> str:
        return (
            "Tabletop pick-and-place: a Franka Panda arm (robosuite OSC controller) "
            "must pick a red 4cm cube and drop it into a light-blue funnel bowl. "
            "World frame: origin on the table center, +x toward the bowl (bowl at "
            "roughly x=+0.16, y=0, tabletop at z=0.80), +z up. The red cube "
            "spawns on the left half of the table (x between -0.12 and -0.04, y "
            "within +/-0.10) with a small random yaw. State 'eef_state' is the "
            "measured gripper pose in the same order and units as move_to "
            "targets: [x, y, z in meters, yaw in radians, gripper normalized "
            "0=closed..1=open]; 'eef_quat' is (w,x,y,z). Camera images follow the "
            "standard image convention (top row = top of the view): 'front' "
            "looks at the table from the front, 'side' from another angle, "
            "'wrist' looks straight down between the fingers and follows the "
            "gripper (world +x right, +y up in it). The gripper points down; yaw "
            "rotates it about the vertical axis. IMPORTANT: the gripper is "
            "binary in this backend with hysteresis -- gripper targets below "
            "0.35 command CLOSE, targets above 0.65 command OPEN, and values in "
            "between keep the previous state (so after closing you can keep "
            "moving without re-sending gripper). Intermediate openings are not "
            "supported. To grasp: with gripper 1 (open), hover ~12cm above the "
            "cube, descend slowly until the TCP is level with the cube's center "
            "(for a cube on the table that is z of roughly 0.82-0.83; the wrist "
            "view should show the cube filling the gap between the fingers), "
            "then set gripper 0 to close. Lift to z 0.95-1.00, "
            "move above the bowl center, lower to about z 0.90, then set "
            "gripper 1 to release; the bowl walls funnel inward so a near-miss "
            "drop still slides in. Avoid the arm base area. Success: cube "
            "resting inside the bowl."
        )

    # ---------------------------------------------------------------- #
    def _ensure_env(self, seed: int):
        """每个 seed 重建环境（robosuite 的放置随机性由构造期 seed 决定）。"""
        from robosuite_bowl_env import RoboBowl

        if self._env is not None:
            self._env.close()
        self._env = RoboBowl(seed=int(seed), **self._make_env_kwargs)

    def reset(self, scene, *, seed: int | None = None):
        s = int(seed) if seed is not None else 0
        self._ensure_env(s)
        self._rng = np.random.RandomState(s)  # 噪声源随 seed 重置：同 seed 轨迹可重放
        self._grip_cmd = -1.0  # 二值夹爪闩锁：初始张开
        self._phase_max = 0
        obs = self._env.reset()
        self._instruction = scene.instruction
        return self._observe(obs)

    def step(self, action):
        target = np.clip(
            np.asarray(action.data, dtype=np.float64),
            self.info.action_space.low, self.info.action_space.high)
        obs = self._env._get_observations()

        eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
        yaw_now = _yaw_from_quat_xyzw(obs["robot0_eef_quat"])

        d_xyz = 8.0 * (target[:3] - eef)
        # 世界系执行噪声（米→归一化输入，OSC 输出限幅 ±0.05m/步）
        d_xyz[:2] += self._rng.normal(0.0, self._noise[0], 2) / 0.05
        d_xyz[2] += self._rng.normal(0.0, self._noise[1]) / 0.05
        d_xyz = np.clip(d_xyz, -1.0, 1.0)
        dyaw = np.clip(4.0 * _wrap(float(target[3]) - yaw_now)
                       + self._rng.normal(0.0, self._noise[2]), -1.0, 1.0)
        # 二值夹爪 + 滞回：目标 <0.35 闭、>0.65 开、中间保持先前状态。
        # 滞回是必须的：move_to 省略 gripper 时目标=实测开度（夹持后 ~0.5），
        # 无滞回会被 0.5 阈值翻成"张开"（2026-09-15 回归实测）。
        if float(target[4]) < 0.35:
            self._grip_cmd = 1.0
        elif float(target[4]) > 0.65:
            self._grip_cmd = -1.0
        g = self._grip_cmd

        rs_action = np.concatenate([d_xyz, [0.0, 0.0, dyaw], [g]])
        obs, _, _, _ = self._env.step(rs_action)
        o = self._observe(obs)

        success = bool(self._env._check_success())  # 单一口径：底层判定（含速度）
        cube = self._cube_pos()
        bowl = np.asarray(self._env.bowl_xy)
        phase = self._phase(cube, eef, bowl)
        self._phase_max = max(self._phase_max, phase)

        from inspect_robots.types import StepResult

        return StepResult(
            observation=o,
            reward=-float(np.hypot(*(cube[:2] - bowl[:2]))),
            terminated=success,
            termination_reason="success" if success else None,
            truncated=False,
            info={"success": success, "phase": phase,
                  "phase_max": self._phase_max, "cube_pos": cube.tolist()},
        )

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    # ---------------------------------------------------------------- #
    @staticmethod
    def _cube_pos_of(env) -> np.ndarray:
        return np.array(env.sim.data.body_xpos[env.cube_body_id], dtype=np.float64)

    def _cube_pos(self) -> np.ndarray:
        return self._cube_pos_of(self._env)

    def _phase(self, cube, eef, bowl) -> int:
        """0 远 / 1 近 / 2 抓起 / 3 碗上方 / 4 入碗。carried 用真实夹持接触。"""
        grasped = bool(self._env._check_grasp(
            gripper=self._env.robots[0].gripper, object_geoms=self._env.cube))
        d_xy = float(np.hypot(*(cube[:2] - eef[:2])))
        d_bowl = float(np.hypot(*(cube[:2] - bowl[:2])))
        rest_z = _TABLE_TOP + 0.032
        lifted = cube[2] > rest_z
        carried = grasped and lifted
        if d_bowl < _BOWL_INNER_R - 0.014 and cube[2] < rest_z:
            return 4
        if carried and d_bowl < 0.10:
            return 3
        if carried or lifted:
            return 2
        if d_xy < 0.08:
            return 1
        return 0

    def _observe(self, obs) -> "object":
        from inspect_robots.types import Observation

        eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
        quat_xyzw = np.asarray(obs["robot0_eef_quat"], dtype=np.float64)
        yaw = _yaw_from_quat_xyzw(quat_xyzw)
        quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
        gap = float(np.sum(np.abs(obs["robot0_gripper_qpos"])))  # 0..0.08m
        grip = float(np.clip(gap / 0.08, 0.0, 1.0))
        images = {}
        if self._cameras_on:
            # 离屏帧为 OpenGL 上下颠倒，翻转为标准图像约定（docs 已声明）
            images = {"front": np.flipud(obs["frontview_image"]).copy(),
                      "side": np.flipud(obs["sideview_image"]).copy(),
                      "wrist": np.flipud(obs["robot0_eye_in_hand_image"]).copy()}
        return Observation(
            images=images,
            state={
                "eef_pos": eef,
                "eef_quat": quat_wxyz,
                "eef_yaw": np.array([yaw], dtype=np.float64),
                "gripper": np.array([grip], dtype=np.float64),
                "eef_state": np.array([*eef, yaw, grip], dtype=np.float64),
                "joint_pos": np.asarray(obs["robot0_joint_pos"], dtype=np.float64),
                "joint_vel": np.asarray(obs["robot0_joint_vel"], dtype=np.float64),
                "joint_eff": self._joint_eff(),
            },
            instruction=self._instruction,
        )

    def _joint_eff(self) -> np.ndarray:
        """臂关节执行器力矩（qfrc_actuator，含重力/惯性补偿），对应原评测
        观测中的 joint_eff（含重力负载的估计关节力矩）。"""
        robot = self._env.robots[0]
        model, data = self._env.sim.model, self._env.sim.data
        adrs = [model.jnt_dofadr[model.joint(j).id]
                for j in robot.robot_model.joints[:7]]
        return np.array([data.qfrc_actuator[a] for a in adrs], dtype=np.float64)
