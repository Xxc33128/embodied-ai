"""robosuite bowl 任务（新口径 §4.1：bowl 化场景）。

在 robosuite Lift 基础上：桌面加一只静态八边形碗（内半径 ~4.9cm，兼容 4cm
方块），成功判定 = 方块静置碗内。夹爪/臂控制全部用 robosuite 现成 OSC
控制器——本文件不含任何自研控制代码。

运行（隔离后端环境）：
  .venv-robosuite/bin/python robosuite_bowl_env.py [seed 数]
"""
from __future__ import annotations

import os

import numpy as np
from robosuite.environments.manipulation.lift import Lift
from robosuite.models.base import MujocoXML
from robosuite.models.tasks import ManipulationTask

_HERE = os.path.dirname(os.path.abspath(__file__))
_TABLE_TOP = 0.8  # TableArena tabletop（= table_offset[2]，E2 实测）
_BOWL_XY = np.array([0.16, 0.0])
_BOWL_INNER_R = 0.049
_CUBE_HALF = 0.021


class RoboBowl(Lift):
    """Lift 的 bowl 版：同一只红方块，改为放入桌面碗内。"""

    def _load_model(self):
        # 与 Lift._load_model 相同的桌/臂/方块装配，追加静态碗并纳入任务
        super()._load_model()  # 先装载机器人模型
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        from robosuite.models.arenas import TableArena

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        tex_attrib = {"type": "cube"}
        mat_attrib = {"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"}
        from robosuite.models.objects import BoxObject
        from robosuite.utils.mjcf_utils import CustomMaterial

        redwood = CustomMaterial(
            texture="WoodRed", tex_name="redwood", mat_name="redwood_mat",
            tex_attrib=tex_attrib, mat_attrib=mat_attrib,
        )
        self.cube = BoxObject(
            name="cube", size_min=[0.020, 0.020, 0.020], size_max=[0.022, 0.022, 0.022],
            rgba=[1, 0, 0, 1], material=redwood, rng=self.rng,
        )
        # 碗 = 静态布景，直接并入 arena（不走 object 系统：无需放置/抓取语义）
        mujoco_arena.merge(MujocoXML(os.path.join(_HERE, "robosuite_bowl_asset.xml")))

        # 方块生成在左半桌、小幅随机 yaw——与 embodiment docs 的说明一致
        # （复核 2026-09-15：父类的 [-0.03,0.03] 采样器与说明冲突，这里强制覆盖）
        from robosuite.utils.placement_samplers import UniformRandomSampler

        self.placement_initializer = UniformRandomSampler(
            name="ObjectSampler", mujoco_objects=self.cube,
            x_range=[-0.12, -0.04], y_range=[-0.10, 0.10],
            rotation=[-0.1745, 0.1745], ensure_object_boundary_in_range=False,
            ensure_valid_placement=True, reference_pos=self.table_offset,
            z_offset=0.01, rng=self.rng,
        )
        self.placement_initializer.reset()
        self.placement_initializer.add_objects(self.cube)

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.cube,
        )

    def _setup_references(self):
        super()._setup_references()
        self.cube_body_id = self.sim.model.body_name2id(self.cube.root_body)
        self.bowl_body_id = self.sim.model.body_name2id("bowl")

    @property
    def bowl_xy(self) -> np.ndarray:
        return np.array(self.sim.data.body_xpos[self.bowl_body_id][:2])

    def _check_success(self):
        """方块静置碗内：中心水平距 < 内半径-2cm、低于碗口、近静止。"""
        cube = np.array(self.sim.data.body_xpos[self.cube_body_id])
        bowl = np.array(self.sim.data.body_xpos[self.bowl_body_id])
        r_xy = float(np.hypot(*(cube[:2] - bowl[:2])))
        low = cube[2] < _TABLE_TOP + 0.03
        name = self.sim.model.body_id2name(self.cube_body_id)
        vel = float(np.linalg.norm(self.sim.data.get_body_xvelp(name)))
        return bool(r_xy < _BOWL_INNER_R - 0.014 and low and vel < 0.15)

    def _reward(self):
        return 1.0 if self._check_success() else 0.0


def scripted_solve(seed: int, max_steps: int = 400, verbose: bool = False) -> dict:
    """独立环境验证脚本（仅验环境，不进正式评测）：OSC 抓方块放入碗。"""
    from robosuite.controllers import load_composite_controller_config

    _cc = load_composite_controller_config(controller="BASIC")
    _cc["body_parts"]["right"]["input_ref_frame"] = "world"
    env = RoboBowl(
        robots="Panda",
        controller_configs=_cc,
        seed=seed,
        has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
        control_freq=20, reward_shaping=False,
    )
    obs = env.reset()
    kp = 6.0
    bowl_xy = env.bowl_xy
    table_top = _TABLE_TOP
    phase, ps = "approach", 0
    success = False
    for t in range(max_steps):
        eef = obs["robot0_eef_pos"]
        cube = obs["cube_pos"]
        gq = obs["robot0_gripper_qpos"]
        closed = bool(np.all(np.abs(gq) < 0.015))
        if phase == "approach":       # 悬停方块上方
            target = cube + [0, 0, 0.12]; g = -1.0
            if np.linalg.norm(eef - target) < 0.02:
                phase, ps = "descend", 0
        elif phase == "descend":      # 下探到方块中心高
            target = cube + [0, 0, 0.004]; g = -1.0
            if np.linalg.norm(eef - target) < 0.012 or ps > 70:
                phase, ps = "close", 0
        elif phase == "close":        # 闭合
            target = eef.copy(); g = 1.0
            if (closed and ps > 15) or ps > 60:
                phase, ps = "lift", 0
        elif phase == "lift":         # 垂直提升（绝对目标，防追赶漂移）
            target = np.array([eef[0], eef[1], _TABLE_TOP + 0.28]); g = 1.0
            if ps > 45:
                phase, ps = "carry", 0
        elif phase == "carry":        # 平移到碗上方
            target = np.array([bowl_xy[0], bowl_xy[1], eef[2]]); g = 1.0
            if np.linalg.norm(eef[:2] - bowl_xy) < 0.015:
                phase, ps = "release", 0
        elif phase == "release":      # 深入碗口再张开
            target = eef + [0, 0, -0.05]; g = -1.0
            if ps > 30:
                phase, ps = "retract", 0
        elif phase == "retract":      # 撤爪：指尖离开碗口再判静置
            target = eef + [0, 0, 0.06]; g = -1.0
            if ps > 20:
                phase, ps = "settle", 0
        else:                         # 静置等待：方块落底且近静止
            target = eef.copy(); g = -1.0
            cz = float(obs["cube_pos"][2])
            v = float(np.linalg.norm(env.sim.data.get_body_xvelp("cube_main")))
            if (cz < _TABLE_TOP + 0.03 and v < 0.1 and ps > 20) or ps > 120:
                success = env._check_success()
                break
        d = np.clip(kp * (target - eef), -1.0, 1.0)
        dyaw = 0.0
        action = np.concatenate([d, [0.0, 0.0, dyaw], [g]])
        obs, _, done, _ = env.step(action)
        ps += 1
        success = env._check_success()
        if verbose and t % 20 == 0:
            cz = float(obs["cube_pos"][2])
            print(f"    t={t:3d} phase={phase:8s} eef_z={eef[2]:.3f} cube_z={cz:.3f} ok={success}")
        if done or (success and ps > 30 and phase in ("release", "settle")):
            break
    out = {"seed": seed, "success": bool(success), "steps": t + 1,
           "cube": np.round(obs["cube_pos"], 3).tolist(), "bowl_xy": bowl_xy.round(3).tolist()}
    env.close()
    return out


if __name__ == "__main__":
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    results = [scripted_solve(1000 + i) for i in range(n)]
    ok = sum(r["success"] for r in results)
    print(f"scripted bowl solve: {ok}/{n}")
    for r in results:
        print(f"  seed={r['seed']} success={r['success']} steps={r['steps']} "
              f"cube={r['cube']} bowl={r['bowl_xy']}")
