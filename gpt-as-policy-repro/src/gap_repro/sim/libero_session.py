"""LP-A1：LIBERO session 适配器（LIBERO-PRO 计划 v0.2 §4 LP1/LP3）。

- 按 suite 名 + 任务序号创建原生 OffScreenRenderEnv；bddl/init 走 LIBERO 原生接口。
- 配对初态：init_states[i] 原样使用（不重采样、不替换）。
- 成功判定：env._check_success()（原生），适配层不再解释。
- 观测指纹：确定性摘要，用于 reset 一致性测试与漂移检测。

指纹协议（LP-A1 审查 P1 修正后的完整口径）：
- 逐位硬门只覆盖 robot+movables：`set_init_state` 经 `set_state_from_flattened`
  恢复 time/qpos/qvel，恰好等于这些键的支撑集（robosuite 1.4.0 不恢复模型场）。
- 场景 fixtures（炉灶/柜/酒架等）的 body_pos/body_quat 是模型场：每次 `env.reset()`
  在 BDDL region 内重采样（`_reset_internal`，`deterministic_reset=False`）且不被
  `set_init_state` 恢复——这是协议内在扰动，与 openpi/LIBERO-PRO 论文协议同构，
  不是缺陷；但必须显式入档并计入 LP-A2 配对噪声模型。
- fixture 位姿经 `fixture_pose_digest()` 并入 `fingerprint_full`：对场景布局漂移
  不再盲（回归测试见 tests/test_libero_session.py 的 fixture 敏感性用例）。
- 渲染图像不作精确配对依据：同状态双渲染 diff（纯渲染器）与跨 reset diff
  （渲染器+fixture 漂移）的分离归因见 scripts/lp_a1_fixture_drift_probe.py 证据。
"""
import hashlib
import os
import sys
import pathlib

import numpy as np


def _ensure_libero_on_path():
    """仅注入 sys.path（导入期需要）。环境变量副作用不得在导入期发生：
    测试进程导入本模块（即使随后整组 skip）也会被 MUJOCO_GL 污染，
    影响同进程其他模块（审计 2026-09-19 §2）。env 在 __init__ 就绪。"""
    root = os.environ.get(
        "LIBERO_REPO_ROOT", "/workspace/repo/upstream/LIBERO-PRO")
    if root and root not in sys.path:
        sys.path.insert(0, root)


def _ensure_runtime_env():
    """构造 env 前的后端/配置 pin（此时才允许全局 env 副作用）。"""
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("LIBERO_CONFIG_PATH", "/workspace/data/libero_config")


_ensure_libero_on_path()

from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

FINGERPRINT_IMAGE_KEYS = ("agentview_image", "robot0_eye_in_hand_image")
# fingerprint_state 硬门键集（与函数体循环一一对应；契约测试双向锁定）
FINGERPRINT_STATE_KEYS = (
    "robot0_joint_pos", "robot0_eef_pos", "robot0_gripper_qpos", "object-state")


class LiberoSession:
    """单环境会话：创建、配对 reset、原生 step/成功判定、观测指纹。"""

    def __init__(self, suite="libero_goal", task_index=0, resolution=224, seed=0):
        _ensure_runtime_env()
        self.suite = suite
        self.task_index = task_index
        self.resolution = resolution
        self.seed = seed
        suite_obj = benchmark.get_benchmark_dict()[suite]()
        self._suite_obj = suite_obj
        self.task = suite_obj.get_task(task_index)
        self.task_description = self.task.language
        bddl = (pathlib.Path(get_libero_path("bddl_files"))
                / self.task.problem_folder / self.task.bddl_file)
        self.bddl_file = str(bddl)
        if not pathlib.Path(self.bddl_file).exists():
            raise FileNotFoundError(f"bddl file missing: {self.bddl_file}")
        self.env = OffScreenRenderEnv(
            bddl_file_name=self.bddl_file,
            camera_heights=resolution, camera_widths=resolution)
        self.env.seed(seed)
        # 渲染预热（生产路径）：首渲染含纹理/帧缓冲暂态（LP1 冒烟观测图像差至
        # 185/255，原始 PNG 在服务器 data/ 未入库）。用一次 mini-episode 消耗
        # 首渲染，随后 reset_to 的正式观测不再含暂态；预热后渲染器行为由
        # lp_a1_fixture_drift_probe.py 的同状态双渲染对照量化。
        self.env.reset()
        self.env.set_init_state(self.init_states[0])
        for _ in range(3):
            self.env.step(list(np.array([0.0] * 6 + [-1.0])))

    @property
    def init_states(self):
        return self._suite_obj.get_task_init_states(self.task_index)

    def reset_to(self, index, wait_steps: int = 10, fixture_poses=None):
        """reset + 写入配对初态 index + 等待步后返回观测。

        wait_steps=10 对齐 openpi 客户端 num_steps_wait=10（E1）：
        set_init_state 后物理需要沉降步，渲染亦需首帧预热。
        注意：本方法不（也无法）恢复 fixture 布局——见模块 docstring；
        reset 间 scene digest 的差异是协议内在现象，用 fingerprint_full 记录。
        fixture_poses（LP-A2 消融臂专用）：给定 {name: {"body_pos": [...]}} 时，
        在 set_init_state 后把 fixture 根 body 钉到该布局——这是对论文协议的
        显式偏离（工程消融），不得用于论文可比口径。
        """
        self.env.reset()
        obs = self.env.set_init_state(self.init_states[index])
        if fixture_poses is not None:
            domain = self.env.env
            sim = domain.sim
            for name, pose in fixture_poses.items():
                body_id = sim.model.body_name2id(domain.fixtures_dict[name].root_body)
                sim.model.body_pos[body_id] = np.asarray(pose["body_pos"], dtype=np.float64)
        dummy = np.array([0.0] * 6 + [-1.0])
        for _ in range(wait_steps):
            obs, _, _, _ = self.env.step(dummy)
        return obs

    def step(self, action):
        return self.env.step(list(action))

    def is_success(self):
        # OffScreenRenderEnv(ControlEnv) 组合内层 robosuite env（self.env.env），
        # LIBERO 原生成功判定在 bddl domain 的 _check_success。
        return bool(self.env.env._check_success())

    @staticmethod
    def fingerprint_state(obs) -> str:
        """movables+robot 状态指纹（逐位硬门）：FINGERPRINT_STATE_KEYS 字节摘要。

        语义边界（LP-A1 审查 P1 修正）：此硬门恰好在 set_init_state 的恢复集上
        成立，对场景 fixtures 漂移零灵敏度是构造性的——场景布局一致性请用
        fixture_pose_digest / fingerprint_full，勿以本指纹单独声明"reset 稳定"。
        """
        h = hashlib.sha256()
        for key in FINGERPRINT_STATE_KEYS:
            h.update(np.ascontiguousarray(obs[key]).astype(np.float64).tobytes())
        return h.hexdigest()

    def fixture_poses(self) -> dict:
        """场景 fixtures 根 body 的世界位姿（证据用原始值）。

        返回 {fixture名: {"body_pos": [x,y,z], "body_quat": [w,x,y,z]}}，
        按 fixture 名排序。fixtures 不在 obs / object-state 内，只能从模型场读。
        """
        domain = self.env.env
        sim = domain.sim
        poses = {}
        for name in sorted(domain.fixtures_dict.keys()):
            body_id = sim.model.body_name2id(domain.fixtures_dict[name].root_body)
            poses[name] = {
                "body_pos": [float(v) for v in sim.model.body_pos[body_id]],
                "body_quat": [float(v) for v in sim.model.body_quat[body_id]],
            }
        return poses

    def fixture_pose_digest(self) -> str:
        """场景布局指纹：全部 fixture 根 body 位姿的字节摘要。

        reset 间预期不同（BDDL region 内重采样）；同一次 episode 内确定。
        无 fixture 的任务族退化为常数摘要（此时 fingerprint_full ≡
        fingerprint_state，硬门已覆盖全部可动状态——审查 P3-5 记录）。
        注意覆盖范围是根 body 位姿：fixture 内部关节（如柜门）由 set_init_state
        的 qpos 恢复保证确定性，不在本摘要内（审查 P3-4，带门任务族接入前补）。
        """
        h = hashlib.sha256()
        for name, pose in self.fixture_poses().items():
            h.update(name.encode())
            h.update(np.ascontiguousarray(pose["body_pos"]).astype(np.float64).tobytes())
            h.update(np.ascontiguousarray(pose["body_quat"]).astype(np.float64).tobytes())
        return h.hexdigest()

    def fingerprint_full(self, obs) -> str:
        """配对身份（LP-A2 用）：movables 硬门摘要 ‖ scene 摘要。

        这是覆盖面完整的状态身份：同一 episode 内逐位确定；跨 reset 时若
        scene 部分变化即为 fixture 重采样（协议内在），配对分析按协变量处理。
        """
        h = hashlib.sha256()
        h.update(bytes.fromhex(self.fingerprint_state(obs)))
        h.update(bytes.fromhex(self.fixture_pose_digest()))
        return h.hexdigest()

    @staticmethod
    def image_diff_stats(o1, o2, key="agentview_image") -> dict:
        """两次观测的同名图像差统计（容差判定用）。

        key 可选 "agentview_image" 或 "robot0_eye_in_hand_image"（后者为策略
        实际输入，LP-A1 审查 P2-1 后纳入监控）。
        """
        d = np.abs(o1[key].astype(int) - o2[key].astype(int))
        dsum = d.sum(axis=2)
        return {"mean_abs": round(float(d.mean()), 4),
                "frac_gt0": round(float((dsum > 0).mean()), 4),
                "frac_gt30": round(float((dsum > 30).mean()), 4),
                "max_sum": int(dsum.max())}

    @staticmethod
    def fingerprint(obs) -> str:
        """完整观测指纹（状态+图像字节）——仅用于可复现性记录，
        reset 一致性判定请用 fingerprint_state（图像不逐位稳定，见模块 docstring）。"""
        h = hashlib.sha256()
        for key in FINGERPRINT_IMAGE_KEYS:
            h.update(np.ascontiguousarray(obs[key]).tobytes())
        for key in FINGERPRINT_STATE_KEYS:
            h.update(np.ascontiguousarray(obs[key]).astype(np.float64).tobytes())
        return h.hexdigest()
