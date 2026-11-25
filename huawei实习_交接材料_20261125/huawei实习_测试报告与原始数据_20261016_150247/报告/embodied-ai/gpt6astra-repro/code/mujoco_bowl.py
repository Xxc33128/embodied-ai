"""MuJoCo "bowl task" embodiment v3：Franka Panda 关节臂版。

协议对齐 RoboCurve 真机形态（差距与对齐清单见保真度审计文档）：
  - 动作空间 eef_abs_pose（x,y,z,yaw,gripper）→ LLM 工具面为 move_to（含旋转）
  - 框架侧把绝对位姿按 max_step 安全插值 + 机器人侧 IK 解关节角：
    与 RoboCurve 完全同构（模型只见末端位姿，机器人侧负责运动学）
  - 观测 = 本体状态（eef/gripper/yaw，无物体位姿）+ 三相机帧（顶/侧/腕载）
  - 阶段评分 0-4（远/近/抓起/碗上方/入碗）进 info
  - 可配置执行噪声（位置/yaw 高斯扰动）与 reset 随机化（方块位姿/姿态）

本体：MuJoCo Menagerie 的 Franka Panda（7 关节位置伺服 + 腱驱动双指），
底座安装在工作区后侧。IK = 阻尼最小二乘（xyz + yaw 四维误差）+ 关节限位
钳制 + 零空间回中（趋向 home 姿态）；每物理子步解算，warm start。
腕载视角 = 高清俯视在 eef 处的裁剪（像素映射 5 点仿射自标定）。
"""

from __future__ import annotations

import os
import re

import numpy as np

from inspect_robots.spaces import ActionSemantics, Box, CameraSpec, ObservationSpace
from inspect_robots.types import Observation, StepResult

_IMG_H, _IMG_W = 240, 320
_HERE = os.path.dirname(os.path.abspath(__file__))
_PANDA_DIR = os.path.join(_HERE, "mujoco_menagerie", "franka_emika_panda")

# 场景常量（世界系原点 = 桌面中心，+z 向上）
_BOWL_XY = np.array([0.2, 0.0])
_BOWL_R = 0.055          # 漏斗底部内半宽（成功判定半径基准）
_CUBE_HALF = 0.02        # 红方块半边长
_EEF_HOME = np.array([-0.05, 0.0, 0.28])   # eef 悬停复位位（桌面上方 28cm）
_EEF_Z_FLOOR = 0.03      # TCP 下限（指尖≈TCP-0.01，不穿桌）
_GRASP_TCP_OFF = 0.018   # 抓取下探偏移（TCP≈38.8mm）。Task 4 实测：TCP 对齐方块
                         # 中心（+0.001）在运动学下限之下不可达（IK 发散）；长垫
                         # 配置下更低会 pad 磕桌。覆盖窗权衡见 Task 4 周报。
_POS_LIM = np.array([0.38, 0.30, 0.40])
_MAX_POS_STEP = 0.04      # 每控制步平移上限（m）
_MAX_YAW_STEP = 0.35      # 每控制步 yaw 上限（rad）
_SUBSTEPS = 50            # 物理子步（dt=0.002 → 控制周期 0.1s）
_TOP_FOVY_DEG = 45.0
_WRIST_WIN = (0.20, 0.15) # 腕视裁剪窗口（米）

# Panda 安装位：工作区后左角（桌面上），远离方块区与碗
_PANDA_BASE_XY = (-0.30, -0.24)
# Panda home 关节角（"ready" 姿态：指向工作区上方）
_Q_HOME = np.array([0.0, -0.45, 0.0, -2.1, 0.0, 1.9, 0.785])
# 标定/躲让姿态：臂向后折叠，不遮挡桌面中央
_Q_STOW = np.array([0.0, 0.8, 0.0, -1.2, 0.0, 1.5, 0.79])
# 夹爪腱控制（Menagerie 映射：ctrl 0-255 ↔ 指总开距 0-0.08m）


def _grip_ctrl(g: float) -> float:
    """gripper 0(闭)→1(开) 映射到腱 ctrl。闭合目标总间隙 0.026（方块 0.04，
    每侧 7mm 挤压量）——位置伺服的挤压深度即抓持法向力来源。"""
    return (0.026 + g * (0.076 - 0.026)) / 0.08 * 255


_TRAY_HALF = 0.055
_WALL_T = 0.004
_WALL_H = 0.035
_WALL_TILT = 20.0


def _walls_xml(*, walls_radian: bool = True) -> str:
    """walls_radian=False 复刻历史单位错误（euler 数值 20 被 radian compiler
    解释为 20rad≈65.92°），仅用于修复前后消融对照，勿用于正式评测。"""
    tilt = float(np.radians(_WALL_TILT)) if walls_radian else float(_WALL_TILT)
    out = ""
    for i, (px, py, sx, sy, eu) in enumerate([
        (_BOWL_XY[0], _BOWL_XY[1] + _TRAY_HALF + _WALL_T, _TRAY_HALF + _WALL_T, _WALL_T, f"{-tilt:.6f} 0 0"),
        (_BOWL_XY[0], _BOWL_XY[1] - _TRAY_HALF - _WALL_T, _TRAY_HALF + _WALL_T, _WALL_T, f"{tilt:.6f} 0 0"),
        (_BOWL_XY[0] - _TRAY_HALF - _WALL_T, _BOWL_XY[1], _WALL_T, _TRAY_HALF, f"0 {-tilt:.6f} 0"),
        (_BOWL_XY[0] + _TRAY_HALF + _WALL_T, _BOWL_XY[1], _WALL_T, _TRAY_HALF, f"0 {tilt:.6f} 0"),
    ]):
        sx_p = sx / np.cos(tilt) if sx > sy else sx
        sy_p = sy / np.cos(tilt) if sy > sx else sy
        out += f"""
    <body name="bowl_wall_{i}" pos="{px:.4f} {py:.4f} {_WALL_H / 2:.4f}" euler="{eu}">
      <geom type="box" size="{sx_p:.4f} {sy_p:.4f} {_WALL_H / 2:.4f}" rgba="0.35 0.4 0.5 1"
            friction="1.0 0.003 0.0002"/>
    </body>"""
    return out


def _panda_sections() -> tuple[str, str, str, str, str, str]:
    """读 Menagerie panda.xml，返回出厂件 (defaults, meshes, worldbody 内容,
    actuators, tendon, equality_contact)，只做结构性修改（meshdir 绝对路径、
    TCP site、安装底座包裹）；执行器/指垫参数改装在 build_xml 按旋钮进行。

    equality（双指 joint 耦合，闭合同步的关键约束）与 contact（link0-link1
    exclude，父子对默认已过滤、恢复只为模型语义忠实）必须原样保留——v3 曾漏掉
    这两节，实测 neq/nexclude=0（test_bowl_model 把关）。
    """
    xml = open(os.path.join(_PANDA_DIR, "panda.xml")).read()
    assets = os.path.join(_PANDA_DIR, "assets")
    xml = xml.replace('meshdir="assets"', f'meshdir="{assets}"')

    defaults = re.search(r"<default>(.*)</default>", xml, re.S).group(1)  # 取内层（嵌套无名 default 不合法）
    meshes = re.search(r"<asset>(.*)</asset>", xml, re.S).group(1)
    tendon = re.search(r"<tendon>(.*)</tendon>", xml, re.S).group(1)
    wb = re.search(r"<worldbody>(.*)</worldbody>", xml, re.S).group(1)
    acts = re.search(r"<actuator>(.*)</actuator>", xml, re.S).group(1)
    eq_contact = (
        "<contact>"
        + re.search(r"<contact>(.*)</contact>", xml, re.S).group(1)
        + "</contact>\n  <equality>"
        + re.search(r"<equality>(.*)</equality>", xml, re.S).group(1)
        + "</equality>"
    )

    # TCP site：加在 hand body 内（指尖之间）
    wb = wb.replace(
        '<body name="hand" pos="0 0 0.107" quat="0.9238795 0 0 -0.3826834">',
        '<body name="hand" pos="0 0 0.107" quat="0.9238795 0 0 -0.3826834">\n'
        '                    <site name="eef_site" pos="0 0 0.1" size="0.004" rgba="1 0.8 0.1 0.6"/>',
        1,
    )
    bx, by = _PANDA_BASE_XY
    wb = f"""
    <body name="panda_base" pos="{bx} {by} 0">
{wb}
    </body>"""
    return defaults, meshes, wb, acts, tendon, eq_contact


_D, _M, _W, _A, _T, _EQC = _panda_sections()
_PANDA_ASSETS = os.path.join(_PANDA_DIR, "assets")


def build_xml(*, equality: bool = True, walls_radian: bool = True,
              grip_kp: float = 800.0, grip_kv: float = 28.0, pad_mu: float = 2.0,
              pad_len: float = 0.022, pad_zoff: float = -0.0045,
              pad_solref: str | None = None, pad_solimp: str | None = None,
              cone: str | None = None, noslip: int = 0) -> str:
    """组装完整场景 XML。equality/walls_radian 供 Task 2 消融；grip_*/pad_*/
    cone/noslip 供 Task 4 单因素实验。pad 默认 = Task 4 几何修正配置。

    gain 项随 grip_kp 等比缩放（g = kp*0.04/255），保持 _grip_ctrl 的
    ctrl→间隙语义不变（ctrl=82.8 ↔ 26mm 自由间隙，E0 标定）。"""
    # 主指垫加长 + 全部指垫高摩擦。TCP 工作下限 ~35mm（更低 IK 发散，E2），
    # pad 覆盖窗 [TCP-33.3, TCP+26.8]mm 才能包住桌面方块相对窗口
    # [TCP-34, +6]mm——旧 22mm 短垫只捏方块上半部，悬臂蠕动必脱（Task 4）。
    defaults = _D.replace(
        '<geom type="box" size="0.0085 0.004 0.0085" pos="0 0.0055 0.0445"/>',
        f'<geom type="box" size="0.0085 0.004 {pad_len:.4f}" pos="0 0.0055 {0.0445 + pad_zoff:.4f}"/>')
    for i in range(1, 6):
        extra = ""
        if pad_solref:
            extra += f' solref="{pad_solref}"'
        if pad_solimp:
            extra += f' solimp="{pad_solimp}"'
        defaults = re.sub(
            rf'(class="fingertip_pad_collision_{i}">\s*<geom[^/>]*)/>',
            rf'\1 friction="{pad_mu} 0.008 0.0004"{extra}/>',
            defaults)
    # 夹爪腱伺服：出厂 kp=100 挤压力仅 0.7N；kp=800 时卡在 40mm 方块上的
    # 腱力 = g*ctrl - kp*L = -5.6N（负=闭合），每指 2.8N（E0 标定映射）。
    gain = grip_kp * 0.04 / 255
    acts = _A.replace('gainprm="0.01568627451 0 0" biasprm="0 -100 -10"',
                      f'gainprm="{gain:.11g} 0 0" biasprm="0 {-grip_kp} {-grip_kv}"')
    eq_xml = f"\n  {_EQC}" if equality else ""
    opt = '<option timestep="0.002" integrator="implicitfast" impratio="25"'
    if cone is not None:
        opt += f' cone="{cone}"'
    if noslip:
        opt += f' noslip_iterations="{noslip}" noslip_tolerance="1e-6"'
    opt += "/>"
    return f"""
<mujoco model="bowl-task-panda">
  <compiler angle="radian" autolimits="true" meshdir="{os.path.join(_PANDA_DIR, 'assets')}"/>
  {opt}
  <visual>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6"/>
  </visual>
  <asset>
{_M}
    <texture type="skybox" builtin="gradient" rgb1="0.6 0.7 0.9" rgb2="0.25 0.3 0.45" width="32" height="32"/>
    <material name="table" rgba="0.55 0.5 0.45 1"/>
  </asset>
  <default>
{defaults}
  </default>
  <worldbody>
    <light pos="0.1 0 1.2" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.5 0.55 0.6 1"/>
    <geom name="table" type="box" pos="0 0 -0.025" size="0.4 0.3 0.025" material="table"/>

    <body name="cube" pos="-0.1 0 0.021">
      <freejoint name="cube_joint"/>
      <geom name="cube_geom" type="box" size="{_CUBE_HALF} {_CUBE_HALF} {_CUBE_HALF}"
            rgba="0.85 0.08 0.08 1" friction="1.6 0.003 0.0002"/>
    </body>
{_walls_xml(walls_radian=walls_radian)}
    <geom name="bowl_visual" type="cylinder" pos="{_BOWL_XY[0]} {_BOWL_XY[1]} 0.024"
          size="0.062 0.024" rgba="0.3 0.45 0.65 0.3" contype="0" conaffinity="0" group="3"/>
{_W}
  </worldbody>{eq_xml}
  <tendon>
{_T}
  </tendon>
  <actuator>
{acts}
  </actuator>
</mujoco>
"""


_XML = build_xml()

# 参考工具姿态：site x=(1,0,0), y=(0,-1,0), z=(0,0,-1)（列向量）
_R_REF = np.array([[1.0, 0.0, 0.0],
                   [0.0, -1.0, 0.0],
                   [0.0, 0.0, -1.0]])


def _rot_z(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                     w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2])


def _orient_error(r_cur: np.ndarray, yaw_t: float) -> np.ndarray:
    """当前姿态 → 目标姿态（Rz(yaw)·R_ref）的轴角误差（世界系，rad）。"""
    q_t = np.empty(4)
    m = (_rot_z(yaw_t) @ _R_REF).flatten()
    from mujoco import mju_mat2Quat
    mju_mat2Quat(q_t, m)
    q_c = np.empty(4)
    mju_mat2Quat(q_c, r_cur.flatten())
    q_c_conj = q_c * np.array([1.0, -1.0, -1.0, -1.0])
    q_e = _quat_mul(q_t, q_c_conj)
    if q_e[0] < 0:
        q_e = -q_e
    v_norm = np.linalg.norm(q_e[1:])
    if v_norm < 1e-9:
        return np.zeros(3)
    theta = 2.0 * np.arctan2(v_norm, q_e[0])
    return q_e[1:] / v_norm * theta


def _yaw_to_quat(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([c, 0.0, 0.0, s])


def _wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


class MuJoCoBowlEmbodiment:
    """抓红块放碗里的 MuJoCo 仿真本体（Panda 关节臂 + eef_abs_pose + 机器人侧 IK）。

    参数
    ----
    execution_noise:
        每控制步 eef 目标的高斯执行噪声 σ，(x/y m, z m, yaw rad)；默认
        (0.0015, 0.001, 0.017)；传全 0 关闭（确定性回归测试用）。
    """

    def __init__(self, *, img_h: int = _IMG_H, img_w: int = _IMG_W,
                 execution_noise: tuple[float, float, float] = (0.0015, 0.001, 0.017),
                 xml: str | None = None,
                 grip_mode: str = "position", grip_force: float = 12.0):
        """xml: 覆盖组装 XML（仅消融/诊断用；None=build_xml() 修正配置）。

        grip_mode: "position"=腱位置伺服（历史行为）；"force"=限力保持——
            闭合指令下首次双侧接触即闩锁，ctrl 固定为 (kp·L₀−F)/g（L₀=闩锁
            时刻腱长）：等效把位置伺服虚目标锚定在接触面内侧 F/kp 处，既
            施加 F 的额外夹持力，又不允许间隙越过闩锁值。E2 教训：纯力反馈
            （每步逆映射）无位置锚，接触蠕变会让指间缓慢合拢、蠕吞方块。
        grip_force: 闩锁后的目标夹持力增量 [N]（腱力模长；每指接触力≈F/2），
            须 < kp·L₀ 否则退化为全闭位置伺服。"""
        import mujoco  # 惰性导入，保持与插件一致的风格

        from inspect_robots import StateField, StateSpec
        from inspect_robots.embodiment import (
            AUTO_RESET,
            PRIVILEGED_SUCCESS,
            RENDERABLE,
            RESETTABLE,
            SEEDABLE,
            EmbodimentInfo,
        )

        self._mj = mujoco
        self._model = mujoco.MjModel.from_xml_string(_XML if xml is None else xml)
        self._data = mujoco.MjData(self._model)

        # Panda 关节/执行器索引
        self._jids = [self._model.joint(f"joint{i}").id for i in range(1, 8)]
        self._qadr = np.array([self._model.jnt_qposadr[j] for j in self._jids])
        self._dadr = np.array([self._model.jnt_dofadr[j] for j in self._jids])
        self._act_arm = [self._model.actuator(f"actuator{i}").id for i in range(1, 8)]
        self._act_grip = self._model.actuator("actuator8").id
        self._finger_qadr = np.array([
            self._model.jnt_qposadr[self._model.joint(f"finger_joint{i}").id]
            for i in (1, 2)])
        self._ctrl_lo = self._model.actuator_ctrlrange[np.array(self._act_arm), 0]
        self._ctrl_hi = self._model.actuator_ctrlrange[np.array(self._act_arm), 1]
        self._site = self._model.site("eef_site").id
        self._cube_bid = self._model.body("cube").id
        self._cube_qadr = self._model.jnt_qposadr[self._model.joint("cube_joint").id]
        self._img_h, self._img_w = img_h, img_w
        self._renderers: dict[str, "mujoco.Renderer"] = {}
        self._noise = np.asarray(execution_noise, dtype=np.float64)
        self._rng = np.random.RandomState(0)

        self.info = EmbodimentInfo(
            name="mujoco-bowl",
            action_space=Box(
                shape=(5,),
                low=np.array([-_POS_LIM[0], -_POS_LIM[1], _EEF_Z_FLOOR, -np.pi, 0.0]),
                high=np.array([_POS_LIM[0], _POS_LIM[1], _POS_LIM[2], np.pi, 1.0]),
                semantics=ActionSemantics(
                    control_mode="eef_abs_pose",
                    rotation_repr="none",  # yaw 作为普通连续维（可按维安全插值）
                    gripper="continuous",
                    frame="base",
                    dim_labels=("x", "y", "z", "yaw", "gripper"),
                    max_step=(_MAX_POS_STEP,) * 3 + (_MAX_YAW_STEP, 1.0),
                ),
            ),
            observation_space=ObservationSpace(
                cameras=(
                    CameraSpec(name="top", height=img_h, width=img_w, channels=3),
                    CameraSpec(name="side", height=img_h, width=img_w, channels=3),
                    CameraSpec(name="wrist", height=img_h, width=img_w, channels=3),
                ),
                state=StateSpec(fields=(
                    StateField("eef_pos", (3,), "m"),
                    StateField("eef_quat", (4,), "unit_quat"),
                    StateField("eef_yaw", (1,), "rad"),
                    StateField("gripper", (1,), "normalized"),
                    # move_to 的唯一本体参考态：与动作同序同纲（agent 插件
                    # 要求 absolute 控制必须有且仅有一个 shape=(dim,) 字段）
                    StateField("eef_state", (5,), "mixed"),
                )),
            ),
            control_hz=10.0,
            is_simulated=True,
            capabilities=frozenset(
                {SEEDABLE, RESETTABLE, AUTO_RESET, PRIVILEGED_SUCCESS, RENDERABLE}
            ),
            docs=(
                "Tabletop pick-and-place with a Franka Panda arm mounted at the back-left "
                "corner of the table. World/base frame: origin at the CENTER of the table "
                "(not the arm base), +x toward the bowl (which sits at roughly x=+0.20, "
                "y=0), +y right, +z up. The red 4cm cube starts somewhere on the left "
                "half of the table (x<0) with a small random yaw. State 'eef_state' is "
                "the measured gripper pose in the same order and units as move_to "
                "targets: [x, y, z in meters, yaw in radians, gripper normalized "
                "0=closed..1=open]. The overview (top) camera shows the whole table from "
                "an elevated front angle (the arm itself is visible at the back-left). "
                "The wrist camera is a close view looking straight down between the open "
                "fingers, following the gripper; in it, world +x points right and +y "
                "points up. Gripper: two fingers close along the gripper y axis "
                "(gripper 0=closed, 1=open); the gripper points straight down; yaw "
                "rotates it about the vertical axis. To grasp the cube: with gripper "
                "open, move directly above it, then descend slowly until the fingers "
                "straddle the 4cm cube body -- for a cube resting on the table that is "
                "a target z of roughly 0.035-0.05 (in the wrist view the cube should "
                "fill the gap between the fingers, its top near the fingertips). Then "
                "set gripper 0 to close, lift to z 0.15+, translate over the bowl, keep "
                "gripper 0 while carrying, hover over the bowl center at z 0.10-0.12, "
                "then set gripper 1 to release; the bowl walls are funneled inward so a "
                "near-miss drop still slides in. Avoid the arm base area (back-left "
                "corner). Success: cube resting inside the bowl."
            ),
        )
        self._instruction: str | None = None
        self._grip = 1.0
        # 力控参数（从编译后模型读回，避免与 XML 旋钮漂移）
        assert grip_mode in ("position", "force"), grip_mode
        self._grip_mode = grip_mode
        self._grip_force = float(grip_force)
        self._grip_gain = float(self._model.actuator_gainprm[self._act_grip][0])
        self._grip_kp = -float(self._model.actuator_biasprm[self._act_grip][1])
        self._grip_kv = -float(self._model.actuator_biasprm[self._act_grip][2])
        self._grip_latch = False
        # 执行层重力沉降补偿（Task 3，E2 证据：位置伺服无重力补偿，静态
        # TCP 残差 5-15mm；真机内环有重力补偿）。近目标(±1cm)且未闭合夹持
        # 时积分累积，夹持中冻结防止把挤压力越推越大；新指令阶段重置。
        self._bias = np.zeros(3)
        self._last_cmd_pos: np.ndarray | None = None
        # 用 IK 从种子姿态解出有效 home（TCP 悬停工作区中央上方 28cm）
        self._q_home_eff = self._ik_static(np.array(_EEF_HOME), 0.0, _Q_HOME, iters=300)

    # ------------------------------------------------------------------ #
    def reset(self, scene, *, seed: int | None = None) -> Observation:
        rng = np.random.RandomState(seed if seed is not None else 0)
        self._rng = np.random.RandomState((seed or 0) + 777)
        mujoco = self._mj
        mujoco.mj_resetData(self._model, self._data)

        # 方块位置 + 小角度姿态随机
        cube_x = rng.uniform(-0.18, -0.02)
        cube_y = rng.uniform(-0.10, 0.10)
        q = _yaw_to_quat(rng.uniform(-np.pi / 18, np.pi / 18))
        self._data.qpos[self._cube_qadr:self._cube_qadr + 7] = np.concatenate(
            [[cube_x, cube_y, _CUBE_HALF + 0.001], q])

        # 臂回 home；夹爪张开；伺服稳定
        self._data.qpos[self._qadr] = self._q_home_eff
        self._data.ctrl[self._act_arm] = self._q_home_eff
        self._data.ctrl[self._act_grip] = _grip_ctrl(1.0)
        self._grip = 1.0
        for _ in range(400):
            mujoco.mj_step(self._model, self._data)

        self._instruction = scene.instruction
        self._grip_latch = False
        return self._observe()

    def step(self, action) -> StepResult:
        mujoco = self._mj
        target = np.clip(np.asarray(action.data, dtype=np.float64),
                         self.info.action_space.low, self.info.action_space.high)

        cur_pos = self._site_pos().copy()
        cur_yaw = self._site_yaw()

        pos_delta = np.clip(target[:3] - cur_pos, -_MAX_POS_STEP, _MAX_POS_STEP)
        yaw_delta = float(np.clip(_wrap(float(target[3]) - cur_yaw), -_MAX_YAW_STEP, _MAX_YAW_STEP))
        grip_start = self._grip
        grip_end = float(np.clip(target[4], 0.0, 1.0))

        # 执行噪声（eef 目标侧，IK 之前的真机残差模型）
        pos_delta[:2] += self._rng.normal(0.0, self._noise[0], 2)
        pos_delta[2] += self._rng.normal(0.0, self._noise[1])
        yaw_delta += float(self._rng.normal(0.0, self._noise[2]))

        new_pos = np.clip(cur_pos + pos_delta,
                          [self.info.action_space.low[0], self.info.action_space.low[1], _EEF_Z_FLOOR],
                          [self.info.action_space.high[0], self.info.action_space.high[1], _POS_LIM[2]])

        # 重力沉降积分补偿：边界姿态的系统性偏移实测 ~11-18mm（joint4 顶限位
        # 的可达折衷），故门宽 25mm；夹爪未闭合夹持时更新，夹持中冻结防止
        # 把挤压力越推越大；新指令阶段重置。
        if self._last_cmd_pos is None or float(np.linalg.norm(new_pos - self._last_cmd_pos)) > 0.005:
            self._bias = np.zeros(3)
        self._last_cmd_pos = new_pos.copy()
        err_v = new_pos - cur_pos
        if float(np.linalg.norm(err_v)) < 0.025 and target[4] >= 0.5:
            self._bias = np.clip(self._bias + 0.3 * err_v, -0.025, 0.025)
        ik_tgt_base = np.clip(new_pos + self._bias,
                              [self.info.action_space.low[0], self.info.action_space.low[1], _EEF_Z_FLOOR],
                              [self.info.action_space.high[0], self.info.action_space.high[1], _POS_LIM[2]])

        # 限力保持闩锁（Task 4 E1）：闭合中首次双侧接触 → 锚定间隙 + 增量夹持力
        if self._grip_mode == "force" and grip_end < 0.5:
            if not self._grip_latch:
                sc = _finger_contact_scan(self)
                if sc["nL"] > 0 and sc["nR"] > 0:
                    self._grip_latch = True
                    L0 = float(self._data.actuator_length[self._act_grip])
                    self._grip_ctrl_hold = float(np.clip(
                        (self._grip_kp * L0 - self._grip_force) / self._grip_gain,
                        0.0, 255.0))
        elif grip_end >= 0.5:
            self._grip_latch = False

        # 子步间连续插值 eef 目标；每子步 IK 解关节目标交给位置伺服
        for k in range(_SUBSTEPS):
            a = (k + 1) / _SUBSTEPS
            tgt_pos = cur_pos + a * (ik_tgt_base - cur_pos)
            self._grip = grip_start + a * (grip_end - grip_start)
            if self._grip_latch:
                self._data.ctrl[self._act_grip] = self._grip_ctrl_hold
            else:
                self._data.ctrl[self._act_grip] = _grip_ctrl(self._grip)
            self._solve_ik(tgt_pos, cur_yaw + a * yaw_delta)
            mujoco.mj_step(self._model, self._data)
        self._grip = grip_end

        cube = self._cube_pos()
        in_bowl = float(np.hypot(cube[0] - _BOWL_XY[0], cube[1] - _BOWL_XY[1])) < _BOWL_R - 0.008
        resting = cube[2] < _CUBE_HALF + 0.006
        off_table = abs(cube[0]) > 0.42 or abs(cube[1]) > 0.32 or cube[2] < -0.1
        success = bool(in_bowl and resting)
        phase = self._phase(cube)

        terminated = success or off_table
        reason = "success" if success else ("out_of_bounds" if off_table else None)
        reward = -float(np.hypot(cube[0] - _BOWL_XY[0], cube[1] - _BOWL_XY[1]))
        return StepResult(
            observation=self._observe(),
            reward=reward,
            terminated=terminated,
            termination_reason=reason,
            truncated=False,
            info={"success": success, "phase": phase, "cube_pos": cube.tolist()},
        )

    def close(self) -> None:
        for r in self._renderers.values():
            r.close()
        self._renderers.clear()

    # ------------------------------------------------------------------ #
    def _site_pos(self) -> np.ndarray:
        return np.array(self._data.site_xpos[self._site], dtype=np.float64)

    def _site_quat(self) -> np.ndarray:
        q = np.empty(4)
        self._mj.mju_mat2Quat(q, self._data.site_xmat[self._site])
        return q

    def _site_yaw(self) -> float:
        m = self._data.site_xmat[self._site].reshape(3, 3)
        return float(np.arctan2(m[1, 0], m[0, 0]))

    def _grip_measured(self) -> float:
        """实测开度归一化到 0(闭)-1(开)，与动作第 5 维同语义。

        归一化窗口 = _grip_ctrl 的实际受控行程 [0.026, 0.076]m（不是关节全行
        程 0.08m）：空闭合到 0.026m 停止 → 0.0；全开 0.076m → 1.0。夹住 4cm
        方块时 gap≈0.043 → ≈0.34（"已闭合夹持"）。区别于命令值 `self._grip`
        ——手指被物体挡住时两者不同，观测必须报测量值。"""
        gap = float(self._data.qpos[self._finger_qadr].sum())
        return float(np.clip((gap - 0.026) / (0.076 - 0.026), 0.0, 1.0))

    def _eef_state(self) -> np.ndarray:
        """move_to 的本体参考态：与动作 (x,y,z,yaw,gripper) 同序同纲的实测值。"""
        return np.array([*self._site_pos(), self._site_yaw(), self._grip_measured()],
                        dtype=np.float64)

    def _ik_static(self, tgt_pos: np.ndarray, tgt_yaw: float, q_seed: np.ndarray,
                   iters: int = 100, damping: float = 0.02) -> np.ndarray:
        """纯运动学 IK（不动力学），返回解关节角——用于求 home 姿态。"""
        mujoco = self._mj
        jacp = np.zeros((3, self._model.nv))
        jacr = np.zeros((3, self._model.nv))
        cols = self._dadr
        q = np.asarray(q_seed, dtype=np.float64).copy()
        for _ in range(iters):
            self._data.qpos[self._qadr] = q
            mujoco.mj_kinematics(self._model, self._data)
            mujoco.mj_comPos(self._model, self._data)
            mujoco.mj_jacSite(self._model, self._data, jacp, jacr, self._site)
            J = np.vstack([jacp[:, cols], jacr[:, cols]])
            err = np.concatenate([
                tgt_pos - self._data.site_xpos[self._site],
                _orient_error(self._data.site_xmat[self._site].reshape(3, 3), tgt_yaw) * 0.5,
            ])
            if np.linalg.norm(err) < 1e-5:
                break
            dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), err)
            q = np.clip(q + dq, self._ctrl_lo, self._ctrl_hi)
        return q

    def _solve_ik(self, tgt_pos: np.ndarray, tgt_yaw: float, iters: int = 6,
                  damping: float = 0.05, null_gain: float = 0.02) -> None:
        """DLS IK（xyz+yaw 四维误差）+ 关节限位钳制 + 零空间回中。

        直接写 qpos 为 IK 解（运动学层），同时把解设为位置伺服目标——
        伺服动力学在 mj_step 中产生跟踪滞后与接触柔顺，即真实执行层。

        Task 3 修复（E2 证据，2026-09-14）：原实现每子步 3 次迭代 + 无条件
        零空间回中，转移途中构型被持续拖拽漂移，joint4 顶死下限位后落入
        不可达分支，静态 TCP 残差 ~15-24mm。现改为：
          1) 零空间回中只在任务残差 <5mm 的精修阶段启用（不在转移中拉扯）；
          2) 迭代后残差仍 >5mm 时做一次性重分支恢复：用纯运动学静态解
             （_ik_static，实测可达 0mm）作为伺服目标，跳出坏分支。
        """
        mujoco = self._mj
        jacp = np.zeros((3, self._model.nv))
        jacr = np.zeros((3, self._model.nv))
        cols = self._dadr
        eye7 = np.eye(7)

        # 关键：只在候选 q 上评估雅可比；结束时恢复真实 qpos、只把解写入
        # 位置伺服目标。直接写 qpos 会造成运动学传送（qvel≈0），接触摩擦
        # 无法带动物体（与 mocap 直驱同一陷阱，实测）。
        q_actual = self._data.qpos[self._qadr].copy()
        q = q_actual.copy()
        for _ in range(iters):
            self._data.qpos[self._qadr] = q
            mujoco.mj_kinematics(self._model, self._data)
            mujoco.mj_comPos(self._model, self._data)
            mujoco.mj_jacSite(self._model, self._data, jacp, jacr, self._site)
            J = np.vstack([jacp[:, cols], jacr[:, cols]])
            err = np.concatenate([
                tgt_pos - self._data.site_xpos[self._site],
                _orient_error(self._data.site_xmat[self._site].reshape(3, 3), tgt_yaw) * 0.5,
            ])
            if np.linalg.norm(err) < 1e-5:
                break
            Jt = J.T
            JJt = J @ Jt + damping**2 * np.eye(6)
            dq = Jt @ np.linalg.solve(JJt, err)
            if np.linalg.norm(err[:3]) < 0.005:  # 精修阶段才做零空间回中
                Jpinv = Jt @ np.linalg.inv(JJt)
                dq = dq + (eye7 - Jpinv @ J) @ (null_gain * (self._q_home_eff - q))
            q = np.clip(q + dq, self._ctrl_lo, self._ctrl_hi)
        # 重分支恢复：残差仍大时尝试静态解换分支；但静态解自己也可能发散
        # （E2：yaw≠0 的边界姿态实测 325mm），只在它自身残差 <5mm 时采纳。
        self._data.qpos[self._qadr] = q
        mujoco.mj_kinematics(self._model, self._data)
        mujoco.mj_comPos(self._model, self._data)
        if np.linalg.norm(tgt_pos - self._data.site_xpos[self._site]) > 0.005:
            q_alt = self._ik_static(tgt_pos, tgt_yaw, self._q_home_eff, iters=200)
            self._data.qpos[self._qadr] = q_alt
            mujoco.mj_kinematics(self._model, self._data)
            mujoco.mj_comPos(self._model, self._data)
            if np.linalg.norm(tgt_pos - self._data.site_xpos[self._site]) < 0.005:
                q = q_alt
        self._data.qpos[self._qadr] = q_actual  # 恢复真实状态
        self._data.ctrl[self._act_arm] = q      # 伺服目标=IK 解

    def _cube_pos(self) -> np.ndarray:
        return np.array(self._data.xpos[self._cube_bid], dtype=np.float64)

    def _phase(self, cube: np.ndarray) -> int:
        eef = self._site_pos()
        d_xy = float(np.hypot(*(cube[:2] - eef[:2])))
        d_bowl = float(np.hypot(cube[0] - _BOWL_XY[0], cube[1] - _BOWL_XY[1]))
        carried = d_xy < 0.06 and cube[2] > _CUBE_HALF + 0.01
        if d_bowl < _BOWL_R - 0.008 and cube[2] < _CUBE_HALF + 0.006:
            return 4
        if carried and d_bowl < 0.08:
            return 3
        if carried or cube[2] > _CUBE_HALF + 0.015:
            return 2
        if d_xy < 0.08:
            return 1
        return 0

    # ------------------------------------------------------------------ #

    def _render(self, cam: str) -> np.ndarray:
        # 全部走自由相机（本版 MuJoCo 的 XML/FIXED 相机在 macOS 离屏下只渲染
        # 天空，实测）。俯视被机械臂自身遮挡 → 全局视角用前侧斜视；腕视 =
        # 指间 TCP+0.02 向下看（穿过张开的指间，真腕载相机位形）。
        mujoco = self._mj
        vcam = mujoco.MjvCamera()
        if cam == "wrist":  # 指间俯视近景：世界对齐（+x 图像右、+y 图像上）
            tcp = self._site_pos()
            vcam.lookat[:] = tcp + np.array([0.0, 0.0, -0.14])
            vcam.distance = 0.16
            vcam.azimuth = 90.0
            vcam.elevation = -90.0
        elif cam == "top":  # 全局斜视（避开臂的自遮挡）
            vcam.lookat[:] = [0.0, 0.0, 0.02]
            vcam.distance = 1.25
            vcam.azimuth = -45.0
            vcam.elevation = -55.0
        else:               # side：另一侧斜视，与 top 互补
            vcam.lookat[:] = [0.05, 0.0, 0.03]
            vcam.distance = 1.0
            vcam.azimuth = -130.0
            vcam.elevation = -25.0
        if cam not in self._renderers:
            self._renderers[cam] = mujoco.Renderer(self._model, height=self._img_h, width=self._img_w)
        self._renderers[cam].update_scene(self._data, camera=vcam)
        return self._renderers[cam].render()

    def _observe(self) -> Observation:
        q = self._site_quat()
        return Observation(
            images={c: self._render(c) for c in ("top", "side", "wrist")},
            state={
                "eef_pos": self._site_pos(),
                "eef_quat": q,
                "eef_yaw": np.array([self._site_yaw()], dtype=np.float64),
                "gripper": np.array([self._grip_measured()], dtype=np.float64),
                "eef_state": self._eef_state(),
            },
            instruction=self._instruction,
        )


# ---------------------------------------------------------------------- #
def _go(emb, x: float, y: float, z: float, *, yaw: float = 0.0, grip: float | None = None,
        tol_pos: float = 0.003, tol_yaw: float = np.radians(2.0), tol_speed: float = 0.002,
        hold: int = 3, max_steps: int = 40) -> dict:
    """严格到位判定：位置/姿态/速度三条件连续 hold 个控制步（0.3s@10Hz）
    同时满足才算 reached；超时或 terminated 都返回失败，由调用方决定是否
    继续下一阶段（规划 §6：超时不得静默进入下一步）。

    tol 默认 = 规划拟议工程标准（3mm / 2° / 2cm每步≈20cm/s），E3，非官方阈值。
    注意：靠近基座的悬停姿态贴近 Panda 肘折叠边界（E2 实测静态残差 ~11mm），
    转移段调用应放宽 tol（见 _grasp），下探等抓取关键位姿保持严格门。
    """
    from inspect_robots.types import Action

    grip_cmd = float(emb._grip if grip is None else grip)
    tgt = np.array([x, y, z])
    prev = emb._site_pos().copy()
    consec, last = 0, {}
    res = None
    for i in range(max_steps):
        res = emb.step(Action(data=np.array([x, y, z, yaw, grip_cmd])))
        pos = emb._site_pos()
        pe = float(np.linalg.norm(pos - tgt))
        ye = float(abs(_wrap(emb._site_yaw() - yaw)))
        sp = float(np.linalg.norm(pos - prev))
        prev = pos.copy()
        last = {"pos_err": round(pe, 5), "yaw_err": round(ye, 5), "speed": round(sp, 5)}
        if res.terminated:
            return {"reached": False, "steps": i + 1, "terminated": True,
                    "reason": res.termination_reason, **last}
        consec = consec + 1 if (pe <= tol_pos and ye <= tol_yaw and sp <= tol_speed) else 0
        if consec >= hold:
            return {"reached": True, "steps": i + 1, "terminated": False,
                    "reason": None, **last}
    return {"reached": False, "steps": max_steps, "terminated": False,
            "reason": "timeout", **last}


def _grasp(emb, cube: np.ndarray, cube_yaw: float, *, pacing: str = "slow") -> dict:
    """S1 空载到位（悬停+对齐下探，不闭合）+ S2 桌面闭合。返回各阶段记录。"""
    from inspect_robots.types import Action

    if pacing == "fast":
        # 工具链节奏：等效 move_to chunk 的连续 playout——按 ≤4cm/步把到
        # 路标点的全程插值成动作序列、每动作恰 1 个控制步、无沉降等待。
        for pt in [(*cube[:2], cube[2] + 0.12), (*cube[:2], cube[2] + _GRASP_TCP_OFF)]:
            cur = emb._site_pos().copy()
            seg = np.array(pt) - cur
            n = max(1, int(np.ceil(np.abs(seg).max() / _MAX_POS_STEP)))
            for k in range(1, n + 1):
                a = cur + seg * (k / n)
                emb.step(Action(data=np.array([a[0], a[1], a[2], cube_yaw, 1.0])))
        cap = []
        for _ in range(6):
            emb.step(Action(data=np.array([cube[0], cube[1], cube[2] + _GRASP_TCP_OFF,
                                           cube_yaw, 0.0])))
            cap.append(_finger_contact_scan(emb))
        return {"approach": {"reached": None, "note": "fast playout, ungated"},
                "closure": _closure_metrics(cap, cube[:2], emb._cube_pos()[:2])}
    approach = _go(emb, cube[0], cube[1], cube[2] + 0.12, yaw=cube_yaw, grip=1.0,
                   tol_pos=0.040, tol_yaw=np.radians(6.0))
    if not approach["reached"]:
        return {"approach": approach, "closure": None}
    descend = _go(emb, cube[0], cube[1], cube[2] + _GRASP_TCP_OFF, yaw=cube_yaw, grip=1.0)
    if not descend["reached"]:
        return {"approach": {**approach, "descend": descend}, "closure": None}
    cap = []
    for _ in range(15):  # 闭合：TCP 目标保持不动
        emb.step(Action(data=np.array([cube[0], cube[1], cube[2] + _GRASP_TCP_OFF,
                                       cube_yaw, 0.0])))
        cap.append(_finger_contact_scan(emb))
    return {"approach": {**approach, "descend": descend},
            "closure": _closure_metrics(cap, cube[:2], emb._cube_pos()[:2])}


def _finger_contact_scan(emb) -> dict:
    """当前步的指-cube 接触快照（单遍扫描）：左右接触数与法向力总和。

    注意 fnL/fnR 是该侧全部接触点的法向力**求和**（接触系 force[0]），
    不是单点最大——指垫有多个碰撞面，取 max 会低估挤压力。"""
    mj, model, data = emb._mj, emb._model, emb._data
    cube_gid = model.geom("cube_geom").id
    bids = {model.body("left_finger").id: "L", model.body("right_finger").id: "R"}
    n = {"L": 0, "R": 0}
    fn = {"L": 0.0, "R": 0.0}
    f = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        if cube_gid not in (c.geom1, c.geom2):
            continue
        b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
        side = next((s for b, s in bids.items() if b in (b1, b2)), None)
        if side is None:
            continue
        mj.mj_contactForce(model, data, i, f)
        n[side] += 1
        fn[side] += float(f[0])
    return {"nL": n["L"], "nR": n["R"], "fnL": round(fn["L"], 2), "fnR": round(fn["R"], 2)}


def _closure_metrics(cap: list[dict], cube_xy0: np.ndarray, cube_xy_now: np.ndarray) -> dict:
    """S2 桌面闭合指标（cap = 逐步 _finger_contact_scan 快照）：
    左右接触起始步、双侧持续率、最大法向力、闭合后方块偏移、闭合后开度。"""
    l_on = next((i for i, c in enumerate(cap) if c["nL"] > 0), None)
    r_on = next((i for i, c in enumerate(cap) if c["nR"] > 0), None)
    both = sum(1 for c in cap if c["nL"] > 0 and c["nR"] > 0)
    return {"steps": len(cap), "left_onset": l_on, "right_onset": r_on,
            "both_frac": round(both / len(cap), 3),
            "fnL_max": max(c["fnL"] for c in cap),
            "fnR_max": max(c["fnR"] for c in cap),
            "cube_offset": round(float(np.linalg.norm(cube_xy_now - cube_xy0)), 5)}


def scripted_solve(emb: MuJoCoBowlEmbodiment, seed: int, *,
                   pacing: str = "slow") -> dict:
    """特权脚本化抓放，分阶段流水线（Task 3 改写：每阶段门控，失败即停）。

    阶段：S1 空载到位 → S2 桌面闭合 → S3 垂直提升(1cm×10 增量+悬停2s) →
    S4 水平搬运 → S5 转腕(悬停±) → S6 自由释放。抓取 yaw 全程保持到 S5。
    pacing="fast" 复现工具链节奏（路标点间无沉降）作对照。
    返回 dict 兼容旧字段（held/success/phase/...），新增 stages 明细。
    """
    from inspect_robots.scene import Scene
    from inspect_robots.types import Action

    emb.reset(Scene(id=f"reg-{seed}", instruction="pick the red cube into the bowl",
                    init_seed=seed), seed=seed)
    cube = np.array(emb._cube_pos())
    qw, qx, qy, qz = emb._data.qpos[emb._cube_qadr + 3:emb._cube_qadr + 7]
    cube_yaw = float(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))

    stages: list[dict] = []

    g = _grasp(emb, cube, cube_yaw, pacing=pacing)
    ap, cl = g["approach"], g["closure"]
    stages.append({"stage": "S1_approach", **ap})
    if cl is None:
        reason = "descend not reached" if ap.get("descend") else "hover not reached"
        stages.append({"stage": "S2_closure", "failed": True, "reason": reason})
        return _solve_result(seed, stages, "grasp_close", emb, cube)
    # S2 门 = 接触建立且保持（双侧都接触过、结束时仍有双侧接触、方块未被
    # 挤飞）。闭合位移（自定心滑移）与力对称作为指标记录，抓持质量由
    # S3 的滑移/失持门判定——旧管线成功闭合的位移实测 3.5-16.6mm，
    # 规划拟议的 ≤3mm 是设计目标而非可行硬门（E2 标定，2026-09-14）。
    scan = _finger_contact_scan(emb)
    cl["fnL_end"], cl["fnR_end"] = scan["fnL"], scan["fnR"]
    ok = (cl["both_frac"] >= 0.5 and cl["cube_offset"] <= 0.025
          and min(scan["fnL"], scan["fnR"]) > 0.3)
    cl["ok"] = bool(ok)
    stages.append({"stage": "S2_closure", **cl})
    if not ok:
        return _solve_result(seed, stages, "grasp_close", emb, cube)

    # S3 垂直提升：1cm 目标增量 ×10，全程 cube_yaw，不横移不转腕
    grip0 = emb._grip_measured()
    hand0 = _cube_in_hand(emb)
    z = cube[2] + _GRASP_TCP_OFF
    for k in range(10):
        z += 0.01
        if pacing == "fast":
            emb.step(Action(data=np.array([cube[0], cube[1], z, cube_yaw, 0.0])))
        else:
            r = _go(emb, cube[0], cube[1], z, yaw=cube_yaw, grip=0.0,
                    tol_pos=0.006, max_steps=12)
            if r["terminated"]:
                break
    for _ in range(20):  # 悬停 2s
        res = emb.step(Action(data=np.array([cube[0], cube[1], z, cube_yaw, 0.0])))
        if res.terminated:
            break
    slip = float(np.linalg.norm(_cube_in_hand(emb) - hand0))
    lost = emb._grip_measured() < 0.1  # 方块脱手后指闭到行程下限
    stages.append({"stage": "S3_lift", "slip_m": round(slip, 5), "lost": bool(lost),
                   "grip_meas": round(emb._grip_measured(), 3),
                   "cube_z": round(float(emb._cube_pos()[2]), 4),
                   "ok": bool(slip <= 0.005 and not lost)})
    if slip > 0.005 or lost:
        return _solve_result(seed, stages, "lift_slip", emb, cube)
    held = True

    # S4 水平搬运：固定高度与 yaw
    carry = (_go(emb, _BOWL_XY[0], _BOWL_XY[1], z, yaw=cube_yaw, grip=0.0, max_steps=40)
             if pacing == "slow" else {"reached": None})
    if pacing == "fast":
        for _ in range(10):
            emb.step(Action(data=np.array([_BOWL_XY[0], _BOWL_XY[1], z, cube_yaw, 0.0])))
    carry_both = 0.0
    stages.append({"stage": "S4_carry", **{k: v for k, v in carry.items()},
                   "ok": carry.get("reached", True) in (True, None)})
    if carry.get("terminated") or carry.get("reached") is False:
        return _solve_result(seed, stages, "carry", emb, cube)

    # S5 转腕：悬停高度独立完成（回到 yaw=0 释放姿态）
    if pacing == "slow":
        turn = _go(emb, _BOWL_XY[0], _BOWL_XY[1], z, yaw=0.0, grip=0.0, max_steps=20)
    else:
        turn = {"reached": None}
        for _ in range(4):
            emb.step(Action(data=np.array([_BOWL_XY[0], _BOWL_XY[1], z, 0.0, 0.0])))
    stages.append({"stage": "S5_wrist", **turn})
    if turn.get("terminated") or turn.get("reached") is False:
        return _solve_result(seed, stages, "wrist", emb, cube)

    # S6 自由释放：碗上方开爪，静置判定
    for _ in range(3):
        emb.step(Action(data=np.array([_BOWL_XY[0], _BOWL_XY[1], 0.16, 0.0, 1.0])))
    res = None
    for _ in range(32):  # 释放 + 2s 静置窗口
        res = emb.step(Action(data=np.array([_BOWL_XY[0], _BOWL_XY[1], 0.18, 0.0, 1.0])))
        if res.terminated:
            break
    stages.append({"stage": "S6_release", "success": bool(res.info["success"]),
                   "cube": np.round(emb._cube_pos(), 4).tolist()})
    return _solve_result(seed, stages, None if res.info["success"] else "release",
                         emb, cube)


def _cube_in_hand(emb) -> np.ndarray:
    """方块在夹爪坐标系的位置（滑移度量基准）。"""
    model, data = emb._model, emb._data
    hand_bid = model.body("hand").id
    return data.xmat[hand_bid].reshape(3, 3).T @ (
        emb._cube_pos() - data.xpos[hand_bid])


def _solve_result(seed: int, stages: list, fail_stage: str | None, emb, cube) -> dict:
    _phase_of = {"grasp_close": 1, "lift_slip": 2, "carry": 3, "wrist": 3, "release": 4, None: 4}
    return {"seed": seed, "stages": stages,
            "fail_stage": fail_stage,
            "held": fail_stage is None or fail_stage in ("carry", "wrist", "release"),
            "success": fail_stage is None,
            "phase": _phase_of[fail_stage],
            "reason": fail_stage,
            "cube": np.round(emb._cube_pos(), 3).tolist(),
            "eef": np.round(emb._site_pos(), 3).tolist()}


if __name__ == "__main__":
    import sys

    noise = (0.0, 0.0, 0.0) if "--deterministic" in sys.argv else (0.0015, 0.001, 0.017)
    n_seeds = int(os.environ.get("REG_SEEDS", "10"))
    results = [scripted_solve(MuJoCoBowlEmbodiment(execution_noise=noise), 1000 + i)
               for i in range(n_seeds)]
    ok = sum(r["success"] for r in results)
    held = sum(r["held"] for r in results)
    print(f"execution_noise={noise}")
    print(f"grasp held: {held}/{n_seeds}   bowl success: {ok}/{n_seeds}")
    for r in results:
        print(f"  seed={r['seed']} held={r['held']} success={r['success']} "
              f"phase={r['phase']} reason={r['reason']} cube={r['cube']} eef={r['eef']}")
