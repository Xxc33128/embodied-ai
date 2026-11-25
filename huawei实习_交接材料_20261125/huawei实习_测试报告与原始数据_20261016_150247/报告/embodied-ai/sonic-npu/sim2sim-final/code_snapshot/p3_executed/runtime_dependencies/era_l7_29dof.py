from era_okcc_humanoid_lab.assets import ASSET_DIR
from era_okcc_humanoid_lab.robots.actuator import DelayedImplicitActuatorCfg

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

ARMATURE_6508_100 = 0.01
ARMATURE_5005 = 0.01
ARMATURE_10520 = 0.16473
ARMATURE_9015 = 0.088
ARMATURE_15017 = 0.0968
ARMATURE_6008_30 = 0.0225

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_6508_100 = ARMATURE_6508_100 * NATURAL_FREQ**2
STIFFNESS_5005 = ARMATURE_5005 * NATURAL_FREQ**2
STIFFNESS_10520 = ARMATURE_10520 * NATURAL_FREQ**2
STIFFNESS_9015 = ARMATURE_9015 * NATURAL_FREQ**2
STIFFNESS_15017 = ARMATURE_15017 * NATURAL_FREQ**2
STIFFNESS_6008_30 = ARMATURE_6008_30 * NATURAL_FREQ**2

DAMPING_6508_100 = 2.0 * DAMPING_RATIO * ARMATURE_6508_100 * NATURAL_FREQ
DAMPING_5005 = 2.0 * DAMPING_RATIO * ARMATURE_5005 * NATURAL_FREQ
DAMPING_10520 = 2.0 * DAMPING_RATIO * ARMATURE_10520 * NATURAL_FREQ
DAMPING_9015 = 2.0 * DAMPING_RATIO * ARMATURE_9015 * NATURAL_FREQ
DAMPING_15017 = 2.0 * DAMPING_RATIO * ARMATURE_15017 * NATURAL_FREQ
DAMPING_6008_30 = 2.0 * DAMPING_RATIO * ARMATURE_6008_30 * NATURAL_FREQ


L7_29DOF_CYLINDER_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        # replace_cylinders_with_capsules=True,
        replace_cylinders_with_capsules=False,
        asset_path=f"{ASSET_DIR}/xbot_description/urdf/l7_29dof_neck_fixed.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        joint_pos={
            ".*_hip_pitch_joint": 0.0,
            ".*_knee_joint": 0.0,
            ".*_ankle_pitch_joint": 0.0,
            ".*_elbow_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_pitch_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DelayedImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_roll_joint",
                ".*_hip_yaw_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_roll_joint": 255.0,
                ".*_hip_yaw_joint": 100.0,
                ".*_hip_pitch_joint": 350.0,
                ".*_knee_joint": 350.0,
            },
            velocity_limit_sim={
                ".*_hip_roll_joint": 7.28,
                ".*_hip_yaw_joint": 13.26,
                ".*_hip_pitch_joint": 20.0,
                ".*_knee_joint": 20.0,
            },
            stiffness={
                ".*_hip_roll_joint": STIFFNESS_10520,
                ".*_hip_yaw_joint": STIFFNESS_9015,
                ".*_hip_pitch_joint": STIFFNESS_15017,
                ".*_knee_joint": STIFFNESS_15017,
            },
            damping={
                ".*_hip_roll_joint": DAMPING_10520,
                ".*_hip_yaw_joint": DAMPING_9015,
                ".*_hip_pitch_joint": DAMPING_15017,
                ".*_knee_joint": DAMPING_15017,
            },
            armature={
                ".*_hip_roll_joint": ARMATURE_10520,
                ".*_hip_yaw_joint": ARMATURE_9015,
                ".*_hip_pitch_joint": ARMATURE_15017,
                ".*_knee_joint": ARMATURE_15017,
            },
            min_delay=0,
            max_delay=4,
        ),
        "feet": DelayedImplicitActuatorCfg(
            effort_limit_sim=50.0,
            velocity_limit_sim=13.65,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=2.0 * STIFFNESS_6008_30,
            damping=2.0 * DAMPING_6008_30,
            armature=2.0 * ARMATURE_6008_30,
            min_delay=0,
            max_delay=4,
        ),
        "waist": DelayedImplicitActuatorCfg(
            effort_limit_sim=110.0,
            velocity_limit_sim=4.19,
            joint_names_expr=["waist_roll_joint", "waist_pitch_joint"],
            stiffness=500.0,  # 2.0 * STIFFNESS_6508_100,
            damping=6.0,
            armature=2.0 * ARMATURE_6508_100,
            min_delay=0,
            max_delay=4,
        ),
        "waist_yaw": DelayedImplicitActuatorCfg(
            effort_limit_sim=110.0,
            velocity_limit_sim=4.19,
            joint_names_expr=["waist_yaw_joint"],
            stiffness=500.0,  # STIFFNESS_6508_100,
            damping=6.0,  # DAMPING_6508_100,
            armature=ARMATURE_6508_100,
            min_delay=0,
            max_delay=4,
        ),
        "arms": DelayedImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_arm_yaw_joint",
                ".*_elbow_pitch_joint",
                ".*_elbow_yaw_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_roll_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 95.0,
                ".*_shoulder_roll_joint": 95.0,
                ".*_arm_yaw_joint": 95.0,
                ".*_elbow_pitch_joint": 95.0,
                ".*_elbow_yaw_joint": 35.0,
                ".*_wrist_pitch_joint": 35.0,
                ".*_wrist_roll_joint": 35.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 4.19,
                ".*_shoulder_roll_joint": 4.19,
                ".*_arm_yaw_joint": 4.19,
                ".*_elbow_pitch_joint": 4.19,
                ".*_elbow_yaw_joint": 4.19,
                ".*_wrist_pitch_joint": 4.19,
                ".*_wrist_roll_joint": 4.19,
            },
            stiffness={
                ".*_shoulder_pitch_joint": STIFFNESS_6508_100,
                ".*_shoulder_roll_joint": STIFFNESS_6508_100,
                ".*_arm_yaw_joint": STIFFNESS_6508_100,
                ".*_elbow_pitch_joint": STIFFNESS_6508_100,
                ".*_elbow_yaw_joint": STIFFNESS_5005,
                ".*_wrist_pitch_joint": STIFFNESS_5005,
                ".*_wrist_roll_joint": STIFFNESS_5005,
            },
            damping={
                ".*_shoulder_pitch_joint": DAMPING_6508_100,
                ".*_shoulder_roll_joint": DAMPING_6508_100,
                ".*_arm_yaw_joint": DAMPING_6508_100,
                ".*_elbow_pitch_joint": DAMPING_6508_100,
                ".*_elbow_yaw_joint": DAMPING_5005,
                ".*_wrist_pitch_joint": DAMPING_5005,
                ".*_wrist_roll_joint": DAMPING_5005,
            },
            armature={
                ".*_shoulder_pitch_joint": ARMATURE_6508_100,
                ".*_shoulder_roll_joint": ARMATURE_6508_100,
                ".*_arm_yaw_joint": ARMATURE_6508_100,
                ".*_elbow_pitch_joint": ARMATURE_6508_100,
                ".*_elbow_yaw_joint": ARMATURE_5005,
                ".*_wrist_pitch_joint": ARMATURE_5005,
                ".*_wrist_roll_joint": ARMATURE_5005,
            },
            min_delay=0,
            max_delay=4,
        ),
    },
)

L7_29DOF_NECK_FIXED_ACTION_SCALE = {}
for a in L7_29DOF_CYLINDER_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            L7_29DOF_NECK_FIXED_ACTION_SCALE[n] = 0.25 * e[n] / s[n]
