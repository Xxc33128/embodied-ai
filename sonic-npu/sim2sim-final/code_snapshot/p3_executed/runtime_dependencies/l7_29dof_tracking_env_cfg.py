# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from era_okcc_humanoid_lab.robots.era_l7_29dof import (
    L7_29DOF_CYLINDER_CFG,
    L7_29DOF_NECK_FIXED_ACTION_SCALE,
)

from ...tracking_env_cfg import TrackingEnvCfg


@configclass
class L7_29DofTrackingEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.use_identify_params = True  # flag to use identify params or not
        self.use_high_waist_stiffness = True  # flag to use high waist stiffness or not

        self.scene.robot = L7_29DOF_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = L7_29DOF_NECK_FIXED_ACTION_SCALE

        self.commands.motion.anchor_body_name = "pelvis"

        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_pitch_link",
            "left_wrist_roll_link",
            "right_shoulder_roll_link",
            "right_elbow_pitch_link",
            "right_wrist_roll_link",
        ]

        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None
