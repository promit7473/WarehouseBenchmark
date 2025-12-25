"""
Warehouse Asset Configurations

This module provides configurations for all warehouse assets including robots, terrains, and objects.
"""

import os
from isaaclab.assets import ArticulationCfg
from isaaclab import sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg


# Pre-configured robot instances
JETBOT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(os.path.dirname(__file__), "jetbot.usd"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.01,
            angular_damping=0.01,
            max_linear_velocity=10.0,
            max_angular_velocity=10.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
            fix_root_link=False,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.15),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
    actuators={
        "diff_drive": ImplicitActuatorCfg(
            joint_names_expr=["left_wheel_joint", "right_wheel_joint"],
            effort_limit=400.0,
            velocity_limit=20.0,
            stiffness=0.0,
            damping=100.0,
        ),
    },
)