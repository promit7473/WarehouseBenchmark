"""
Advanced Warehouse MDP Configuration

This module provides sophisticated MDP components specifically designed for warehouse navigation,
incorporating knowledge from NVIDIA Isaac Sim warehouse logistics tutorials and extensions.
"""

from isaaclab.managers import CommandTermCfg as CmdTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as TermTerm
from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp_utils

from . import rewards as warehouse_rewards
from . import observations as warehouse_obs
from . import events as warehouse_events


@configclass
class AdvancedWarehouseObservationsCfg:
    """Advanced observation configuration for warehouse navigation."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations with warehouse awareness."""

        # Standard robot state
        base_lin_vel = ObsTerm(func=mdp_utils.base_lin_vel, scale=2.0)
        base_ang_vel = ObsTerm(func=mdp_utils.base_ang_vel, scale=0.25)
        projected_gravity = ObsTerm(func=mdp_utils.projected_gravity)
        last_action = ObsTerm(func=mdp_utils.last_action)

        # Waypoint navigation
        waypoint_command = ObsTerm(
            func=mdp_utils.generated_commands,
            params={"command_name": "waypoint_nav"}
        )

        # Warehouse-specific advanced observations
        obstacle_proximity = ObsTerm(
            func=warehouse_obs.obstacle_proximity_observation,
            params={"sensor_cfg": SceneEntityCfg("contact_forces"), "max_distance": 5.0}
        )

        aisle_alignment = ObsTerm(
            func=warehouse_obs.aisle_alignment_observation,
            params={"command_name": "waypoint_nav"}
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class AdvancedWarehouseRewardsCfg:
    """Advanced reward configuration optimized for warehouse navigation."""

    # Survival and safety
    alive = RewTerm(func=mdp_utils.is_alive, weight=0.5)

    # Collision penalties (multi-layered)
    collision = RewTerm(
        func=mdp_utils.undesired_contacts,
        weight=-3.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*"), "threshold": 1.0}
    )

    collision_prediction = RewTerm(
        func=warehouse_rewards.collision_prediction_penalty,
        weight=-1.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces"), "prediction_distance": 2.0}
    )

    # Action penalties
    action_rate = RewTerm(func=mdp_utils.action_rate_l2, weight=-0.05)

    # Navigation rewards
    track_lin_vel_xy = RewTerm(
        func=mdp_utils.track_lin_vel_xy_exp,
        weight=1.5,
        params={"command_name": "waypoint_nav", "std": 0.5}
    )

    track_ang_vel_z = RewTerm(
        func=mdp_utils.track_ang_vel_z_exp,
        weight=0.75,
        params={"command_name": "waypoint_nav", "std": 0.5}
    )

    # Warehouse-specific advanced rewards
    path_efficiency = RewTerm(
        func=warehouse_rewards.path_efficiency_reward,
        weight=0.8,
        params={"command_name": "waypoint_nav"}
    )

    aisle_navigation = RewTerm(
        func=warehouse_rewards.aisle_navigation_reward,
        weight=0.3,
        params={"command_name": "waypoint_nav"}
    )

    shelf_interaction = RewTerm(
        func=warehouse_rewards.shelf_interaction_bonus,
        weight=0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces"), "interaction_distance": 1.5}
    )


@configclass
class AdvancedWarehouseCommandsCfg:
    """Advanced command configuration for warehouse-aware waypoint generation."""

    waypoint_nav = CmdTerm(
        func=mdp_utils.uniform_pose_2d_command,
        params={
            "asset_name": "robot",
            "simple_heading": True,
            "heading_command": True,
            "resampling_time_range": (1500.0, 1500.0),  # Longer episodes for warehouse navigation
            "debug_vis": True,
            "ranges": mdp_utils.UniformPose2dCommandCfg.Ranges(
                pos_x=(-26.0, 6.0),  # full_warehouse.usd actual: X=[-28, 8] with 2m margin
                pos_y=(-39.4, 31.42),  # full_warehouse.usd actual: Y=[-41.4, 33.42] with 2m margin
                heading=(-3.14, 3.14),
            ),
        },
    )


@configclass
class AdvancedWarehouseEventsCfg:
    """Advanced event configuration for warehouse randomization."""

    # Reset robot position - spawn at origin for visibility
    reset_robot = EventTerm(
        func=mdp_utils.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # Initialize warehouse-aware waypoints
    init_waypoints = EventTerm(
        func=warehouse_events.warehouse_aware_waypoint_generation,
        mode="reset",
        params={"command_name": "waypoint_nav"}
    )

    # Progress through waypoints
    progress_waypoints = EventTerm(
        func=mdp_utils.reset_root_state_uniform,  # Placeholder - would use waypoint progression
        mode="interval",
        interval_range_s=(0.04, 0.04),
    )

    # Randomize warehouse obstacles
    randomize_obstacles = EventTerm(
        func=warehouse_events.randomize_warehouse_obstacles,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("crate_1"),
            "x_range": (-10.0, 10.0),
            "y_range": (-10.0, 10.0)
        }
    )


@configclass
class AdvancedWarehouseTerminationsCfg:
    """Advanced termination conditions for warehouse safety."""

    time_out = TermTerm(func=mdp_utils.time_out, time_out=True)

    robot_fallen = TermTerm(
        func=lambda env: env.scene["robot"].data.root_state_w[:, 2] < -0.5,
        time_out=False
    )

    excessive_velocity = TermTerm(
        func=lambda env: torch.norm(env.scene["robot"].data.root_state_w[:, 7:10], dim=1) > 20.0,
        time_out=False
    )

    # Warehouse-specific terminations
    warehouse_bounds = TermTerm(
        func=lambda env: torch.norm(env.scene["robot"].data.root_state_w[:, :2], dim=1) > 20.0,
        time_out=False
    )

    # Severe collision termination
    severe_collision = TermTerm(
        func=mdp_utils.undesired_contacts,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*"), "threshold": 5.0},
        time_out=False
    )