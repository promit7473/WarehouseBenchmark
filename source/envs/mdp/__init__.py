# source/envs/mdp/__init__.py
"""MDP components for warehouse navigation environment.

This module exports observation, reward, termination, and event functions
following Isaac Lab patterns. Explicit imports are used instead of star imports
for better code clarity and IDE support.
"""

# Import specific Isaac Lab MDP terms that we use
from isaaclab.envs.mdp import (
    # Observations
    base_lin_vel,
    base_ang_vel,
    projected_gravity,
    last_action,
    generated_commands,
    # Rewards
    action_rate_l2,
    is_alive,
    # Terminations
    time_out,
)

# Import local observation functions
from .observations import (
    ObservationsCfg,
    obstacle_proximity_observation,
    aisle_alignment_observation,
    noisy_waypoint_command,
    lidar_observations,
    height_scan_warehouse,
    camera_rgb_observations,
    camera_depth_observations,
)

# Import local reward functions
from .rewards import (
    RewardsCfg,
    waypoint_reach_reward,
    waypoint_reached_reward,
    waypoint_progress_reward,
    collision_prediction_penalty,
    path_efficiency_reward,
    aisle_navigation_reward,
)

# Import local termination functions
from .terminations import (
    is_success,
    far_from_target,
    collision_with_obstacles,
    robot_fallen,
    robot_flipped,
    out_of_bounds,
)

# Import local event functions
from .events import (
    randomize_obstacles,
    warehouse_aware_waypoint_generation,
    randomize_warehouse_obstacles,
    initialize_waypoints,
    progress_waypoints,
    update_current_goal_indicator,
)

# Import curriculum functions
from .curriculum import (
    waypoint_distance_curriculum,
)

__all__ = [
    # Isaac Lab MDP terms
    "base_lin_vel",
    "base_ang_vel",
    "projected_gravity",
    "last_action",
    "generated_commands",
    "action_rate_l2",
    "is_alive",
    "time_out",
    # Local observations
    "ObservationsCfg",
    "obstacle_proximity_observation",
    "aisle_alignment_observation",
    "noisy_waypoint_command",
    "lidar_observations",
    "height_scan_warehouse",
    "camera_rgb_observations",
    "camera_depth_observations",
    # Local rewards
    "RewardsCfg",
    "waypoint_reach_reward",
    "waypoint_reached_reward",
    "waypoint_progress_reward",
    "collision_prediction_penalty",
    "path_efficiency_reward",
    "aisle_navigation_reward",
    # Local terminations
    "is_success",
    "far_from_target",
    "collision_with_obstacles",
    "robot_fallen",
    "robot_flipped",
    "out_of_bounds",
    # Local events
    "randomize_obstacles",
    "warehouse_aware_waypoint_generation",
    "randomize_warehouse_obstacles",
    "initialize_waypoints",
    "progress_waypoints",
    "update_current_goal_indicator",
    # Curriculum
    "waypoint_distance_curriculum",
]