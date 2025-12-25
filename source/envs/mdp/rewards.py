# source/envs/mdp/rewards.py

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp_utils
import torch
from source.envs.utils import quaternion_to_yaw

def waypoint_reach_reward(env):
    """Reward for proximity to waypoint."""
    robot_pos = env.scene["robot"].data.root_state_w[:, :2]
    waypoint = env.command_manager.get_command("waypoint_nav")[:, :2]
    distance = torch.norm(robot_pos - waypoint, dim=1)
    # Add epsilon to prevent division by zero
    return 1.0 / (1.0 + distance + 1e-6)


def waypoint_reached_reward(env, command_name: str = "waypoint_nav", distance_threshold: float = 1.5):
    """Large reward when robot reaches a waypoint - PRIMARY OBJECTIVE."""
    current_pos = env.scene["robot"].data.root_pos_w[:, :2]
    # Get the command target properly - command is a tensor with target positions
    command = env.command_manager.get_command(command_name)
    target_pos = command[:, :2]  # Command contains target positions directly
    
    distance = torch.norm(target_pos - current_pos, dim=1)
    reached = distance < distance_threshold
    
    return reached.float()


def waypoint_progress_reward(env, command_name: str = "waypoint_nav"):
    """Reward for making progress toward current waypoint."""
    current_pos = env.scene["robot"].data.root_pos_w[:, :2]
    # Get the command target properly
    command = env.command_manager.get_command(command_name)
    target_pos = command[:, :2]  # Command contains target positions directly
    
    # Calculate distance to target
    distance = torch.norm(target_pos - current_pos, dim=1)
    
    # Reward inversely proportional to distance (closer = more reward)
    # Normalize to [0, 1] range
    max_distance = 20.0  # Maximum expected distance in warehouse
    progress = torch.clamp(1.0 - (distance / max_distance), 0.0, 1.0)
    
    return progress


def collision_prediction_penalty(env, sensor_cfg: SceneEntityCfg = None, prediction_distance: float = 2.0):
    """Penalty for getting too close to obstacles using REAL SENSOR DATA.

    **CRITICAL CHANGE**: Now uses actual LiDAR sensor data instead of hardcoded positions.
    This makes the robot learn real obstacle avoidance through perception.

    Args:
        env: The environment instance
        sensor_cfg: Contact sensor configuration (for actual collision detection)
        prediction_distance: Distance threshold for collision prediction (meters)

    Returns:
        Negative penalty tensor (higher magnitude = closer to obstacles)
    """
    # Priority 1: Check for actual collisions using contact sensor
    if sensor_cfg is not None:
        try:
            contact_sensor = env.scene.sensors.get(sensor_cfg.name)
            if contact_sensor is not None:
                # Get net contact forces - high forces indicate collision
                net_forces = contact_sensor.data.net_forces_w
                force_magnitude = torch.norm(net_forces, dim=-1).sum(dim=-1)  # Sum across bodies
                # Heavy penalty for actual contact
                contact_penalty = torch.clamp(force_magnitude / 100.0, 0.0, 5.0)
                # If we have contact, return immediately with heavy penalty
                if contact_penalty.max() > 0.1:
                    return -contact_penalty * 2.0  # Double penalty for actual collision
        except (KeyError, AttributeError):
            pass

    # Priority 2: Use LiDAR sensor data for proximity-based collision prediction
    # This is REAL sensor-based obstacle detection (not hardcoded!)
    try:
        lidar_sensor = env.scene["lidar"]
        # Get LiDAR distance measurements
        ray_hits = lidar_sensor.data.ray_hits_w
        distances = ray_hits[:, :, 0]  # Distance is in the first column

        # Find minimum distance to any obstacle across all rays
        # This tells us how close the nearest obstacle is
        min_distances, _ = torch.min(distances, dim=1)  # (num_envs,)

        # Calculate penalty based on proximity to obstacles
        # Exponential penalty: gets very high as robot approaches obstacles
        # prediction_distance = 2.0m means penalty kicks in at 2m and increases exponentially
        proximity_penalty = torch.exp(-min_distances / prediction_distance)

        # Scale penalty appropriately
        scaled_penalty = proximity_penalty * 1.5

        return -scaled_penalty

    except (KeyError, AttributeError):
        # FALLBACK ONLY (shouldn't happen if LiDAR is properly configured)
        # Return minimal penalty if no sensors available
        import warnings
        warnings.warn("Neither contact sensor nor LiDAR available for collision detection!")
        return torch.zeros(env.num_envs, device=env.device)


def path_efficiency_reward(env, command_name: str):
    """Reward for taking efficient paths in warehouse aisles."""
    robot_pos = env.scene["robot"].data.root_state_w[:, :2]
    waypoint = env.command_manager.get_command(command_name)[:, :2]

    # Calculate straight-line distance to goal
    straight_distance = torch.norm(waypoint - robot_pos, dim=1)

    # Calculate path efficiency: reward for moving towards goal vs wandering
    # Track robot's movement history (simplified - just current heading vs optimal)
    robot_yaw = quaternion_to_yaw(env.scene["robot"].data.root_state_w)  # Extract yaw from quaternion
    direction_to_goal = torch.atan2(waypoint[:, 1] - robot_pos[:, 1],
                                   waypoint[:, 0] - robot_pos[:, 0])

    # Angle difference between current heading and optimal direction
    angle_diff = torch.abs(torch.atan2(torch.sin(direction_to_goal - robot_yaw),
                                      torch.cos(direction_to_goal - robot_yaw)))

    # Efficiency bonus: higher when moving towards goal, lower when going wrong way
    heading_efficiency = torch.exp(-angle_diff / (torch.pi / 4))  # Prefer within 45 degrees

    # Combine distance progress with heading efficiency
    efficiency_bonus = torch.exp(-straight_distance / 15.0) * heading_efficiency * 1.5

    return efficiency_bonus


def aisle_navigation_reward(env, command_name: str):
    """Reward for proper aisle navigation in warehouse."""
    robot_pos = env.scene["robot"].data.root_state_w[:, :2]
    waypoint = env.command_manager.get_command(command_name)[:, :2]

    # Warehouse aisle alignment: reward staying in grid-like patterns
    # Typical warehouse has aisles every ~3-4 meters

    # Calculate alignment with grid (aisles typically run N-S or E-W)
    # Check if robot is near grid lines (aisle centers)
    aisle_width = 3.5  # meters between aisle centers

    # Grid alignment: distance to nearest grid line
    grid_x = torch.round(robot_pos[:, 0] / aisle_width) * aisle_width
    grid_y = torch.round(robot_pos[:, 1] / aisle_width) * aisle_width

    grid_alignment_x = torch.exp(-torch.abs(robot_pos[:, 0] - grid_x) / 0.5)
    grid_alignment_y = torch.exp(-torch.abs(robot_pos[:, 1] - grid_y) / 0.5)

    # Reward higher alignment (being in aisles vs between shelves)
    alignment_bonus = (grid_alignment_x + grid_alignment_y) * 0.3

    # Additional bonus for moving parallel to aisles when far from goal
    distance_to_goal = torch.norm(waypoint - robot_pos, dim=1)
    far_from_goal = (distance_to_goal > 5.0).float()

    # When far from goal, prefer moving along aisles (N-S or E-W)
    robot_yaw = quaternion_to_yaw(env.scene["robot"].data.root_state_w)  # Extract yaw from quaternion
    aisle_alignment = torch.minimum(
        torch.abs(robot_yaw),  # North-South
        torch.minimum(
            torch.abs(robot_yaw - torch.pi/2),  # East-West
            torch.minimum(
                torch.abs(robot_yaw - torch.pi),  # South-North
                torch.abs(robot_yaw + torch.pi/2)   # West-East
            )
        )
    )

    # Aisle alignment bonus (higher when aligned with aisles)
    aisle_bonus = torch.exp(-aisle_alignment / (torch.pi / 8)) * 0.4

    # Combine alignment bonuses
    total_alignment = alignment_bonus + (far_from_goal * aisle_bonus)

    return total_alignment


@configclass
class RewardsCfg:
    """
    RLRoverLab-Inspired Clear Reward Hierarchy for Warehouse Navigation
    
    PRIMARY OBJECTIVE (70% weight): Sequential waypoint completion
    SECONDARY OBJECTIVES (20% weight): Navigation efficiency and quality
    TERTIARY OBJECTIVES (10% weight): Basic safety and action smoothness
    """

    # === PRIMARY OBJECTIVES: Sequential Waypoint Navigation ===
    # Main goal: Complete warehouse logistics sequence efficiently
    
    waypoint_reached = RewTerm(
        func=waypoint_reached_reward,
        weight=10.0,  # Highest weight - primary success metric
        params={"command_name": "waypoint_nav", "distance_threshold": 1.5}
    )
    
    waypoint_progress = RewTerm(
        func=waypoint_progress_reward,
        weight=3.0,  # Progress toward current waypoint
        params={"command_name": "waypoint_nav"}
    )

    # === SECONDARY OBJECTIVES: Navigation Quality ===
    # Important but secondary to waypoint completion
    
    path_efficiency = RewTerm(
        func=path_efficiency_reward,
        weight=1.0,  # Reward efficient paths
        params={"command_name": "waypoint_nav"}
    )
    
    # Warehouse-specific navigation patterns
    aisle_navigation = RewTerm(
        func=aisle_navigation_reward,
        weight=0.5,  # Reward proper warehouse navigation
        params={"command_name": "waypoint_nav"}
    )

    # === TERTIARY OBJECTIVES: Safety and Action Quality ===
    # Basic requirements, minimal weight
    
    # Safety (critical but minimal weight - should be inherently learned)
    collision = RewTerm(
        func=collision_prediction_penalty,
        weight=-0.5,  # Gentle penalty for collision avoidance
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*"), "prediction_distance": 2.0}
    )
    
    # Action smoothness (prevent oscillation)
    action_rate = RewTerm(
        func=mdp_utils.action_rate_l2,
        weight=-0.1,  # Very small penalty
    )
    
    # Basic survival (minimal weight - should be maintained naturally)
    alive = RewTerm(
        func=mdp_utils.is_alive,
        weight=0.1  # Tiny reward for staying alive
    )

    # === REMOVED: Confusing Rewards ===
    # These rewards from original setup confused the agent:
    # - Velocity tracking: Rewarded speed over goal achievement
    # - Complex multi-objective: Too many competing signals
    # - High penalty weights: Created risk-averse behavior
    
    # Key insight from RLRoverLab: Clear, simple objective hierarchy