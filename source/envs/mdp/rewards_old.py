# source/envs/mdp/rewards.py

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp_utils
import torch
from isaaclab.envs import ManagerBasedRLEnv

def waypoint_reach_reward(env):
    """Reward for proximity to waypoint."""
    robot_pos = env.scene["robot"].data.root_state_w[:, :2]
    waypoint = env.command_manager.get_command("waypoint_nav")[:, :2]
    distance = torch.norm(robot_pos - waypoint, dim=1)
    return 1.0 / (1.0 + distance)


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


def waypoint_reached_reward(env, command_name: str = "waypoint_nav", distance_threshold: float = 1.5):
    """Large reward when robot reaches a waypoint - PRIMARY OBJECTIVE."""
    current_pos = env.scene["robot"].data.root_pos_w[:, :2]
    target_pos = env.command_manager.get_command(command_name).target[:, :2]
    
    distance = torch.norm(target_pos - current_pos, dim=1)
    reached = distance < distance_threshold
    
    return reached.float()


def waypoint_progress_reward(env, command_name: str = "waypoint_nav"):
    """Reward for making progress toward current waypoint."""
    current_pos = env.scene["robot"].data.root_pos_w[:, :2]
    target_pos = env.command_manager.get_command(command_name).target[:, :2]
    
    # Calculate distance to target
    distance = torch.norm(target_pos - current_pos, dim=1)
    
    # Reward inversely proportional to distance (closer = more reward)
    # Normalize to [0, 1] range
    max_distance = 20.0  # Maximum expected distance in warehouse
    progress = torch.clamp(1.0 - (distance / max_distance), 0.0, 1.0)
    
    return progress


def collision_prediction_penalty(env, sensor_cfg: SceneEntityCfg, prediction_distance: float = 2.0):
    """Penalty for getting too close to obstacles (collision prediction)."""
    # Use simplified distance-based collision prediction
    robot_pos = env.scene["robot"].data.root_state_w[:, :2]

    # Define obstacle positions (full_warehouse.usd warehouse layout: 36m x 74.82m)
    obstacle_positions = torch.tensor([
        [5.0, 5.0], [-5.0, 5.0], [5.0, -5.0], [-5.0, -5.0],  # Inner corners
        [10.0, 0.0], [-10.0, 0.0], [0.0, 15.0], [0.0, -15.0],  # Mid edges (adjusted for 60m Y)
        [16.0, 20.0], [-16.0, 20.0], [16.0, -20.0], [-16.0, -20.0]  # Outer shelves
    ], device=env.device)

    # Calculate minimum distance to any obstacle for each environment
    # robot_pos: (num_envs, 2), obstacle_positions: (num_obstacles, 2)
    distances = torch.cdist(robot_pos, obstacle_positions)  # (num_envs, num_obstacles)
    min_distances = distances.min(dim=1)[0]  # (num_envs,)
    obstacle_penalty = torch.exp(-min_distances / prediction_distance) * 2.0

    return -obstacle_penalty


def path_efficiency_reward(env, command_name: str):
    """Reward for taking efficient paths in warehouse aisles."""
    robot_pos = env.scene["robot"].data.root_state_w[:, :2]
    waypoint = env.command_manager.get_command(command_name)[:, :2]

    # Calculate straight-line distance to goal
    straight_distance = torch.norm(waypoint - robot_pos, dim=1)

    # Calculate path efficiency: reward for moving towards goal vs wandering
    # Track robot's movement history (simplified - just current heading vs optimal)
    robot_yaw = env.scene["robot"].data.root_state_w[:, 6]  # Yaw angle
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
    robot_yaw = env.scene["robot"].data.root_state_w[:, 6]
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

    aisle_movement_bonus = torch.exp(-aisle_alignment / (torch.pi / 6)) * far_from_goal * 0.2

    return alignment_bonus + aisle_movement_bonus


def shelf_interaction_bonus(env, sensor_cfg: SceneEntityCfg, interaction_distance: float = 1.5):
    """Bonus for appropriate proximity to shelves (for picking tasks)."""
    # This would be used in future picking/placing tasks
    # Based on warehouse logistics tutorial for item manipulation
    robot_pos = env.scene["robot"].data.root_state_w[:, :2]

    # Placeholder for shelf proximity detection
    shelf_bonus = torch.zeros(env.num_envs, device=env.device)

    return shelf_bonus

@configclass
class RewardsCfg:
    """Advanced reward terms for warehouse navigation."""

    # 1. Survival Reward
    alive = RewTerm(func=mdp_utils.is_alive, weight=0.5)

    # 2. Collision Penalty (Enhanced)
    collision = RewTerm(
        func=mdp_utils.undesired_contacts,
        weight=-3.0,  # Increased penalty for warehouse safety
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*"), "threshold": 1.0}
    )

    # 3. Collision Prediction (Advanced MDP)
    collision_prediction = RewTerm(
        func=collision_prediction_penalty,
        weight=-1.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces"), "prediction_distance": 2.0}
    )

    # 4. Action Smoothness
    action_rate = RewTerm(
        func=mdp_utils.action_rate_l2,
        weight=-0.05
    )

    # 5. PRIMARY OBJECTIVE: Waypoint Navigation Rewards
    # Clear objective: Navigate to sequential waypoints efficiently
    waypoint_reached = RewTerm(
        func=waypoint_reached_reward,
        weight=10.0,  # High weight for primary objective
        params={"command_name": "waypoint_nav", "distance_threshold": 1.5}
    )

    waypoint_progress = RewTerm(
        func=waypoint_progress_reward,
        weight=2.0,  # Progress toward current waypoint
        params={"command_name": "waypoint_nav"}
    )

    # 6. SECONDARY: Navigation Efficiency
    path_efficiency = RewTerm(
        func=path_efficiency_reward,
        weight=0.5,  # Reduced weight - secondary objective
        params={"command_name": "waypoint_nav"}
    )

    # 7. Aisle Navigation (Warehouse-specific) - Simplified
    aisle_navigation = RewTerm(
        func=aisle_navigation_reward,
        weight=0.6,
        params={"command_name": "waypoint_nav"}
    )

    # 8. Shelf Interaction (For future picking tasks)
    shelf_interaction = RewTerm(
        func=shelf_interaction_bonus,
        weight=0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces"), "interaction_distance": 1.5}
    )

    # 9. Reward for reaching waypoint (inverse distance)
    waypoint_reach = RewTerm(
        func=waypoint_reach_reward,
        weight=2.0
    )


