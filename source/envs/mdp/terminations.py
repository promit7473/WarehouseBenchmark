# source/envs/mdp/terminations.py
"""Termination conditions for warehouse navigation environment.

This module provides termination functions following RLRoverLab patterns:
- Proper type hints
- Contact sensor integration
- Clear documentation
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

# Import centralized warehouse constants
from source.envs.warehouse_constants import WAREHOUSE_NAVIGABLE_BOUNDS

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def is_success(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float = 1.5
) -> torch.Tensor:
    """
    Determine whether the robot has successfully reached the target waypoint.

    Args:
        env: The environment instance
        command_name: Name of the command containing target position
        threshold: Distance threshold for success (meters)

    Returns:
        Boolean tensor indicating success for each environment
    """
    # Get target position from command manager
    target = env.command_manager.get_command(command_name)
    target_position = target[:, :2]

    # Get robot position
    robot_pos = env.scene["robot"].data.root_pos_w[:, :2]

    # Calculate distance to target
    distance = torch.norm(target_position - robot_pos, p=2, dim=-1)

    # Check if heading is also aligned (if available in command)
    if target.shape[1] > 2:
        target_heading = target[:, 2]
        from source.envs.utils import quaternion_to_yaw
        robot_heading = quaternion_to_yaw(env.scene["robot"].data.root_state_w)
        heading_diff = torch.abs(torch.atan2(
            torch.sin(target_heading - robot_heading),
            torch.cos(target_heading - robot_heading)
        ))
        # Success requires both position and heading alignment
        return (distance < threshold) & (heading_diff < 0.3)

    return distance < threshold


def far_from_target(
    env: ManagerBasedRLEnv,
    command_name: str,
    max_distance: float = 50.0
) -> torch.Tensor:
    """
    Terminate if robot is too far from target (wandered off).

    Args:
        env: The environment instance
        command_name: Name of the command containing target position
        max_distance: Maximum allowed distance from target (meters)

    Returns:
        Boolean tensor indicating if robot is too far for each environment
    """
    target = env.command_manager.get_command(command_name)
    target_position = target[:, :2]

    robot_pos = env.scene["robot"].data.root_pos_w[:, :2]
    distance = torch.norm(target_position - robot_pos, p=2, dim=-1)

    return distance > max_distance


def collision_with_obstacles(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0
) -> torch.Tensor:
    """
    Check for collision with obstacles using contact sensor.

    Args:
        env: The environment instance
        sensor_cfg: Configuration for the contact sensor
        threshold: Force threshold for collision detection

    Returns:
        Boolean tensor indicating collision for each environment
    """
    try:
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

        # Get force matrix and reshape to (num_envs, num_bodies, 3)
        force_matrix = contact_sensor.data.force_matrix_w.view(env.num_envs, -1, 3)

        # Calculate normalized forces per body
        normalized_forces = torch.norm(force_matrix, dim=-1)

        # Sum forces across all bodies and check threshold
        total_force = torch.sum(normalized_forces, dim=-1)
        forces_active = total_force > threshold

        return forces_active

    except (KeyError, AttributeError):
        # Contact sensor not available - return no collision
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def robot_fallen(
    env: ManagerBasedRLEnv,
    min_height: float = -0.5
) -> torch.Tensor:
    """
    Check if robot has fallen below minimum height threshold.

    Args:
        env: The environment instance
        min_height: Minimum allowed Z height (meters)

    Returns:
        Boolean tensor indicating if robot has fallen for each environment
    """
    robot_z = env.scene["robot"].data.root_state_w[:, 2]
    return robot_z < min_height


def robot_flipped(
    env: ManagerBasedRLEnv,
    max_tilt: float = 1.0
) -> torch.Tensor:
    """
    Check if robot has flipped over (excessive tilt).

    Args:
        env: The environment instance
        max_tilt: Maximum allowed tilt angle (radians, ~57 degrees)

    Returns:
        Boolean tensor indicating if robot is flipped for each environment
    """
    # Get projected gravity vector (indicates orientation)
    # When upright, projected gravity should be [0, 0, -1]
    # Get quaternion from root state
    quat = env.scene["robot"].data.root_state_w[:, 3:7]  # [qw, qx, qy, qz]

    # Calculate the z-component of the up vector after rotation
    # For quaternion (w, x, y, z), the rotated z-axis z-component is:
    # 1 - 2*(qx^2 + qy^2)
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    up_z = 1.0 - 2.0 * (qx * qx + qy * qy)

    # Tilt angle from vertical (acos of up_z component)
    # Clamp to valid range for acos
    up_z_clamped = torch.clamp(up_z, -1.0, 1.0)
    tilt_angle = torch.acos(up_z_clamped)

    return tilt_angle > max_tilt


def out_of_bounds(
    env: ManagerBasedRLEnv,
    bounds: tuple = WAREHOUSE_NAVIGABLE_BOUNDS,
    margin: float = 0.0  # Navigable bounds already include safety margin
) -> torch.Tensor:
    """
    Check if robot has left the warehouse bounds.

    Args:
        env: The environment instance
        bounds: Warehouse bounds (min_x, min_y, max_x, max_y) in meters (default: navigable bounds)
        margin: Additional safety margin from bounds (meters) - default 0.0 since navigable bounds already include margin

    Returns:
        Boolean tensor indicating if robot is out of bounds for each environment
    """
    min_x, min_y, max_x, max_y = bounds
    robot_pos = env.scene["robot"].data.root_state_w[:, :2]

    x_out = (robot_pos[:, 0] < min_x + margin) | (robot_pos[:, 0] > max_x - margin)
    y_out = (robot_pos[:, 1] < min_y + margin) | (robot_pos[:, 1] > max_y - margin)

    return x_out | y_out
