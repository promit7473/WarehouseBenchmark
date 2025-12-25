from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as TermTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
import os
from typing import Sequence

# 1. Correct Import for Joint Velocity
from isaaclab.envs.mdp.actions.actions_cfg import JointVelocityActionCfg

# 2. General MDP import
import isaaclab.envs.mdp as mdp_utils

# 3. Local Imports
import source.envs.mdp as local_mdp
from source.envs.config.warehouse_static_cfg import WarehouseSceneCfg

# 4. Simulation imports for PhysX configuration
from isaaclab.sim import PhysxCfg
from isaaclab.sim import SimulationCfg as SimCfg

# 5. Curriculum learning imports
from source.envs.curriculum.warehouse_curriculum import (
    obstacle_density_curriculum,
    waypoint_complexity_curriculum,
    sensor_availability_curriculum
)

import torch
import gymnasium as gym
import numpy as np
from source.envs.utils import quaternion_to_yaw

# Import centralized warehouse constants
from source.envs.warehouse_constants import (
    WAREHOUSE_MIN_X, WAREHOUSE_MAX_X, WAREHOUSE_MIN_Y, WAREHOUSE_MAX_Y,
    WAREHOUSE_WIDTH_M, WAREHOUSE_HEIGHT_M, WAREHOUSE_WALL_MARGIN_M,
    WAREHOUSE_AISLE_WIDTH_M, WAREHOUSE_BOUNDS,
    WAREHOUSE_NAVIGABLE_X_MIN, WAREHOUSE_NAVIGABLE_X_MAX,
    WAREHOUSE_NAVIGABLE_Y_MIN, WAREHOUSE_NAVIGABLE_Y_MAX,
    WAREHOUSE_NAVIGABLE_BOUNDS, ROBOT_SPAWN_HEIGHT_M
)

# Enhanced Spawn System - Warehouse-Aware Reset
class WarehouseSpawnManager:
    """
    Warehouse-aware spawn management system.
    Ensures robots always spawn in valid, reachable positions by checking bounds and obstacles.
    """
    
    def __init__(self, warehouse_bounds: tuple = WAREHOUSE_BOUNDS, device: torch.device | None = None):
        # Auto-detect warehouse bounds based on USD file or defaults
        self.warehouse_bounds = self._detect_warehouse_bounds(warehouse_bounds)
        self.aisle_width = WAREHOUSE_AISLE_WIDTH_M
        self.wall_margin = WAREHOUSE_WALL_MARGIN_M
        if device is not None:
            self._device = device
        else:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obstacle_positions = self._define_spawn_obstacles(self._device)

    def _detect_warehouse_bounds(self, default_bounds: tuple) -> tuple:
        """
        Define bounds (min_x, min_y, max_x, max_y).
        Based on full_warehouse.usd: 36m × 75m warehouse.
        """
        # Use global warehouse bounds constant
        return WAREHOUSE_BOUNDS
        
    def _define_spawn_obstacles(self, device: torch.device | None = None):
        """Define static obstacles (shelves) to avoid DURING SPAWN ONLY.

        IMPORTANT: These hardcoded positions are ONLY used for spawn validation
        to ensure robots don't spawn inside shelves. During training, the robot
        uses LiDAR sensor data for obstacle detection, NOT these hardcoded positions.

        This is necessary because we need to know obstacle positions before the
        simulation starts to prevent invalid spawns.

        Args:
            device: torch device to place the tensor on (default: cuda if available, else cpu)
        """
        if device is None:
            device = self._device

        # Approximate positions of major obstacles in full_warehouse.usd
        # These are rough estimates used only for spawn validation
        obstacles = torch.tensor([
            [0.0, 0.0],      # Center storage
            [7.0, 0.0],      # Right storage
            [-7.0, 0.0],     # Left storage
            [0.0, 7.0],      # Upper storage
            [0.0, -7.0],     # Lower storage
            [14.0, 0.0],     # Loading dock (within bounds)
            [-14.0, 0.0],    # Receiving dock (within bounds)
            [12.0, 12.0],    # Corner clutter (within bounds)
            [-12.0, 12.0],
            [12.0, -12.0],
            [-12.0, -12.0],
        ], dtype=torch.float32, device=device)
        return obstacles
        
    def generate_valid_spawn_positions(self, env, num_positions: int, device: torch.device) -> torch.Tensor:
        """
        Corner-based spawning for warehouse.usd with fallback to distributed spawning.
        Places robots in corners first, then distributes remaining across navigable area.

        Args:
            env: The environment instance
            num_positions: Number of spawn positions to generate
            device: torch device to place tensors on

        Returns:
            Tensor of shape (num_positions, 3) with [x, y, yaw] for each position
        """
        # Update obstacle positions device if needed
        if self.obstacle_positions.device != device:
            self.obstacle_positions = self.obstacle_positions.to(device)

        # Define four corners with margins (using navigable bounds)
        corners = torch.tensor([
            [WAREHOUSE_NAVIGABLE_X_MIN + 2.0, WAREHOUSE_NAVIGABLE_Y_MIN + 2.0],  # Bottom-left
            [WAREHOUSE_NAVIGABLE_X_MAX - 2.0, WAREHOUSE_NAVIGABLE_Y_MIN + 2.0],  # Bottom-right
            [WAREHOUSE_NAVIGABLE_X_MIN + 2.0, WAREHOUSE_NAVIGABLE_Y_MAX - 2.0],  # Top-left
            [WAREHOUSE_NAVIGABLE_X_MAX - 2.0, WAREHOUSE_NAVIGABLE_Y_MAX - 2.0],  # Top-right
        ], dtype=torch.float32, device=device)

        positions = []
        for i in range(num_positions):
            if i < 4:
                # First 4 robots spawn in corners with small random offset
                corner = corners[i % 4]
                offset = (torch.rand(2, device=device) - 0.5) * 2.0  # +/- 1m offset
                x = corner[0] + offset[0]
                y = corner[1] + offset[1]
            else:
                # Remaining robots distributed across navigable area
                x = torch.rand(1, device=device) * (WAREHOUSE_NAVIGABLE_X_MAX - WAREHOUSE_NAVIGABLE_X_MIN) + WAREHOUSE_NAVIGABLE_X_MIN
                y = torch.rand(1, device=device) * (WAREHOUSE_NAVIGABLE_Y_MAX - WAREHOUSE_NAVIGABLE_Y_MIN) + WAREHOUSE_NAVIGABLE_Y_MIN
                x = x.squeeze()
                y = y.squeeze()

            # Random orientation (yaw around Z-axis)
            yaw = torch.rand(1, device=device).squeeze() * 2 * 3.14159 - 3.14159

            # Stack as [x, y, yaw] - z and quaternion are added by warehouse_aware_reset
            pos = torch.stack([x, y, yaw])
            positions.append(pos)

        return torch.stack(positions)

    def _validate_position_in_warehouse(self, position: torch.Tensor, warehouse_bounds: tuple) -> bool:
        """
        Validate that position is within warehouse bounds.
        """
        x, y = position[0], position[1]
        min_x, min_y, max_x, max_y = warehouse_bounds

        # Check bounds with small margin
        margin = 0.5
        return (min_x + margin <= x <= max_x - margin and
                min_y + margin <= y <= max_y - margin)

    def _generate_positions_in_area(self, num_positions: int, area_bounds: tuple, device: torch.device) -> torch.Tensor:
        """
        Generate positions within a specific bounded area.
        """
        min_x, min_y, max_x, max_y = area_bounds
        valid_positions = []
        max_attempts_per_robot = 30
        min_dist_sq = 2.0 ** 2  # 2m minimum distance

        for i in range(num_positions):
            position_found = False

            for attempt in range(max_attempts_per_robot):
                # Generate position within the specified area
                x = (torch.rand(1, device=device) * (max_x - min_x)) + min_x
                y = (torch.rand(1, device=device) * (max_y - min_y)) + min_y
                yaw = (torch.rand(1, device=device) * 6.28) - 3.14

                candidate = torch.tensor([x.item(), y.item(), yaw.item()], device=device)

                # Validate position
                if not self._validate_position_in_area(candidate, area_bounds):
                    continue

                # Check distance from other robots in this area
                if len(valid_positions) > 0:
                    existing = torch.stack(valid_positions)
                    dists = torch.sum((existing[:, :2] - candidate[:2])**2, dim=1)
                    if torch.min(dists) < min_dist_sq:
                        continue

                valid_positions.append(candidate)
                position_found = True
                break

            if not position_found:
                # Fallback: place in center of area
                center_x = (min_x + max_x) / 2
                center_y = (min_y + max_y) / 2
                fallback = torch.tensor([center_x, center_y, 0.0], device=device)
                valid_positions.append(fallback)

        return torch.stack(valid_positions)

    def _validate_position_in_area(self, position: torch.Tensor, area_bounds: tuple) -> bool:
        """
        Validate position is within the specified area bounds.
        """
        x, y = position[0], position[1]
        min_x, min_y, max_x, max_y = area_bounds

        # Check area bounds
        if not (min_x <= x <= max_x and min_y <= y <= max_y):
            return False

        # Check against warehouse obstacles (using original bounds)
        return self._validate_spawn_position(position)

    def _generate_standard_distribution(self, num_positions: int, device: torch.device) -> torch.Tensor:
        """
        Standard distribution for smaller numbers of robots.
        """
        valid_positions = []
        max_attempts_per_robot = 50
        min_dist_sq = 2.5 ** 2

        for i in range(num_positions):
            position_found = False

            for attempt in range(max_attempts_per_robot):
                # 70% aisle, 30% random
                if attempt < max_attempts_per_robot * 0.7:
                    candidate = self._generate_candidate_position(device)
                else:
                    candidate = self._generate_random_position(device)

                if not self._validate_spawn_position(candidate):
                    continue

                # Check distance from other robots
                if len(valid_positions) > 0:
                    existing = torch.stack(valid_positions)
                    dists = torch.sum((existing[:, :2] - candidate[:2])**2, dim=1)
                    if torch.min(dists) < min_dist_sq:
                        continue

                valid_positions.append(candidate)
                position_found = True
                break

            if not position_found:
                valid_positions.append(self._generate_fallback_position(device))

        return torch.stack(valid_positions)

    def _generate_random_position(self, device):
        """Pure random position within bounds."""
        min_x, min_y, max_x, max_y = self.warehouse_bounds
        x = (torch.rand(1, device=device) * (max_x - min_x)) + min_x
        y = (torch.rand(1, device=device) * (max_y - min_y)) + min_y
        yaw = (torch.rand(1, device=device) * 2 * 3.14159) - 3.14159
        return torch.tensor([x.item(), y.item(), yaw.item()], device=device)

    def _generate_candidate_position(self, device: torch.device) -> torch.Tensor:
        """Generate a candidate spawn position aligned with aisles."""
        min_x, min_y, max_x, max_y = self.warehouse_bounds
        
        # Grid of potential aisle centers
        aisle_centers_x = torch.arange(min_x + self.wall_margin, max_x - self.wall_margin, self.aisle_width, device=device)
        aisle_centers_y = torch.arange(min_y + self.wall_margin, max_y - self.wall_margin, self.aisle_width, device=device)
        
        if len(aisle_centers_x) == 0 or len(aisle_centers_y) == 0:
            return self._generate_random_position(device)

        # Pick random aisle
        x = aisle_centers_x[torch.randint(0, len(aisle_centers_x), (1,))]
        y = aisle_centers_y[torch.randint(0, len(aisle_centers_y), (1,))]
        
        # Add jitter
        x += (torch.rand(1, device=device) - 0.5) * 1.5
        y += (torch.rand(1, device=device) - 0.5) * 1.5
        yaw = (torch.rand(1, device=device) * 6.28) - 3.14
        
        return torch.tensor([x.item(), y.item(), yaw.item()], device=device)

    def _generate_challenging_position(self, device: torch.device) -> torch.Tensor:
        """Generate positions near obstacles for challenging training scenarios."""
        # Pick a random obstacle and spawn near it (but not too close)
        obstacle_idx = torch.randint(0, len(self.obstacle_positions), (1,))
        obstacle_pos = self.obstacle_positions[obstacle_idx].squeeze()

        # Spawn 2.0-3.0m away from obstacle (challenging but not impossible)
        angle = torch.rand(1, device=device) * 2 * 3.14159
        distance = 2.0 + torch.rand(1, device=device) * 1.0  # 2.0-3.0m

        x = obstacle_pos[0] + distance * torch.cos(angle)
        y = obstacle_pos[1] + distance * torch.sin(angle)
        yaw = (torch.rand(1, device=device) * 6.28) - 3.14

        # Ensure within bounds
        min_x, min_y, max_x, max_y = self.warehouse_bounds
        x = torch.clamp(x, min_x + self.wall_margin, max_x - self.wall_margin)
        y = torch.clamp(y, min_y + self.wall_margin, max_y - self.wall_margin)

        return torch.tensor([x.item(), y.item(), yaw.item()], device=device)

    def _validate_spawn_position(self, position: torch.Tensor) -> bool:
        """Check if position collides with static obstacles or walls."""
        x, y = position[0], position[1]
        min_x, min_y, max_x, max_y = self.warehouse_bounds
        
        # 1. Bounds Check
        if not (min_x + self.wall_margin <= x <= max_x - self.wall_margin):
            return False
        if not (min_y + self.wall_margin <= y <= max_y - self.wall_margin):
            return False
            
        # 2. Obstacle Check
        pos_2d = position[:2].unsqueeze(0).to(self.obstacle_positions.device) # (1, 2)
        # Dist to all obstacles
        dists = torch.cdist(pos_2d, self.obstacle_positions)
        # Allow spawning closer to shelves (1.5m instead of 2.5m) for more challenging positions
        # But still maintain safety - don't spawn directly on obstacles
        if dists.min() < 1.5:
            return False
            
        return True
        
    def _generate_fallback_position(self, device: torch.device) -> torch.Tensor:
        """Guaranteed safe positions if generation fails.

        These positions are away from the center obstacle at [0,0] and within navigable bounds.
        """
        # Safe spots away from center obstacle (which is at [0.0, 0.0])
        # Minimum 3m from any obstacle for safety
        safe_spots = torch.tensor([
            [5.0, 5.0, 0.0],      # Northeast quadrant
            [-5.0, 5.0, 0.78],    # Northwest quadrant
            [-5.0, -5.0, -0.78],  # Southwest quadrant
            [5.0, -5.0, 1.57]     # Southeast quadrant
        ], device=device)
        idx = torch.randint(0, len(safe_spots), (1,))
        # Add tiny noise to avoid exact stacking
        pos = safe_spots[idx].squeeze().clone()
        pos[:2] += (torch.rand(2, device=device) - 0.5) * 0.5
        return pos


# Global spawn manager instance
_spawn_manager = None

def get_spawn_manager() -> WarehouseSpawnManager:
    global _spawn_manager
    if _spawn_manager is None:
        _spawn_manager = WarehouseSpawnManager()
    return _spawn_manager

def normalize_quaternion(qw: torch.Tensor, qx: torch.Tensor,
                         qy: torch.Tensor, qz: torch.Tensor) -> tuple:
    """
    Normalize quaternion to unit length to ensure orthonormal transformation matrix.

    This prevents the "OrthogonalizeBasis did not converge" USD warning.
    Handles degenerate quaternions (all zeros) by replacing with identity quaternion.
    """
    norm_sq = qw * qw + qx * qx + qy * qy + qz * qz

    # Check for degenerate quaternion (all zeros) - prevents NaN propagation
    # If norm is too small, replace with identity quaternion (1, 0, 0, 0)
    is_degenerate = norm_sq < 1e-10
    norm_sq = torch.where(is_degenerate, torch.ones_like(norm_sq), norm_sq)

    norm = torch.sqrt(norm_sq)
    qw_norm = torch.where(is_degenerate, torch.ones_like(qw), qw / norm)
    qx_norm = torch.where(is_degenerate, torch.zeros_like(qx), qx / norm)
    qy_norm = torch.where(is_degenerate, torch.zeros_like(qy), qy / norm)
    qz_norm = torch.where(is_degenerate, torch.zeros_like(qz), qz / norm)

    return qw_norm, qx_norm, qy_norm, qz_norm


def warehouse_aware_reset(env, env_ids, pose_range, velocity_range, asset_cfg):
    """
    Reset function called by the Event Manager.

    Generates valid spawn positions and applies them to the robot with
    properly normalized quaternions to avoid USD orthonormalization warnings.
    """
    spawn_manager = get_spawn_manager()
    device = env.device
    num_resets = len(env_ids)

    # 1. Get valid positions [x, y, yaw]
    valid_poses_2d = spawn_manager.generate_valid_spawn_positions(env, num_resets, device)

    # 2. Convert to tensor if needed
    if isinstance(valid_poses_2d, list):
        valid_poses_2d = torch.stack(valid_poses_2d)

    # 3. Convert to Isaac Lab format [x, y, z, qw, qx, qy, qz]
    positions = valid_poses_2d[:, :2]
    yaws = valid_poses_2d[:, 2]

    # Calculate Quaternions (rotate around Z-axis for yaw in 3D space)
    # Use double precision for intermediate calculation to reduce numerical error
    yaws_half = yaws.double() / 2.0
    qw = torch.cos(yaws_half).float()
    qx = torch.zeros_like(yaws)
    qy = torch.zeros_like(yaws)
    qz = torch.sin(yaws_half).float()

    # Normalize quaternion to ensure unit length (prevents orthonormalization warnings)
    qw, qx, qy, qz = normalize_quaternion(qw, qx, qy, qz)

    # Robot Z height (spawn with safety clearance - increased from 0.1m to 0.5m like RLRoverLab)
    z_height = torch.full((num_resets, 1), 0.5, device=device, dtype=torch.float32)

    # Assemble full state [x, y, z, qw, qx, qy, qz]
    full_poses = torch.cat([
        positions,
        z_height,
        qw.unsqueeze(1),
        qx.unsqueeze(1),
        qy.unsqueeze(1),
        qz.unsqueeze(1)
    ], dim=1)

    # Apply to simulation
    asset = env.scene[asset_cfg.name]
    asset.write_root_pose_to_sim(full_poses, env_ids)

    # Reset velocities to zero
    zero_vels = torch.zeros((num_resets, 6), device=device, dtype=torch.float32)
    asset.write_root_velocity_to_sim(zero_vels, env_ids)


class WarehouseWaypointCommand(mdp_utils.UniformPose2dCommand):
    """Custom waypoint command that generates waypoints within warehouse bounds."""

    def _resample_command(self, env_ids):
        # Override parent implementation to generate waypoints within warehouse bounds
        # AND avoid obstacles (shelves) by checking distance to known shelf positions

        # CURRICULUM LEARNING: Start with nearby waypoints for easier training
        # Use robot's current position as center, gradually increase range
        robot_pos = self.robot.data.root_pos_w[env_ids, :2]  # [num_envs, 2]

        # Distance range based on curriculum (progressive difficulty)
        min_distance = 3.0  # Minimum distance from robot (avoid trivial goals)
        # Use curriculum max_distance if available, otherwise default to 8.0m (early training)
        max_distance = getattr(self, 'curriculum_max_distance', 8.0)

        # Warehouse navigable bounds (with 2m safety margins)
        WAREHOUSE_X_MIN = -24.3
        WAREHOUSE_X_MAX = 3.5
        WAREHOUSE_Y_MIN = -21.4
        WAREHOUSE_Y_MAX = 28.6

        # Minimum safe distance from obstacles - robot needs space to navigate
        MIN_OBSTACLE_DISTANCE = 2.5  # meters (robot can fit + safety margin)

        # Get approximate shelf positions from spawn manager
        spawn_manager = get_spawn_manager()
        obstacles = spawn_manager.obstacle_positions  # [num_obstacles, 2]

        # Generate obstacle-free waypoints with retry logic
        r = torch.empty(len(env_ids), device=self.device)
        valid = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)

        # Try up to 100 times to find valid positions (more attempts for distance constraint)
        for attempt in range(100):
            need_waypoint = ~valid
            if not need_waypoint.any():
                break

            # Generate candidates using polar coordinates from robot position
            # This ensures waypoints are within [min_distance, max_distance] from robot
            num_needed = need_waypoint.sum().item()
            needed_robot_pos = robot_pos[need_waypoint]  # [num_needed, 2]

            # Sample distance and angle
            dist = torch.empty(num_needed, device=self.device).uniform_(min_distance, max_distance)
            angle = torch.empty(num_needed, device=self.device).uniform_(-torch.pi, torch.pi)

            # Convert to Cartesian coordinates relative to robot
            dx = dist * torch.cos(angle)
            dy = dist * torch.sin(angle)
            candidates = needed_robot_pos + torch.stack([dx, dy], dim=1)  # [num_needed, 2]

            # Clamp to warehouse bounds
            candidates[:, 0] = torch.clamp(candidates[:, 0], WAREHOUSE_X_MIN, WAREHOUSE_X_MAX)
            candidates[:, 1] = torch.clamp(candidates[:, 1], WAREHOUSE_Y_MIN, WAREHOUSE_Y_MAX)

            # Check distance to all obstacles: [num_needed, num_obstacles]
            dists_to_obstacles = torch.cdist(candidates, obstacles.float())
            min_obstacle_dist, _ = torch.min(dists_to_obstacles, dim=1)

            # Accept candidates that are far enough from all obstacles
            is_safe = min_obstacle_dist >= MIN_OBSTACLE_DISTANCE
            valid_idx = torch.where(need_waypoint)[0][is_safe]
            safe_pos = candidates[is_safe]

            if len(valid_idx) > 0:
                self.pos_command_w[env_ids[valid_idx], 0] = safe_pos[:, 0]
                self.pos_command_w[env_ids[valid_idx], 1] = safe_pos[:, 1]
                self.pos_command_w[env_ids[valid_idx], 2] = self.robot.data.default_root_state[env_ids[valid_idx], 2]
                valid[valid_idx] = True

        # Fallback: use warehouse center for any remaining invalid waypoints
        if not valid.all():
            remaining = torch.where(~valid)[0]
            self.pos_command_w[env_ids[remaining], 0] = (WAREHOUSE_X_MIN + WAREHOUSE_X_MAX) / 2
            self.pos_command_w[env_ids[remaining], 1] = (WAREHOUSE_Y_MIN + WAREHOUSE_Y_MAX) / 2
            self.pos_command_w[env_ids[remaining], 2] = self.robot.data.default_root_state[env_ids[remaining], 2]
            print(f"[WAYPOINT WARN] {len(remaining)} waypoints used safe fallback (warehouse center)")

        # Set heading to point towards target
        if self.cfg.simple_heading:
            target_vec = self.pos_command_w[env_ids] - self.robot.data.root_pos_w[env_ids]
            target_direction = torch.atan2(target_vec[:, 1], target_vec[:, 0])
            from isaaclab.utils.math import wrap_to_pi
            heading_noise = torch.empty(len(env_ids), device=self.device).uniform_(-0.5, 0.5)
            self.heading_command_w[env_ids] = wrap_to_pi(target_direction + heading_noise)
        else:
            self.heading_command_w[env_ids] = r.uniform_(*self.cfg.ranges.heading)

        num_valid = valid.sum().item()
        print(f"[WAYPOINT] {len(env_ids)} goals: {num_valid} valid ({min_distance}-{max_distance}m), {len(env_ids)-num_valid} fallback")


@configclass
class ActionsCfg:
    # Jackal differential drive: 4 wheel joints (front + rear on each side)
    body_vel = JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["front_left_wheel_joint", "front_right_wheel_joint",
                     "rear_left_wheel_joint", "rear_right_wheel_joint"],
        scale=10.0,
    )

@configclass
class CommandsCfg:
    waypoint_nav = mdp_utils.UniformPose2dCommandCfg(
        class_type=WarehouseWaypointCommand,  # Use custom command with bounds checking
        asset_name="robot",
        simple_heading=True,
        resampling_time_range=(60.0, 60.0),  # Resample at episode end (changed from 1000.0)
        debug_vis=True,  # Enable waypoint visualization for debugging
        ranges=mdp_utils.UniformPose2dCommandCfg.Ranges(
            pos_x=(-6.0, 6.0),  # ±6m range around each environment (very conservative)
            pos_y=(-5.0, 5.0),  # ±5m range around each environment (very conservative)
            heading=(-3.14, 3.14),
        ),
    )

@configclass
class CurriculumCfg:
    """
    Curriculum configuration for progressive waypoint difficulty.
    Gradually increases waypoint distance as agent improves.
    """
    # Waypoint distance curriculum: Start at 8m, scale up to 25m
    waypoint_distance = CurrTerm(
        func=local_mdp.waypoint_distance_curriculum,
        params={"min_dist": 3.0, "max_dist": 25.0}
    )

@configclass
class WarehouseEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the warehouse RL environment."""

    # 1. Episode Settings
    episode_length_s = 60.0  # Increased from 10.0 for realistic warehouse navigation
    decimation = 6  # Increased from 2 (following RLRoverLab pattern)

    # 2. Scene - Warehouse environment
    scene = WarehouseSceneCfg(num_envs=32, env_spacing=3.5)

    # 3. Simulation PhysX Configuration (RLRoverLab pattern)
    sim: SimCfg = SimCfg(
        dt=1/60.0,  # 60 FPS simulation
        physx=PhysxCfg(
            enable_stabilization=True,
            gpu_max_rigid_contact_count=8388608,
            gpu_max_rigid_patch_count=262144,
            gpu_found_lost_pairs_capacity=2**21,
            gpu_found_lost_aggregate_pairs_capacity=2**25,
            gpu_total_aggregate_pairs_capacity=2**21,
            gpu_max_soft_body_contacts=1048576,
            gpu_max_particle_contacts=1048576,
            gpu_heap_capacity=67108864,
            gpu_temp_buffer_capacity=16777216,
            gpu_max_num_partitions=8,
            gpu_collision_stack_size=2**28,
            friction_correlation_distance=0.025,
            friction_offset_threshold=0.04,
            bounce_threshold_velocity=2.0,
        )
    )

    # 4. Actions
    actions = ActionsCfg()

    # 5. Managers
    observations = local_mdp.ObservationsCfg()
    rewards = local_mdp.RewardsCfg()
    commands = CommandsCfg()

    # 6. Events (Spawning + Domain Randomization)
    events = {
        # Robot reset and spawning
        "reset_robot": EventTerm(
            func=warehouse_aware_reset,
            mode="reset",
            params={
                "pose_range": {}, # Handled internally by spawn manager
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),

        # Domain Randomization for Sim-to-Real Transfer
        # Randomize robot mass (80%-120% of nominal)
        "randomize_robot_mass": EventTerm(
            func=mdp_utils.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "mass_distribution_params": (0.8, 1.2),
                "operation": "scale",
            },
        ),

        # Randomize ground friction (70%-130% of nominal)
        "randomize_ground_friction": EventTerm(
            func=mdp_utils.randomize_rigid_body_material,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "static_friction_range": (0.7, 1.3),
                "dynamic_friction_range": (0.6, 1.2),
                "restitution_range": (0.0, 0.1),
                "num_buckets": 64,  # Match number of environments
            },
        ),

        # Randomize actuator gains (90%-110% of nominal)
        "randomize_actuator_gains": EventTerm(
            func=mdp_utils.randomize_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stiffness_distribution_params": (0.9, 1.1),
                "damping_distribution_params": (0.9, 1.1),
                "operation": "scale",
            },
        ),

        # Randomize joint velocities at reset (small variations)
        "randomize_joint_velocities": EventTerm(
            func=mdp_utils.reset_joints_by_offset,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*_wheel_joint"),
                "position_range": (-0.05, 0.05),
                "velocity_range": (-0.1, 0.1),
            },
        ),
    }

    # 7. Terminations
    terminations = {
        "time_out": TermTerm(func=mdp_utils.time_out, time_out=True),
        "robot_fallen": TermTerm(
            func=lambda env: env.scene["robot"].data.root_state_w[:, 2] < -0.5,
            time_out=False
        ),
          "warehouse_collision": TermTerm(
               # Proper rectangular bounds check using navigable area constants
               # Warehouse.usd bounds with 2m safety margins
               func=lambda env: (env.scene["robot"].data.root_state_w[:, 0] < WAREHOUSE_NAVIGABLE_X_MIN) |
                                (env.scene["robot"].data.root_state_w[:, 0] > WAREHOUSE_NAVIGABLE_X_MAX) |
                                (env.scene["robot"].data.root_state_w[:, 1] < WAREHOUSE_NAVIGABLE_Y_MIN) |
                                (env.scene["robot"].data.root_state_w[:, 1] > WAREHOUSE_NAVIGABLE_Y_MAX),
               time_out=False
           )
    }

    # 8. Curriculum Learning (Progressive Difficulty)
    curriculum: CurriculumCfg = CurriculumCfg()


def adjust_env_origins_for_warehouse(env):
    """
    Create custom environment origins that fit within warehouse bounds.

    This replaces the GridCloner's origins with a custom layout that ensures
    all environments are safely within warehouse walls.
    """
    num_envs = len(env.scene.env_origins)

    # Warehouse navigable bounds (with safety margins)
    # X: [-24.3, 3.5] (27.8m width), Y: [-21.4, 28.6] (50m height)
    x_min, x_max = -24.3, 3.5
    y_min, y_max = -21.4, 28.6

    # Calculate optimal grid layout
    env_spacing = 3.5  # Match the spacing used
    width = x_max - x_min
    height = y_max - y_min

    # Calculate grid dimensions to fit as many environments as possible
    envs_per_row = max(1, int(width / env_spacing))
    envs_per_col = max(1, int(height / env_spacing))

    # Adjust to fit exactly the requested number of environments
    total_possible = envs_per_row * envs_per_col
    if num_envs > total_possible:
        # If we have more environments than can fit in a perfect grid,
        # increase the number of rows
        envs_per_row = min(num_envs, envs_per_row + 1)
        envs_per_col = (num_envs + envs_per_row - 1) // envs_per_row  # Ceiling division

    # Create grid origins centered within navigable bounds
    origins = []
    start_x = x_min + (width - (envs_per_row - 1) * env_spacing) / 2
    start_y = y_min + (height - (envs_per_col - 1) * env_spacing) / 2

    for i in range(num_envs):
        row = i // envs_per_row
        col = i % envs_per_row
        x = start_x + col * env_spacing
        y = start_y + row * env_spacing
        origins.append([x, y, 0.0])

    # Convert to tensor
    new_origins = torch.tensor(origins, dtype=torch.float32, device=env.scene.env_origins.device)

    # Override the scene's origins
    if env.scene._terrain is not None:
        env.scene._terrain.env_origins = new_origins
    else:
        env.scene._default_env_origins = new_origins

    print(f"[INFO] Created custom environment layout: {envs_per_row}x{envs_per_col} grid within warehouse bounds")
    print(f"[DEBUG] Environment origins (first 8):")
    for i in range(min(8, len(new_origins))):
        x, y, z = new_origins[i]
        in_warehouse = (-26.3 <= x <= 5.5) and (-23.4 <= y <= 30.6)
        status = "✓" if in_warehouse else "✗ OUTSIDE"
        print(f"  Env {i} origin: ({x:.1f}, {y:.1f}, {z:.1f}) {status}")


def log_waypoint_bounds_check(env, step_count=0):
    """Check and log waypoint positions to ensure they're within warehouse bounds."""
    try:
        waypoint_term = env.command_manager.get_term("waypoint_nav")
        if hasattr(waypoint_term, 'pos_command_w'):
            positions = waypoint_term.pos_command_w
            if step_count % 100 == 0:  # Log every 100 steps
                print(f"[WAYPOINT CHECK] Step {step_count}:")
                outside_count = 0
                for i in range(len(positions)):
                    x, y, z = positions[i]
                    in_warehouse = (-26.3 <= x <= 5.5) and (-23.4 <= y <= 30.6)
                    if not in_warehouse:
                        outside_count += 1
                        print(f"  ✗ Env {i}: ({x:.1f}, {y:.1f}) OUTSIDE warehouse!")
                if outside_count == 0:
                    print(f"  ✓ All {len(positions)} waypoints within warehouse bounds")
                else:
                    print(f"  ⚠️ {outside_count}/{len(positions)} waypoints outside warehouse!")
    except Exception as e:
        if step_count % 500 == 0:  # Less frequent error logging
            print(f"[WAYPOINT CHECK] Error checking waypoints: {e}")