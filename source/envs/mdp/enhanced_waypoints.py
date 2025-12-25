# source/envs/mdp/enhanced_waypoints.py

"""
Enhanced Waypoint System for Warehouse Navigation

This module provides a cleaner, more robust waypoint system inspired by RLRoverLab's
approach but adapted for warehouse sequential navigation tasks.
"""

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from typing import TYPE_CHECKING

# Import centralized warehouse constants
from source.envs.warehouse_constants import (
    WAREHOUSE_NAVIGABLE_BOUNDS,
    WAREHOUSE_AISLE_WIDTH_M,
    WAREHOUSE_WALL_MARGIN_M
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class WarehouseWaypointValidator:
    """
    Robust waypoint validation system inspired by RLRoverLab's approach.
    Ensures waypoints are always reachable and valid for warehouse navigation.
    """

    def __init__(self, warehouse_bounds: tuple = WAREHOUSE_NAVIGABLE_BOUNDS):
        self.warehouse_bounds = warehouse_bounds
        self.wall_margin = WAREHOUSE_WALL_MARGIN_M  # Safety margin from walls (already in navigable bounds)
        self.aisle_width = WAREHOUSE_AISLE_WIDTH_M  # Standard warehouse aisle width
        self.shelf_positions = self._define_warehouse_obstacles()
        
    def _define_warehouse_obstacles(self):
        """Define known obstacle positions in the warehouse layout."""
        # Based on typical warehouse layout - shelf positions and obstacles
        obstacles = torch.tensor([
            # Main shelving aisles (every 3.5m)
            [0.0, 0.0],    # Center area - often blocked
            [7.0, 0.0],    # Shelf positions
            [-7.0, 0.0],
            [0.0, 7.0],
            [0.0, -7.0],
            [14.0, 0.0],   # Outer shelves
            [-14.0, 0.0],
            [0.0, 14.0],
            [0.0, -14.0],
            # Corner obstacles
            [10.0, 10.0],
            [-10.0, 10.0],
            [10.0, -10.0],
            [-10.0, -10.0],
            # Loading dock areas
            [20.0, 0.0],   # Shipping area
            [-20.0, 0.0],  # Receiving area
        ], dtype=torch.float32)
        return obstacles
        
    def validate_waypoint_position(self, env, position: torch.Tensor) -> torch.Tensor:
        """
        Enhanced validation with multiple checks inspired by RLRoverLab's approach.
        
        Args:
            env: Environment instance
            position: Waypoint positions to validate (num_envs, 2)
            
        Returns:
            Boolean tensor indicating valid positions
        """
        device = position.device
        x, y = position[:, 0], position[:, 1]
        
        # 1. Basic bounds checking (warehouse_bounds: min_x, min_y, max_x, max_y)
        within_bounds = (x >= self.warehouse_bounds[0] + self.wall_margin) & \
                       (x <= self.warehouse_bounds[2] - self.wall_margin) & \
                       (y >= self.warehouse_bounds[1] + self.wall_margin) & \
                       (y <= self.warehouse_bounds[3] - self.wall_margin)
        
        # 2. Aisle alignment check - prefer positions in aisles, not between shelves
        grid_x = torch.round(x / self.aisle_width) * self.aisle_width
        grid_y = torch.round(y / self.aisle_width) * self.aisle_width
        
        # Distance to nearest aisle center
        dist_to_aisle_x = torch.abs(x - grid_x)
        dist_to_aisle_y = torch.abs(y - grid_y)
        min_dist_to_aisle = torch.minimum(dist_to_aisle_x, dist_to_aisle_y)
        
        # Prefer positions within 1.0m of aisle centers
        aisle_aligned = min_dist_to_aisle <= 1.0
        
        # 3. Obstacle clearance check
        # Move obstacles to same device as position
        obstacles_local = self.shelf_positions.to(device)
        
        # Calculate minimum distance to any obstacle for each waypoint
        # position: (num_envs, 2), obstacles: (num_obstacles, 2)
        distances = torch.cdist(position, obstacles_local)  # (num_envs, num_obstacles)
        min_distances = distances.min(dim=1)[0]  # (num_envs,)
        
        # Require minimum 2.0m clearance from obstacles
        obstacle_clear = min_distances >= 2.0
        
        # 4. Robot reachability check - ensure waypoint is reachable from current position
        try:
            robot_pos = env.scene["robot"].data.root_state_w[:, :2]
            distance_to_robot = torch.norm(position - robot_pos, dim=1)
            # Waypoints should be within reasonable navigation distance (5-30m)
            reachable = (distance_to_robot >= 3.0) & (distance_to_robot <= 30.0)
        except:
            # Fallback if robot not available
            reachable = torch.ones_like(within_bounds, dtype=torch.bool)
        
        # Combine all validation checks
        valid = within_bounds & aisle_aligned & obstacle_clear & reachable
        
        return valid
        
    def resample_invalid_waypoints(self, env, invalid_positions: torch.Tensor, 
                                 env_ids: torch.Tensor) -> torch.Tensor:
        """
        Resample invalid waypoints until valid ones are found.
        Inspired by RLRoverLab's validation loop approach.
        
        Args:
            env: Environment instance
            invalid_positions: Current invalid positions
            env_ids: Environment IDs that need resampling
            
        Returns:
            Valid waypoint positions
        """
        device = invalid_positions.device
        original_env_ids = env_ids
        max_attempts = 50  # Prevent infinite loops
        attempt = 0
        
        while len(env_ids) > 0 and attempt < max_attempts:
            # Generate new candidate positions
            new_positions = self._generate_candidate_positions(env, len(env_ids), device)
            
            # Validate new positions
            valid_mask = self.validate_waypoint_position(env, new_positions)
            
            # Update valid positions
            valid_env_ids = env_ids[valid_mask]
            if len(valid_env_ids) > 0:
                invalid_positions[valid_env_ids] = new_positions[valid_mask]
            
            # Keep trying for still-invalid positions
            env_ids = env_ids[~valid_mask]
            attempt += 1
        
        # If still invalid after max attempts, use fallback positions
        if len(env_ids) > 0:
            fallback_positions = self._generate_fallback_positions(env, len(env_ids), device)
            invalid_positions[env_ids] = fallback_positions
        
        return invalid_positions
        
    def _generate_candidate_positions(self, env, num_positions: int, device: torch.device) -> torch.Tensor:
        """Generate candidate waypoint positions for validation."""
        # Generate positions aligned with warehouse grid (aisles)
        # Use aisle-aligned positions for higher success rate
        
        # Sample aisle grid positions (warehouse_bounds: min_x, min_y, max_x, max_y)
        aisle_centers_x = torch.arange(
            self.warehouse_bounds[0] + self.wall_margin,
            self.warehouse_bounds[2] - self.wall_margin,
            self.aisle_width,
            device=device
        )
        aisle_centers_y = torch.arange(
            self.warehouse_bounds[1] + self.wall_margin,
            self.warehouse_bounds[3] - self.wall_margin,
            self.aisle_width,
            device=device
        )
        
        # Randomly select aisle centers
        x_indices = torch.randint(0, len(aisle_centers_x), (num_positions,), device=device)
        y_indices = torch.randint(0, len(aisle_centers_y), (num_positions,), device=device)
        
        positions = torch.zeros((num_positions, 2), device=device)
        positions[:, 0] = aisle_centers_x[x_indices]
        positions[:, 1] = aisle_centers_y[y_indices]
        
        # Add small random offset within aisle (±0.5m)
        offset = (torch.rand(num_positions, 2, device=device) - 0.5) * 1.0
        positions += offset
        
        return positions
        
    def _generate_fallback_positions(self, env, num_positions: int, device: torch.device) -> torch.Tensor:
        """Generate guaranteed valid fallback positions."""
        # Use predefined safe positions in the warehouse
        safe_positions = torch.tensor([
            [5.0, 5.0],    # Safe aisle positions
            [-5.0, 5.0],
            [5.0, -5.0],
            [-5.0, -5.0],
            [10.0, 0.0],
            [-10.0, 0.0],
            [0.0, 10.0],
            [0.0, -10.0],
        ], device=device)
        
        # Repeat safe positions if needed
        if num_positions > len(safe_positions):
            repeats = (num_positions // len(safe_positions)) + 1
            safe_positions = safe_positions.repeat(repeats, 1)[:num_positions]
        else:
            safe_positions = safe_positions[:num_positions]
            
        return safe_positions


# Global validator instance
_waypoint_validator = None

def get_waypoint_validator() -> WarehouseWaypointValidator:
    """Get or create the global waypoint validator instance."""
    global _waypoint_validator
    if _waypoint_validator is None:
        _waypoint_validator = WarehouseWaypointValidator()
    return _waypoint_validator

def validate_waypoint_position(env, position: torch.Tensor, warehouse_bounds: tuple = WAREHOUSE_NAVIGABLE_BOUNDS) -> torch.Tensor:
    """
    Enhanced waypoint validation using the robust validator system.
    """
    validator = get_waypoint_validator()
    return validator.validate_waypoint_position(env, position)


def height_scan_warehouse(env, sensor_cfg: SceneEntityCfg):
    """
    Height scanning for warehouse obstacle awareness.
    """
    try:
        # Extract the used quantities
        sensor = env.scene.sensors[sensor_cfg.name]
        
        # Height scan: height = sensor_height - hit_point_z - robot_base_height
        # Note: 0.26878 is distance between sensor and robot's base
        robot_base_height = 0.26878
        height_scan = sensor.data.pos_w[:, 2].unsqueeze(1) - \
                    sensor.data.ray_hits_w[..., 2] - robot_base_height
        
        return height_scan
        
    except (KeyError, AttributeError):
        # Fallback if height scanner not available
        # Return simulated height data based on robot position
        robot_pos = env.scene["robot"].data.root_state_w[:, :2]
        
        # Simulate basic height variation (flat warehouse floor with small variations)
        height_variation = torch.sin(robot_pos[:, 0] * 0.1) * torch.cos(robot_pos[:, 1] * 0.1) * 0.05
        
        # Return height scan with some variation
        num_rays = 64  # Typical LiDAR resolution
        return torch.ones(env.num_envs, num_rays, device=env.device) * height_variation.unsqueeze(1)


def dynamic_waypoint_command(env, command_name: str, num_waypoints: int = 4, 
                       resampling_time_range: tuple = (1000.0, 1000.0)):
    """
    Dynamic waypoint command system with validation.
    """
    # Get current command to check if we need new waypoints
    try:
        current_command = env.command_manager.get_command(command_name)
        # Check if we've reached current waypoint sequence
        if hasattr(env, '_waypoint_sequence_completed'):
            if env._waypoint_sequence_completed:
                # Generate new waypoint sequence
                return generate_valid_waypoint_sequence(env, num_waypoints)
        else:
            return current_command
    except:
        # Initialize waypoint system
        return generate_valid_waypoint_sequence(env, num_waypoints)


def generate_valid_waypoint_sequence(env, num_waypoints: int):
    """
    Generate a sequence of valid waypoints for warehouse navigation.
    Enhanced with robust validation loops inspired by RLRoverLab's approach.
    """
    # Use centralized warehouse bounds
    warehouse_bounds = WAREHOUSE_NAVIGABLE_BOUNDS  # (min_x, min_y, max_x, max_y) with safety margin

    # Define logical waypoint patterns for warehouse operations
    patterns = [
        # Reception area pattern
        [(15.0, 15.0), (10.0, 15.0), (5.0, 15.0), (0.0, 15.0)],

        # Storage area pattern
        [(0.0, 0.0), (-10.0, 0.0), (-10.0, -10.0), (0.0, -10.0)],

        # Shipping area pattern
        [(-15.0, -15.0), (-10.0, -15.0), (-5.0, -15.0), (0.0, -15.0)],

        # Diagonal pattern for full warehouse coverage
        [(15.0, 10.0), (10.0, 5.0), (5.0, 0.0), (0.0, -5.0)]
    ]

    # Select a pattern (could be curriculum-based)
    pattern_idx = torch.randint(0, len(patterns), (1,)).item()
    selected_pattern = patterns[int(pattern_idx)]

    # Enhanced validation with resampling loops (RLRoverLab style)
    validator = get_waypoint_validator()
    valid_waypoints = []

    # Process each waypoint with validation loop
    for i, (x, y) in enumerate(selected_pattern):
        pos = torch.tensor([[x, y]], device=env.device)
        env_ids = torch.tensor([0], device=env.device)  # Single environment

        # Use validator's resampling loop for guaranteed valid positions
        valid_pos = validator.resample_invalid_waypoints(env, pos, env_ids)

        valid_x = valid_pos[0, 0].item()
        valid_y = valid_pos[0, 1].item()
        valid_waypoints.append([valid_x, valid_y])

    # Create command tensor: [x, y, z, heading] for each waypoint
    # z=0 for floor level, heading=0 (no specific heading required)
    waypoint_commands = []
    for x, y in valid_waypoints:
        waypoint_commands.append([x, y, 0.0, 0.0])  # x, y, z, heading

    # Store sequence tracking
    env._current_waypoint_index = 0
    env._total_waypoints = len(waypoint_commands)
    env._waypoint_sequence_completed = False

    return torch.tensor(waypoint_commands, device=env.device).unsqueeze(0)


def get_current_waypoint_info(env):
    """
    Get information about current waypoint for visualization and feedback.
    """
    if not hasattr(env, '_current_waypoint_index'):
        return {"index": 0, "total": 4, "progress": 0.0}
    
    current_idx = env._current_waypoint_index
    total_waypoints = getattr(env, '_total_waypoints', 4)
    progress = current_idx / max(1, total_waypoints - 1)
    
    return {
        "index": current_idx,
        "total": total_waypoints, 
        "progress": progress,
        "completed": env._waypoint_sequence_completed
    }


def advance_waypoint_sequence(env):
    """
    Advance to next waypoint in sequence.
    """
    if not hasattr(env, '_current_waypoint_index'):
        return False
        
    current_idx = env._current_waypoint_index
    total_waypoints = env._total_waypoints
    
    if current_idx < total_waypoints - 1:
        env._current_waypoint_index = current_idx + 1
        return True
    else:
        # Mark sequence as completed
        env._waypoint_sequence_completed = True
        return False


def distance_to_current_waypoint(env, command_name: str):
    """
    Calculate distance to current active waypoint.
    """
    try:
        command = env.command_manager.get_command(command_name)
        if command.dim() == 0:  # No active waypoints
            return torch.ones(env.num_envs, device=env.device) * 999.0
            
        # Get current waypoint position
        current_idx = getattr(env, '_current_waypoint_index', 0)
        if current_idx < command.shape[0]:
            current_waypoint = command[current_idx:current_idx+1, :2]
            robot_pos = env.scene["robot"].data.root_state_w[:, :2]
            distance = torch.norm(robot_pos - current_waypoint, p=2, dim=-1)
            return distance
        else:
            return torch.ones(env.num_envs, device=env.device) * 999.0
            
    except:
        return torch.ones(env.num_envs, device=env.device) * 999.0


@configclass
class EnhancedWaypointMarkerCfg:
    """Configuration for enhanced waypoint marker system."""
    
    # Single dynamic waypoint marker (replaces multiple colored cylinders)
    active_waypoint = AssetBaseCfg(
        prim_path="/World/Waypoints/ActiveTarget",
        spawn=sim_utils.SphereCfg(
            radius=0.8,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.2, 0.0),  # Orange-red for visibility
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0))  # Floating above ground
    )
    
    # Progress indicator ring around active waypoint
    progress_ring = AssetBaseCfg(
        prim_path="/World/Waypoints/ProgressRing", 
        spawn=sim_utils.CylinderCfg(
            radius=1.2,
            height=0.1,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 1.0, 0.0),  # Green
                metallic=0.8,
                roughness=0.2
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5))
    )
    
    # Completed waypoint markers (subtle indication)
    completed_waypoint = AssetBaseCfg(
        prim_path="/World/Waypoints/CompletedMarker",
        spawn=sim_utils.CylinderCfg(
            radius=0.3,
            height=0.1,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.3, 0.3, 0.3),  # Gray
                metallic=0.1,
                roughness=0.8
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.05))  # Almost flat
    )