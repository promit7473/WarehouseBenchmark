# source/envs/mdp/observations.py
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp_utils
import torch
import logging
from source.envs.utils import quaternion_to_yaw

# Setup logger
logger = logging.getLogger(__name__)

# Realistic localization noise parameters (simulates GPS/SLAM uncertainty)
WAYPOINT_POSITION_NOISE_STD = 0.15  # 15cm standard deviation in position
WAYPOINT_HEADING_NOISE_STD = 0.05   # ~3 degrees standard deviation in heading


def noisy_waypoint_command(env, command_name: str):
    """Get waypoint command with realistic localization noise.

    Real robots don't have perfect GPS/localization. This adds realistic noise
    to simulate GPS uncertainty, SLAM drift, and map-matching errors.

    Args:
        env: Environment instance
        command_name: Name of the command to retrieve

    Returns:
        Noisy waypoint command tensor (num_envs, command_dim)
    """
    # Get perfect command from command manager
    perfect_command = env.command_manager.get_command(command_name)

    # Add Gaussian noise to position (x, y, z if present)
    # Simulates GPS/SLAM localization uncertainty (~15cm std dev is realistic)
    num_position_dims = min(3, perfect_command.shape[1] - 1)  # Up to 3 position dims, leaving 1 for heading
    position_noise = torch.randn(
        env.num_envs, num_position_dims,
        device=env.device,
        dtype=perfect_command.dtype
    ) * WAYPOINT_POSITION_NOISE_STD

    # Add noise to heading if command has heading component (last dimension)
    if perfect_command.shape[1] > num_position_dims:
        heading_noise = torch.randn(
            env.num_envs, 1,
            device=env.device,
            dtype=perfect_command.dtype
        ) * WAYPOINT_HEADING_NOISE_STD

        # Apply noise to full command
        noise = torch.cat([position_noise, heading_noise], dim=1)
    else:
        # Only position noise
        noise = position_noise

    noisy_command = perfect_command + noise

    return noisy_command


def obstacle_proximity_observation(env, sensor_cfg, max_distance: float = 5.0):
    """Observe proximity to obstacles using contact sensors."""
    # This would integrate with warehouse proximity sensors
    # Based on the warehouse logistics tutorial
    robot_pos = env.scene["robot"].data.root_state_w[:, :2]

    # Placeholder for obstacle proximity detection
    # Would use actual sensor data from warehouse assets
    proximity_obs = torch.zeros(env.num_envs, 8, device=env.device)  # 8 proximity sectors

    return proximity_obs


def aisle_alignment_observation(env, command_name: str):
    """Observe alignment with warehouse aisles."""
    robot_pos = env.scene["robot"].data.root_state_w[:, :2]
    robot_yaw = quaternion_to_yaw(env.scene["robot"].data.root_state_w)  # Extract yaw from quaternion
    waypoint = env.command_manager.get_command(command_name)[:, :2]

    # Calculate warehouse grid alignment
    aisle_width = 3.5  # meters between aisle centers

    # Distance to nearest grid lines (aisle centers)
    grid_x = torch.round(robot_pos[:, 0] / aisle_width) * aisle_width
    grid_y = torch.round(robot_pos[:, 1] / aisle_width) * aisle_width

    dist_to_aisle_x = torch.abs(robot_pos[:, 0] - grid_x)
    dist_to_aisle_y = torch.abs(robot_pos[:, 1] - grid_y)

    # Aisle alignment: 1.0 when in aisle center, 0.0 when between aisles
    aisle_alignment = torch.exp(-torch.minimum(dist_to_aisle_x, dist_to_aisle_y) / 0.3)

    # Optimal heading: angle towards waypoint
    optimal_heading = torch.atan2(waypoint[:, 1] - robot_pos[:, 1],
                                 waypoint[:, 0] - robot_pos[:, 0])

    # Heading alignment: how well current heading matches optimal
    heading_diff = torch.abs(torch.atan2(torch.sin(robot_yaw - optimal_heading),
                                        torch.cos(robot_yaw - optimal_heading)))
    heading_alignment = torch.exp(-heading_diff / (torch.pi / 4))  # Within 45 degrees

    alignment_obs = torch.stack([aisle_alignment, heading_alignment], dim=1)

    return alignment_obs


def lidar_observations(env, sensor_cfg: SceneEntityCfg):
    """Process LiDAR sensor data for obstacle detection.

    Returns normalized distance measurements from 2D LiDAR scan.
    Real warehouse navigation requires actual sensor perception.

    Args:
        env: Environment instance
        sensor_cfg: Sensor configuration

    Returns:
        Tensor of normalized distances (num_envs, num_rays) in range [0, 1]
        where 0 = obstacle at sensor, 1 = no obstacle within max_range
    """
    try:
        # Get LiDAR sensor data from scene
        lidar_sensor = env.scene["lidar"]

        # RayCaster returns hit information: [distance, normal_x, normal_y] or similar
        # When no hit, values are inf. We extract distance (first column)
        ray_hits = lidar_sensor.data.ray_hits_w
        distances = ray_hits[:, :, 0]  # Distance is in the first column

        # Get max range from sensor config (10.0m as configured)
        max_distance = 10.0

        # Clamp distances to valid range [0, max_distance]
        # Invalid rays (didn't hit anything) return large values, clamp them to max
        distances = torch.clamp(distances, 0.0, max_distance)

        # Normalize to [0, 1] range for neural network
        # 0.0 = obstacle right at sensor (0m)
        # 1.0 = no obstacle within range (10m)
        normalized_distances = distances / max_distance

        return normalized_distances

    except (KeyError, AttributeError) as e:
        # Fallback if LiDAR sensor not available (shouldn't happen in real training)
        # This is only for debugging/testing without proper sensor setup
        logger.warning(f"LiDAR sensor not available, using dummy data. Error: {e}. "
                      "This may indicate improper sensor configuration.")

        # Return dummy data with same shape as expected (180 rays for our config)
        num_rays = 180  # horizontal_fov_range / horizontal_res = 360 / 2 = 180
        dummy_distances = torch.ones(env.num_envs, num_rays, device=env.device) * 0.8
        return dummy_distances
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"Unexpected error accessing LiDAR sensor: {e}")
        raise  # Re-raise to make failures visible


def height_scan_warehouse(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Height scanning for warehouse obstacle awareness.
    
    Enhanced version based on RLRoverLab's approach for warehouse environments.
    """
    try:
        # Extract the used quantities
        sensor = env.scene.sensors[sensor_cfg.name]
        
        # Height scan: height = sensor_height - hit_point_z - robot_base_height
        # Note: 0.26878 is the distance between sensor and robot's base
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


def camera_rgb_observations(env, sensor_cfg: SceneEntityCfg):
    """Process RGB camera data for visual navigation."""
    try:
        # Get RGB camera sensor data
        camera_sensor = env.scene["camera_rgb"]
        # Get RGB image data
        rgb_data = camera_sensor.data.output["rgb"]
        # rgb_data shape: (num_envs, height, width, 3)
        # Convert from uint8 to float for PyTorch operations
        rgb_data = rgb_data.float()
        # For now, return mean RGB values as simple features
        # Could be enhanced with CNN feature extraction
        visual_features = rgb_data.mean(dim=(1, 2))  # Average over height and width
        # Normalize to 0-1 range (data is already 0-255, so divide by 255)
        visual_features = visual_features / 255.0
        return visual_features
    except (KeyError, AttributeError):
        # Fallback if camera not available
        # Return zero features
        return torch.zeros(env.num_envs, 3, device=env.device)


def camera_depth_observations(env, sensor_cfg: SceneEntityCfg):
    """Process depth camera data for heightmap generation."""
    try:
        # Get depth camera sensor data
        camera_sensor = env.scene["camera_depth"]
        # Get depth image data
        depth_data = camera_sensor.data.output["depth"]
        # depth_data shape: (num_envs, height, width, 1)
        # Convert to float if needed
        depth_data = depth_data.float()
        # For heightmap, we can downsample and flatten
        # Simple approach: average depth values in grid cells
        height, width = depth_data.shape[1:3]
        # Create a simple 8x8 heightmap
        grid_size = 8
        cell_h = height // grid_size
        cell_w = width // grid_size

        heightmap_features = []
        for i in range(grid_size):
            for j in range(grid_size):
                cell = depth_data[:, i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w, 0]
                heightmap_features.append(cell.mean(dim=(1, 2)))

        heightmap = torch.stack(heightmap_features, dim=1)
        # Normalize depth (assuming max range is 50.0 as set in camera config)
        heightmap = torch.clamp(heightmap, 0.0, 50.0) / 50.0
        return heightmap
    except (KeyError, AttributeError):
        # Fallback if camera not available
        # Return zero features
        return torch.zeros(env.num_envs, 64, device=env.device)  # 8x8 = 64 features


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # Robot state
        base_lin_vel = ObsTerm(func=mdp_utils.base_lin_vel, scale=2.0)
        base_ang_vel = ObsTerm(func=mdp_utils.base_ang_vel, scale=0.25)
        projected_gravity = ObsTerm(func=mdp_utils.projected_gravity)
        last_action = ObsTerm(func=mdp_utils.last_action)

        # Waypoint navigation with REALISTIC localization noise
        # Real robots don't have perfect GPS - adds 15cm position noise + 3° heading noise
        waypoint_command = ObsTerm(
            func=noisy_waypoint_command,
            params={"command_name": "waypoint_nav"}
        )

        # LiDAR sensor observations - ENABLED for real warehouse navigation
        # 2D LiDAR with 180 rays provides 360° obstacle detection
        lidar_scan = ObsTerm(
            func=lidar_observations,
            params={"sensor_cfg": SceneEntityCfg("lidar")},
            scale=1.0  # Distances already normalized in lidar_observations function
        )

        # NOTE: Camera sensors intentionally disabled for headless training compatibility
        # To enable visual navigation: uncomment and configure camera_rgb in scene config

        # Enhanced path planning observations
        aisle_alignment = ObsTerm(
            func=aisle_alignment_observation,
            params={"command_name": "waypoint_nav"},
            scale=1.0
        )
        
        # NOTE: Height scanner intentionally disabled due to raycast pose errors in current Isaac Lab version
        # LiDAR provides sufficient obstacle detection for 2D warehouse navigation

    policy: PolicyCfg = PolicyCfg()