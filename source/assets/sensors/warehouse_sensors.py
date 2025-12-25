"""
Advanced Sensor Configurations for Warehouse Navigation

This module provides comprehensive sensor suites for warehouse robotics,
incorporating knowledge from NVIDIA Isaac Sim warehouse logistics tutorials.
"""

import os
from isaaclab.sensors import CameraCfg, RayCasterCfg, ImuCfg
from isaaclab.sensors.camera.tiled_camera_cfg import TiledCameraCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.sensors import patterns


@configclass
class WarehouseSensorSuite:
    """Comprehensive sensor suite for warehouse navigation."""

    # LiDAR for 360-degree obstacle detection
    lidar_360 = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/lidar_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.1)),
        ray_alignment="yaw",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=64,
            vertical_fov_range=(-15.0, 15.0),
            horizontal_fov_range=(-180.0, 180.0)
        ),
        debug_vis=False,
        mesh_prim_paths=["/World/warehouse/*"],  # Only detect warehouse assets
        max_distance=10.0,
        update_period=0.04  # 25Hz
    )

    # RGB-D Camera for item detection and navigation
    rgb_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/camera_link",
        offset=CameraCfg.OffsetCfg(pos=(0.2, 0.0, 0.3), rot=(0.0, 0.0, 0.0, 1.0)),
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            vertical_aperture=15.2908,
            clipping_range=(0.1, 100.0)
        ),
        width=640,
        height=480,
        update_period=0.04  # 25Hz
    )

    # IMU for stability monitoring
    imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/imu_link",
        offset=ImuCfg.OffsetCfg(pos=(0.0, 0.0, 0.1)),
        update_period=0.04  # 25Hz
    )

    # Proximity sensors for close-range obstacle detection
    proximity_front = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/proximity_front",
        offset=RayCasterCfg.OffsetCfg(pos=(0.3, 0.0, 0.1)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.5, 0.5)),
        debug_vis=False,
        mesh_prim_paths=["/World/warehouse/*"],
        max_distance=2.0,
        update_period=0.04
    )

    proximity_left = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/proximity_left",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.3, 0.1), rot=(0.0, 0.0, 0.707, 0.707)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.5, 0.5)),
        debug_vis=False,
        mesh_prim_paths=["/World/warehouse/*"],
        max_distance=2.0,
        update_period=0.04
    )

    proximity_right = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/proximity_right",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, -0.3, 0.1), rot=(0.0, 0.0, -0.707, 0.707)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.5, 0.5)),
        debug_vis=False,
        mesh_prim_paths=["/World/warehouse/*"],
        max_distance=2.0,
        update_period=0.04
    )

    # Wide-angle camera for warehouse monitoring
    wide_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/wide_camera",
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,  # Wider FOV
            focus_distance=400.0,
            horizontal_aperture=30.0,  # Wider aperture for wider FOV
            vertical_aperture=20.0,
            clipping_range=(0.1, 100.0)
        ),
        width=640,
        height=360,  # Wider aspect ratio
        update_period=0.1  # 10Hz
    )


@configclass
class MinimalWarehouseSensors:
    """Minimal sensor suite for basic warehouse navigation."""

    # Basic LiDAR
    lidar = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.1)),
        ray_alignment="yaw",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=32,
            vertical_fov_range=(-10.0, 10.0),
            horizontal_fov_range=(-180.0, 180.0)
        ),
        debug_vis=False,
        mesh_prim_paths=["/World/warehouse/*"],
        max_distance=8.0,
        update_period=0.04
    )

    # Basic RGB camera
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/camera_link",
        offset=CameraCfg.OffsetCfg(pos=(0.2, 0.0, 0.3)),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            vertical_aperture=15.2908,
            clipping_range=(0.1, 50.0)
        ),
        width=320,
        height=240,
        update_period=0.04
    )


@configclass
class AdvancedWarehouseSensors:
    """Advanced sensor suite for complex warehouse tasks."""

    # All sensors from comprehensive suite plus additional ones
    lidar_360 = WarehouseSensorSuite.lidar_360
    rgb_camera = WarehouseSensorSuite.rgb_camera
    imu = WarehouseSensorSuite.imu
    proximity_front = WarehouseSensorSuite.proximity_front
    proximity_left = WarehouseSensorSuite.proximity_left
    proximity_right = WarehouseSensorSuite.proximity_right
    wide_camera = WarehouseSensorSuite.wide_camera

    # Additional sensors for advanced tasks
    depth_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/depth_camera_link",
        offset=CameraCfg.OffsetCfg(pos=(0.2, 0.0, 0.3)),
        data_types=["depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            vertical_aperture=15.2908,
            clipping_range=(0.1, 20.0)
        ),
        width=640,
        height=480,
        update_period=0.04
    )

    # Semantic segmentation camera for item recognition
    semantic_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/semantic_camera_link",
        offset=CameraCfg.OffsetCfg(pos=(0.2, 0.0, 0.3)),
        data_types=["semantic_segmentation"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            vertical_aperture=15.2908,
            clipping_range=(0.1, 50.0)
        ),
        width=640,
        height=480,
        update_period=0.1  # Lower frequency for semantic processing
    )