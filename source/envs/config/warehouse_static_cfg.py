# source/envs/config/warehouse_static_cfg.py

import os
from pathlib import Path
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.sim import GroundPlaneCfg, DomeLightCfg
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg
from isaaclab.sensors import patterns
from isaaclab.terrains import TerrainImporterCfg

# Get absolute path to project root (WarehouseBenchmark/)
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # Go up from source/envs/config/ to project root
ASSETS_DIR = PROJECT_ROOT / "assets"

@configclass
class WarehouseSceneCfg(InteractiveSceneCfg):
    """Configuration for the warehouse scene with Blender USD assets.

    Warehouse bounds: X=[-26.3, 5.5] (31.8m), Y=[-23.4, 30.6] (54m)
    Center: X=-10.4, Y=3.6
    Navigable area: X=[-24.3, 3.5] (27.8m), Y=[-21.4, 28.6] (50m)
    """

    # Note: Warehouse USD already contains floor geometry, so no additional ground plane needed
    # This prevents origin mismatches and ensures proper collision detection

    # Dome light for visibility
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=1500.0,
            color=(1.0, 1.0, 1.0),
        ),
    )

    # Warehouse environment - standard warehouse
    warehouse = AssetBaseCfg(
        prim_path="/World/warehouse",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(ASSETS_DIR / "warehouse" / "full_warehouse.usd"),
            visible=True,
            scale=(1.0, 1.0, 1.0),
            # Enable collision on USD meshes for obstacles (walls, shelves)
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))
    )

    # 3. The Robot (Clearpath Jackal - differential drive robot)
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(ASSETS_DIR / "robots" / "Clearpath" / "Jackal" / "jackal.usd"),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.5,
                angular_damping=0.5,
                max_linear_velocity=10.0,
                max_angular_velocity=10.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
                fix_root_link=False,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.3),  # Jackal spawn height
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "diff_drive": ImplicitActuatorCfg(
                joint_names_expr=[
                    "front_left_wheel_joint",
                    "front_right_wheel_joint",
                    "rear_left_wheel_joint",
                    "rear_right_wheel_joint"
                ],
                effort_limit_sim=400.0,
                velocity_limit_sim=20.0,
                stiffness=0.0,
                damping=100.0,
            ),
        },
    )

    # 4. Contact Sensor (Separate Entity)
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=False,
    )

    # 5. LiDAR sensor for obstacle detection and navigation
    lidar = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot",  # Attach to robot root
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.2)),  # Mount 20cm above base
        ray_alignment="yaw",  # Rays align with robot's heading
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,  # Single horizontal plane (2D LiDAR)
            vertical_fov_range=(0.0, 0.0),  # Horizontal scan only
            horizontal_fov_range=(-180.0, 180.0),  # Full 360° coverage
            horizontal_res=4.0  # 90 rays (every 4 degrees) - FASTER simulation
        ),
        debug_vis=False,
        mesh_prim_paths=["/World/warehouse"],  # Scan warehouse environment
        max_distance=10.0,  # 10m range
        drift_range=(-0.0, 0.0),
    )




