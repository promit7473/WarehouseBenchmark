"""
Warehouse Terrain Configurations

This module provides different terrain configurations for warehouse environments.
"""

import os
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.sim import GroundPlaneCfg, DomeLightCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass


@configclass
class BasicWarehouseSceneCfg(InteractiveSceneCfg):
    """Basic warehouse scene with ground plane and lighting."""

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=GroundPlaneCfg(size=(100.0, 100.0))
    )

    # Lighting
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=DomeLightCfg(intensity=3000.0, color=(0.9, 0.9, 0.9))
    )


@configclass
class WarehouseWithObstaclesSceneCfg(BasicWarehouseSceneCfg):
    """Warehouse scene with static obstacles (crates, pallets)."""

    # Warehouse obstacles
    crate_1 = AssetBaseCfg(
        prim_path="/World/Obstacles/Crate1",
        spawn=sim_utils.CuboidCfg(
            size=(2.0, 2.0, 1.5),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.4, 0.2)),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.6),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(5.0, 5.0, 0.75))
    )

    crate_2 = AssetBaseCfg(
        prim_path="/World/Obstacles/Crate2",
        spawn=sim_utils.CuboidCfg(
            size=(1.5, 3.0, 1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.6, 0.8)),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.6),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-8.0, 3.0, 0.5))
    )

    pallet_1 = AssetBaseCfg(
        prim_path="/World/Obstacles/Pallet1",
        spawn=sim_utils.CuboidCfg(
            size=(1.2, 0.8, 0.15),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.7, 0.5)),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.7),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(3.0, -6.0, 0.075))
    )


@configclass
class WarehouseWithShelvesSceneCfg(WarehouseWithObstaclesSceneCfg):
    """Warehouse scene with shelves and storage units."""

    # Warehouse shelves (simplified as tall cuboids)
    shelf_1 = AssetBaseCfg(
        prim_path="/World/Shelves/Shelf1",
        spawn=sim_utils.CuboidCfg(
            size=(0.3, 2.0, 2.5),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.6)),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.6),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(10.0, 0.0, 1.25))
    )

    shelf_2 = AssetBaseCfg(
        prim_path="/World/Shelves/Shelf2",
        spawn=sim_utils.CuboidCfg(
            size=(0.3, 2.0, 2.5),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.6)),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.6),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-10.0, 0.0, 1.25))
    )


@configclass
class SmallWarehouseSceneCfg(BasicWarehouseSceneCfg):
    """Small warehouse scene with basic shelving and obstacles."""

    # Load small warehouse USD file
    small_warehouse = AssetBaseCfg(
        prim_path="/World/warehouse",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(os.path.dirname(__file__), "small_warehouse", "small_warehouse.usd"),
            visible=True,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))
    )


@configclass
class MediumWarehouseSceneCfg(BasicWarehouseSceneCfg):
    """Medium warehouse scene with extensive shelving and multiple aisles."""

    # Load medium warehouse USD file
    medium_warehouse = AssetBaseCfg(
        prim_path="/World/warehouse",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(os.path.dirname(__file__), "medium_warehouse", "medium_warehouse.usd"),
            visible=True,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))
    )


@configclass
class LargeWarehouseSceneCfg(BasicWarehouseSceneCfg):
    """Large warehouse scene with complex layout and multiple zones."""

    # Load large warehouse USD file
    large_warehouse = AssetBaseCfg(
        prim_path="/World/warehouse",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(os.path.dirname(__file__), "large_warehouse", "large_warehouse.usd"),
            visible=True,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))
    )


@configclass
class RealisticWarehouseSceneCfg(WarehouseWithShelvesSceneCfg):
    """Realistic warehouse scene loaded from USD files."""

    # This would load actual USD files from Blender
    # For now, using procedural assets as placeholders

    warehouse_floor = AssetBaseCfg(
        prim_path="/World/warehouse/floor",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(os.path.dirname(__file__), "warehouse_floor.usd"),
            visible=True,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))
    )

    warehouse_walls = AssetBaseCfg(
        prim_path="/World/warehouse/walls",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(os.path.dirname(__file__), "warehouse_walls.usd"),
            visible=True,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))
    )

    warehouse_shelves = AssetBaseCfg(
        prim_path="/World/warehouse/shelves",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(os.path.dirname(__file__), "warehouse_shelves.usd"),
            visible=True,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))
    )