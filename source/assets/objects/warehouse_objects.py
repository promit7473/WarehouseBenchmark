"""
Warehouse Object Configurations

This module provides configurations for warehouse objects like crates, pallets, etc.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass


# Pre-configured warehouse objects
CRATE_CFG = AssetBaseCfg(
    spawn=sim_utils.CuboidCfg(
        size=(2.0, 2.0, 1.5),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.4, 0.2)),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.6),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
    ),
    init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.75))
)

PALLET_CFG = AssetBaseCfg(
    spawn=sim_utils.CuboidCfg(
        size=(1.2, 0.8, 0.15),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.7, 0.5)),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.7),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
    ),
    init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.075))
)

SHELF_CFG = AssetBaseCfg(
    spawn=sim_utils.CuboidCfg(
        size=(0.3, 2.0, 2.5),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.6)),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.6),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
    ),
    init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 1.25))
)


@configclass
class WarehouseObjectManager:
    """Manager for creating warehouse objects with different configurations."""

    @staticmethod
    def create_crate(size=(2.0, 2.0, 1.5), color=(0.8, 0.4, 0.2), position=(0.0, 0.0, 0.75)):
        """Create a crate with custom parameters."""
        return AssetBaseCfg(
            spawn=sim_utils.CuboidCfg(
                size=size,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.6),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=position)
        )

    @staticmethod
    def create_pallet(size=(1.2, 0.8, 0.15), color=(0.7, 0.7, 0.5), position=(0.0, 0.0, 0.075)):
        """Create a pallet with custom parameters."""
        return AssetBaseCfg(
            spawn=sim_utils.CuboidCfg(
                size=size,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.7),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=position)
        )

    @staticmethod
    def create_shelf(size=(0.3, 2.0, 2.5), color=(0.5, 0.5, 0.6), position=(0.0, 0.0, 1.25)):
        """Create a shelf with custom parameters."""
        return AssetBaseCfg(
            spawn=sim_utils.CuboidCfg(
                size=size,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.6),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=position)
        )