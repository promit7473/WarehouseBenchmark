"""
Warehouse Assets Module

This module provides modular configurations for all warehouse assets including:
- Robots (Jetbot configurations)
- Terrains (different warehouse layouts)
- Objects (crates, pallets, shelves)
- Sensors (LiDAR, cameras, IMU)
- Textures (shared materials)
"""

from .robots.jetbot import JETBOT_CFG
from .terrains.warehouse_terrains import (
    BasicWarehouseSceneCfg,
    WarehouseWithObstaclesSceneCfg,
    WarehouseWithShelvesSceneCfg,
    SmallWarehouseSceneCfg,
    MediumWarehouseSceneCfg,
    LargeWarehouseSceneCfg,
    RealisticWarehouseSceneCfg,
)
from .objects.warehouse_objects import (
    CRATE_CFG,
    PALLET_CFG,
    SHELF_CFG,
    WarehouseObjectManager,
)
from .sensors.warehouse_sensors import (
    WarehouseSensorSuite,
    MinimalWarehouseSensors,
    AdvancedWarehouseSensors,
)

__all__ = [
    # Robots
    "JETBOT_CFG",

    # Terrains
    "BasicWarehouseSceneCfg",
    "WarehouseWithObstaclesSceneCfg",
    "WarehouseWithShelvesSceneCfg",
    "SmallWarehouseSceneCfg",
    "MediumWarehouseSceneCfg",
    "LargeWarehouseSceneCfg",
    "RealisticWarehouseSceneCfg",

    # Objects
    "CRATE_CFG",
    "PALLET_CFG",
    "SHELF_CFG",
    "WarehouseObjectManager",

    # Sensors
    "WarehouseSensorSuite",
    "MinimalWarehouseSensors",
    "AdvancedWarehouseSensors",
]