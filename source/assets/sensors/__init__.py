"""
Warehouse Sensors Module

This module provides sensor configurations for warehouse robotics,
including LiDAR, cameras, IMU, and proximity sensors.
"""

from .warehouse_sensors import (
    WarehouseSensorSuite,
    MinimalWarehouseSensors,
    AdvancedWarehouseSensors,
)

__all__ = [
    "WarehouseSensorSuite",
    "MinimalWarehouseSensors",
    "AdvancedWarehouseSensors",
]