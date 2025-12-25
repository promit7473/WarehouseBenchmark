"""
Warehouse Curriculum Module

This module provides curriculum learning functionality for progressive
warehouse navigation difficulty, inspired by RLRoverLab's curriculum system.
"""

from .warehouse_curriculum import (
    WarehouseCurriculumManager,
    BASIC_WAREHOUSE_CURRICULUM,
    ADVANCED_WAREHOUSE_CURRICULUM,
)

__all__ = [
    "WarehouseCurriculumManager",
    "BASIC_WAREHOUSE_CURRICULUM",
    "ADVANCED_WAREHOUSE_CURRICULUM",
]