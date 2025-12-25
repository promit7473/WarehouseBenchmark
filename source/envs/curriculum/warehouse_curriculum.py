"""
Warehouse Curriculum Learning

This module implements progressive difficulty curriculum for warehouse navigation,
inspired by RLRoverLab's curriculum system. The curriculum gradually increases
warehouse complexity from simple navigation to advanced multi-agent coordination.
"""

import torch
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import AssetBaseCfg
import isaaclab.sim as sim_utils


class WarehouseCurriculumManager:
    """Manages progressive curriculum learning for warehouse tasks."""

    def __init__(self):
        self.current_stage = 0
        self.stages = [
            "empty_warehouse",      # Stage 0: Basic navigation
            "static_obstacles",     # Stage 1: Fixed crates/pallets
            "narrow_aisles",        # Stage 2: Shelf navigation
            "dynamic_obstacles",    # Stage 3: Moving elements
            "multi_agent"          # Stage 4: Multiple robots
        ]

    def get_current_stage(self):
        """Get current curriculum stage."""
        return self.current_stage

    def advance_stage(self, success_rate: float, min_success_threshold: float = 0.8):
        """Advance to next stage if success criteria met."""
        if success_rate >= min_success_threshold and self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            return True
        return False

    def get_stage_config(self, stage: int):
        """Get configuration for specific curriculum stage."""
        configs = {
            0: self._empty_warehouse_config(),
            1: self._static_obstacles_config(),
            2: self._narrow_aisles_config(),
            3: self._dynamic_obstacles_config(),
            4: self._multi_agent_config()
        }
        return configs.get(stage, self._empty_warehouse_config())

    def _empty_warehouse_config(self):
        """Stage 0: Empty warehouse - basic navigation."""
        return {
            "scene": "BasicWarehouseSceneCfg",
            "max_obstacles": 0,
            "waypoint_complexity": "simple",
            "success_threshold": 0.7
        }

    def _static_obstacles_config(self):
        """Stage 1: Static obstacles - crates and pallets."""
        return {
            "scene": "WarehouseWithObstaclesSceneCfg",
            "max_obstacles": 5,
            "waypoint_complexity": "simple",
            "success_threshold": 0.75
        }

    def _narrow_aisles_config(self):
        """Stage 2: Narrow aisles - shelf navigation."""
        return {
            "scene": "WarehouseWithShelvesSceneCfg",
            "max_obstacles": 8,
            "waypoint_complexity": "aisle_navigation",
            "success_threshold": 0.8
        }

    def _dynamic_obstacles_config(self):
        """Stage 3: Dynamic obstacles - moving elements."""
        return {
            "scene": "WarehouseWithShelvesSceneCfg",
            "max_obstacles": 10,
            "waypoint_complexity": "complex",
            "dynamic_elements": True,
            "success_threshold": 0.85
        }

    def _multi_agent_config(self):
        """Stage 4: Multi-agent coordination."""
        return {
            "scene": "LargeWarehouseSceneCfg",
            "max_obstacles": 15,
            "waypoint_complexity": "complex",
            "dynamic_elements": True,
            "multi_agent": True,
            "success_threshold": 0.9
        }


# Curriculum term functions
def obstacle_density_curriculum(env, stage: int, max_obstacles: int):
    """Curriculum for obstacle density progression."""
    # Return scaling factor for obstacle density (0.0 to 1.0)
    density_scale = min(1.0, stage / 4.0)  # Scale from 0 to 1 across stages
    return {"density_scale": density_scale, "max_obstacles": int(max_obstacles * density_scale)}


def waypoint_complexity_curriculum(env, stage: int):
    """Curriculum for waypoint complexity."""
    # Return scaling factor for waypoint complexity
    complexity_scale = min(1.0, stage / 4.0)
    return {"complexity_scale": complexity_scale, "max_distance": 5.0 + complexity_scale * 15.0}


def sensor_availability_curriculum(env, stage: int):
    """Curriculum for sensor availability (returns scaling factor)."""
    # Return numerical scaling factor instead of string
    # Stage 0-1: Basic sensors only (scale 0.0-0.3)
    # Stage 2-3: Add cameras (scale 0.4-0.7)
    # Stage 4: Full sensor suite (scale 0.8-1.0)
    if stage <= 1:
        return 0.2 + stage * 0.1  # 0.2 to 0.3
    elif stage <= 3:
        return 0.4 + (stage - 2) * 0.15  # 0.4 to 0.7
    else:
        return 0.8 + (stage - 4) * 0.05  # 0.8 to 0.9


# Pre-configured curriculum instances
BASIC_WAREHOUSE_CURRICULUM = WarehouseCurriculumManager()
ADVANCED_WAREHOUSE_CURRICULUM = WarehouseCurriculumManager()