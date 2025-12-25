"""Warehouse environment registration and configuration exports."""

import os
import gymnasium as gym

from .warehouse_env import WarehouseEnvCfg

# Register the environment with proper kwargs pattern (like RLRoverLab)
# This allows dynamic configuration lookup via gym.spec()
gym.register(
    id="Warehouse-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": WarehouseEnvCfg,
        "skrl_cfgs": {
            "PPO": os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                              "configs", "ppo_warehouse.yaml"),
            "PPO_ENHANCED": os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                        "configs", "ppo_enhanced_warehouse.yaml"),
            "SAC": os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                               "configs", "sac_warehouse.yaml"),
            "TD3": os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                               "configs", "td3_warehouse.yaml"),
        },
    }
)

# Export configuration class for external use
__all__ = ["WarehouseEnvCfg"]