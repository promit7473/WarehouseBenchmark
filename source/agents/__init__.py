# source/agents/__init__.py
"""Agent module for warehouse navigation RL algorithms.

This module provides:
- AgentFactory: Factory class for creating RL agents (PPO, SAC, TD3)
- Neural network models for policy and value functions
- Configuration utilities for algorithm-specific settings

Usage:
    from source.agents import AgentFactory
    agent = AgentFactory.create_agent("PPO", env, config_path, device)
"""

from .factory import AgentFactory
from .models import (
    # PPO models
    PolicyPPO,
    ValuePPO,
    # SAC models
    PolicySAC,
    CriticSAC,
    # TD3 models
    ActorTD3,
    CriticTD3,
    # Vision models (optional)
    PolicyPPOVision,
    ValuePPOVision,
    # CNN components
    ConvEncoder,
    HeightmapCNN,
    VisionEncoder,
)

__all__ = [
    # Factory
    "AgentFactory",
    # PPO
    "PolicyPPO",
    "ValuePPO",
    # SAC
    "PolicySAC",
    "CriticSAC",
    # TD3
    "ActorTD3",
    "CriticTD3",
    # Vision
    "PolicyPPOVision",
    "ValuePPOVision",
    # CNN
    "ConvEncoder",
    "HeightmapCNN",
    "VisionEncoder",
]
