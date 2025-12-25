"""
Agent factory for creating RL agents based on algorithm name.

This module provides a factory pattern for creating different RL agents (PPO, SAC, TD3)
with consistent interfaces and configurations.
"""

import yaml
import torch
import gymnasium as gym
from pathlib import Path

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.agents.torch.td3 import TD3, TD3_DEFAULT_CONFIG



from skrl.memories.torch import RandomMemory
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.noises.torch import GaussianNoise

from source.agents.models import (
    PolicyPPO, ValuePPO,
    PolicySAC, CriticSAC,
    ActorTD3, CriticTD3
)


class AgentFactory:
    """Factory for creating RL agents based on algorithm name."""

    SUPPORTED_ALGORITHMS = ["PPO", "SAC", "TD3", "PPO_ENHANCED"]
    _custom_algorithms = {}  # Registry for custom algorithms

    @staticmethod
    def load_config(config_path):
        """Load YAML configuration file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        return config

    @staticmethod
    def create_ppo_agent(env, config, device):
        """Create PPO agent (on-policy)."""
        print("[INFO] Creating PPO agent...")

        # Create models
        models = {
            "policy": PolicyPPO(env.observation_space, env.action_space, device, clip_actions=True),
            "value": ValuePPO(env.observation_space, env.action_space, device)
        }

        # Create configuration
        ppo_cfg = PPO_DEFAULT_CONFIG.copy()
        ppo_cfg.update(config["agent"])

        # PPO usually handles preprocessors fine, but we configure it carefully
        if ppo_cfg.get("learning_rate_scheduler") == "KLAdaptiveRL":
            ppo_cfg["learning_rate_scheduler"] = KLAdaptiveRL
            ppo_cfg["learning_rate_scheduler_kwargs"] = {
                "kl_threshold": ppo_cfg.get("kl_threshold", 0.008)
            }

        # Enable preprocessor for PPO (it uses rollouts, so it stabilizes quickly)
        if ppo_cfg.get("state_preprocessor") == "RunningStandardScaler":
            ppo_cfg["state_preprocessor"] = RunningStandardScaler
            ppo_cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}

        if ppo_cfg.get("value_preprocessor") == "RunningStandardScaler":
            ppo_cfg["value_preprocessor"] = RunningStandardScaler
            ppo_cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}

        # Create memory
        memory_size = ppo_cfg.get("rollouts", 24)
        memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=device)

        # Create agent
        agent = PPO(
            models=models,
            memory=memory,
            cfg=ppo_cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device
        )

        print(f"[INFO] PPO agent created with {memory_size} rollouts")
        return agent

    @staticmethod
    def create_enhanced_ppo_agent(env, config, device):
        """Create Enhanced PPO agent with RSL-RL style optimizations."""
        print("[INFO] Creating Enhanced PPO agent (RSL-RL style)...")

        # Create models
        models = {
            "policy": PolicyPPO(env.observation_space, env.action_space, device, clip_actions=True),
            "value": ValuePPO(env.observation_space, env.action_space, device)
        }

        # Create configuration with enhanced defaults
        ppo_cfg = PPO_DEFAULT_CONFIG.copy()
        ppo_cfg.update(config["agent"])

        # Enhanced PPO features (RSL-RL style)
        ppo_cfg["clip_value"] = True  # Clip value function
        ppo_cfg["value_loss_scale"] = 1.0  # Scale value loss
        ppo_cfg["entropy_loss_scale"] = 0.01  # Entropy regularization

        # Adaptive KL coefficient (if available)
        if ppo_cfg.get("learning_rate_scheduler") == "KLAdaptiveRL":
            ppo_cfg["learning_rate_scheduler"] = KLAdaptiveRL
            ppo_cfg["learning_rate_scheduler_kwargs"] = {
                "kl_threshold": ppo_cfg.get("kl_threshold", 0.016)
            }

        # Enhanced preprocessing
        if ppo_cfg.get("state_preprocessor") == "RunningStandardScaler":
            ppo_cfg["state_preprocessor"] = RunningStandardScaler
            ppo_cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}

        if ppo_cfg.get("value_preprocessor") == "RunningStandardScaler":
            ppo_cfg["value_preprocessor"] = RunningStandardScaler
            ppo_cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}

        # Create memory with potentially larger rollouts for stability
        memory_size = ppo_cfg.get("rollouts", 32)  # Larger rollouts for enhanced PPO
        memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=device)

        # Create agent
        agent = PPO(
            models=models,
            memory=memory,
            cfg=ppo_cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device
        )

        print(f"[INFO] Enhanced PPO agent created with {memory_size} rollouts")
        return agent

    @staticmethod
    def create_sac_agent(env, config, device):
        """
        Create SAC agent (off-policy, stochastic).
        """
        print("[INFO] Creating SAC agent...")

        # Create models
        models = {
            "policy": PolicySAC(env.observation_space, env.action_space, device, clip_actions=True),
            "critic_1": CriticSAC(env.observation_space, env.action_space, device),
            "critic_2": CriticSAC(env.observation_space, env.action_space, device),
            "target_critic_1": CriticSAC(env.observation_space, env.action_space, device),
            "target_critic_2": CriticSAC(env.observation_space, env.action_space, device)
        }

        # Initialize target networks
        models["target_critic_1"].load_state_dict(models["critic_1"].state_dict())
        models["target_critic_2"].load_state_dict(models["critic_2"].state_dict())

        # Create configuration
        sac_cfg = SAC_DEFAULT_CONFIG.copy()
        sac_cfg.update(config["agent"])

        # === STATE PREPROCESSOR - DISABLED FOR STABILITY ===
        # RunningStandardScaler can cause NaN issues if not properly initialized
        # Disable for now to ensure stable training
        sac_cfg["state_preprocessor"] = False
        sac_cfg["state_preprocessor_kwargs"] = {}
        print("[INFO] State preprocessor DISABLED for SAC (stability fix)")

        # Create memory
        memory_size = sac_cfg.get("memory_size", 100000)
        memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=device)

        # Create agent
        agent = SAC(
            models=models,
            memory=memory,
            cfg=sac_cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device
        )

        print(f"[INFO] SAC agent created with replay buffer size {memory_size}")
        return agent

    @staticmethod
    def create_td3_agent(env, config, device):
        """
        Create TD3 agent (off-policy, deterministic).
        """
        print("[INFO] Creating TD3 agent...")

        # Create models
        models = {
            "policy": ActorTD3(env.observation_space, env.action_space, device, clip_actions=True),
            "target_policy": ActorTD3(env.observation_space, env.action_space, device, clip_actions=True),
            "critic_1": CriticTD3(env.observation_space, env.action_space, device),
            "critic_2": CriticTD3(env.observation_space, env.action_space, device),
            "target_critic_1": CriticTD3(env.observation_space, env.action_space, device),
            "target_critic_2": CriticTD3(env.observation_space, env.action_space, device)
        }

        # Initialize targets
        models["target_policy"].load_state_dict(models["policy"].state_dict())
        models["target_critic_1"].load_state_dict(models["critic_1"].state_dict())
        models["target_critic_2"].load_state_dict(models["critic_2"].state_dict())

        # Create configuration
        td3_cfg = TD3_DEFAULT_CONFIG.copy()
        td3_cfg.update(config["agent"])

        # Handle exploration noise
        if "exploration" in td3_cfg and td3_cfg["exploration"].get("noise_type") == "gaussian":
            initial_scale = td3_cfg["exploration"].get("initial_scale", 0.1)
            td3_cfg["exploration"]["noise"] = GaussianNoise(mean=0, std=initial_scale, device=device)

        # === STATE PREPROCESSOR - DISABLED FOR STABILITY ===
        # RunningStandardScaler can cause NaN issues if not properly initialized
        # Disable for now to ensure stable training
        td3_cfg["state_preprocessor"] = False
        td3_cfg["state_preprocessor_kwargs"] = {}
        print("[INFO] State preprocessor DISABLED for TD3 (stability fix)")

        # Create memory
        memory_size = td3_cfg.get("memory_size", 100000)
        memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=device)

        # Create agent
        agent = TD3(
            models=models,
            memory=memory,
            cfg=td3_cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device
        )

        # Set action clipping attributes for TD3 (required by SKRL)
        agent.clip_actions_min = torch.tensor(-1.0, device=device)
        agent.clip_actions_max = torch.tensor(1.0, device=device)

        print(f"[INFO] TD3 agent created with replay buffer size {memory_size}")
        return agent





    @classmethod
    def create_agent(cls, algorithm, env, config_path, device):
        """Main factory method."""
        algorithm = algorithm.upper()

        if algorithm not in cls.SUPPORTED_ALGORITHMS:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        config = cls.load_config(config_path)

        # Check if it's a custom algorithm
        if algorithm in cls._custom_algorithms:
            create_func = cls._custom_algorithms[algorithm]['create_func']
            return create_func(env, config, device)

        # Built-in algorithms
        if algorithm == "PPO":
            return cls.create_ppo_agent(env, config, device)
        elif algorithm == "PPO_ENHANCED":
            return cls.create_enhanced_ppo_agent(env, config, device)
        elif algorithm == "SAC":
            return cls.create_sac_agent(env, config, device)
        elif algorithm == "TD3":
            return cls.create_td3_agent(env, config, device)
        else:
            raise ValueError(f"Algorithm {algorithm} not implemented")

    @classmethod
    def get_config_path(cls, algorithm):
        # Check if custom algorithm has its own config path function
        algorithm = algorithm.upper()
        if algorithm in cls._custom_algorithms and cls._custom_algorithms[algorithm]['config_path_func']:
            return cls._custom_algorithms[algorithm]['config_path_func'](algorithm)

        # Default path using Path for better cross-platform compatibility
        from pathlib import Path
        source_dir = Path(__file__).parent.parent
        config_path = source_dir.parent / "configs" / f"{algorithm.lower()}_warehouse.yaml"
        return str(config_path)

    @classmethod
    def register_algorithm(cls, name, create_func, config_path_func=None):
        """Register a custom algorithm.

        Args:
            name: Algorithm name (uppercase)
            create_func: Function that takes (env, config, device) and returns agent
            config_path_func: Optional function to get config path for algorithm
        """
        cls._custom_algorithms[name.upper()] = {
            'create_func': create_func,
            'config_path_func': config_path_func
        }
        if name.upper() not in cls.SUPPORTED_ALGORITHMS:
            cls.SUPPORTED_ALGORITHMS.append(name.upper())

    @staticmethod
    def list_available_algorithms():
        return AgentFactory.SUPPORTED_ALGORITHMS.copy()


if __name__ == "__main__":
    # Print factory info when run directly
    print("Agent Factory Loaded Successfully")