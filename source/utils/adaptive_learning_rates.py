# source/utils/adaptive_learning_rates.py

"""
Adaptive Learning Rate Configuration for Different Algorithms

This module provides algorithm-specific learning rates optimized for each RL method's
characteristics and training dynamics.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class LearningRateConfig:
    """Algorithm-specific learning rate configurations."""
    
    # PPO variants
    ppo_standard: float = 3.0e-4
    ppo_enhanced: float = 1.0e-3  # RSL-RL style - higher for stability
    ppo_curriculum: float = 5.0e-4  # For curriculum learning
    
    # SAC variants
    sac_standard: float = 3.0e-4
    sac_adaptive: float = 5.0e-4  # With adaptive alpha
    
    # TD3 variants  
    td3_standard: float = 1.0e-3
    td3_conservative: float = 1.0e-4  # More conservative for stability
    
    # Training mode specific
    gui_mode_multiplier: float = 2.0  # Increase LR in GUI for faster learning
    headless_multiplier: float = 1.0  # Standard LR for headless
    
    # Curriculum learning
    curriculum_start_lr: float = 5.0e-4  # Higher LR at start
    curriculum_end_lr: float = 1.0e-4   # Maintain stable LR
    
    def get_lr(self, algorithm: str, is_gui: bool = False) -> float:
        """Get algorithm-specific learning rate."""
        base_lr = getattr(self, f"{algorithm.lower()}_standard", 3.0e-4)
        
        # Apply GUI multiplier if needed
        if is_gui:
            base_lr *= self.gui_mode_multiplier
            
        return base_lr


def get_adaptive_config() -> LearningRateConfig:
    """Get adaptive learning rate configuration."""
    return LearningRateConfig()


def update_algorithm_config(config_path: str, algorithm: str, is_gui: bool = False) -> Dict[str, Any]:
    """
    Update algorithm configuration with adaptive learning rates.
    
    Args:
        config_path: Path to config file
        algorithm: Algorithm name  
        is_gui: Whether running in GUI mode
        
    Returns:
        Updated configuration dictionary
    """
    import yaml
    import os
    
    lr_config = get_adaptive_config()
    learning_rate = lr_config.get_lr(algorithm, is_gui)
    
    # Read existing config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update learning rate
    if 'agent' in config and 'learning_rate' in config['agent']:
        config['agent']['learning_rate'] = learning_rate
        print(f"[LR UPDATE] {algorithm}: {learning_rate:.2e}")
    
    # Add training mode info
    config['training_mode'] = 'gui' if is_gui else 'headless'
    
    return config


def print_lr_recommendations():
    """Print learning rate recommendations for all algorithms."""
    lr_config = get_adaptive_config()
    
    print("🎯 Learning Rate Recommendations:")
    print("=" * 50)
    
    algorithms = [
        ("PPO", lr_config.ppo_standard, "Standard PPO"),
        ("PPO_ENHANCED", lr_config.ppo_enhanced, "RSL-RL Style PPO"),
        ("SAC", lr_config.sac_standard, "Standard SAC"),
        ("TD3", lr_config.td3_standard, "Standard TD3")
    ]
    
    for algo, lr, description in algorithms:
        print(f"  {algo:12} | {lr:.2e} | {description}")
    
    print("=" * 50)
    print("💡 GUI Mode: 2x learning rates for faster convergence")
    print("💡 Use curriculum learning for progressive difficulty")