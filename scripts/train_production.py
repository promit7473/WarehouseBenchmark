#!/usr/bin/env python3
"""
Production Training Script with Curriculum Learning

This script implements advanced training features:
- Progressive curriculum learning
- Dynamic difficulty adjustment
- Comprehensive logging and monitoring
- Production-scale training (1M+ timesteps)
"""

import argparse
import os
import time
from pathlib import Path

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.insert(0, project_root)

# 1. LAUNCH APP FIRST
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description="Production training with curriculum learning")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--algorithm", type=str, default="PPO_ENHANCED",
                    choices=["PPO", "PPO_ENHANCED", "SAC", "TD3"],
                    help="RL algorithm to use")
parser.add_argument("--curriculum", action="store_true",
                    help="Enable curriculum learning")
parser.add_argument("--max_timesteps", type=int, default=1000000,
                    help="Maximum training timesteps")
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 2. Enhanced logging setup
try:
    import wandb
    # Check if wandb has the required functions
    if hasattr(wandb, 'init') and hasattr(wandb, 'log') and hasattr(wandb, 'finish'):
        WANDB_AVAILABLE = True
    else:
        WANDB_AVAILABLE = False
        wandb = None
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

import carb
try:
    settings = carb.settings.get_settings()
    settings.set("/persistent/physics/warnOnNonOrthonormal", False)
    settings.set("/persistent/omnihydra/useSceneGraphInstancing", True)
    settings.set("/persistent/simulation/minFrameRate", 30)
    settings.set("/app/player/useFixedTimeStepping", False)
    print("[INFO] USD visualization warnings suppressed")
except Exception as e:
    print(f"[WARNING] Could not suppress warnings: {e}")

import torch
import gymnasium as gym
from skrl.envs.wrappers.torch import wrap_env
from skrl.trainers.torch import SequentialTrainer

import source.envs
from source.envs.warehouse_env import WarehouseEnvCfg
from source.agents.factory import AgentFactory


class CurriculumManager:
    """Manages progressive difficulty increase during training."""

    def __init__(self):
        self.stages = [
            {"name": "Beginner", "progress": 0.0, "max_distance": 8.0, "timestep_range": (0, 200000)},
            {"name": "Intermediate", "progress": 0.2, "max_distance": 12.0, "timestep_range": (200000, 400000)},
            {"name": "Advanced", "progress": 0.4, "max_distance": 16.0, "timestep_range": (400000, 600000)},
            {"name": "Expert", "progress": 0.6, "max_distance": 20.0, "timestep_range": (600000, 800000)},
            {"name": "Master", "progress": 0.8, "max_distance": 25.0, "timestep_range": (800000, 1000000)},
        ]
        self.current_stage = 0

    def get_current_stage(self, timestep):
        """Get current curriculum stage based on timestep."""
        for i, stage in enumerate(self.stages):
            start, end = stage["timestep_range"]
            if start <= timestep < end:
                return i, stage
        return len(self.stages) - 1, self.stages[-1]

    def update_environment(self, env, timestep):
        """Update environment difficulty based on current timestep."""
        stage_idx, stage = self.get_current_stage(timestep)

        if stage_idx != self.current_stage:
            self.current_stage = stage_idx
            print(f"[CURRICULUM] Entering {stage['name']} stage - Max distance: {stage['max_distance']}m")

            # Update command ranges dynamically
            if hasattr(env, 'command_manager') and "waypoint_nav" in env.command_manager._terms:
                command_term = env.command_manager._terms["waypoint_nav"]
                max_dist = stage["max_distance"]
                command_term._cfg.ranges.pos_x = (-max_dist, max_dist)
                command_term._cfg.ranges.pos_y = (-max_dist, max_dist)
                print(f"[CURRICULUM] Updated waypoint ranges to ±{max_dist}m")

        return stage


def main():
    """Production training with curriculum learning."""

    # Initialize wandb
    if WANDB_AVAILABLE:
        wandb.init(
            project="WarehouseBenchmark-Production",
            name=f"{args_cli.algorithm}_curriculum_{time.strftime('%Y%m%d_%H%M%S')}",
            config={
                "algorithm": args_cli.algorithm,
                "curriculum_enabled": args_cli.curriculum,
                "max_timesteps": args_cli.max_timesteps,
                "num_envs": 128,
            },
            tags=["production", "curriculum", args_cli.algorithm.lower(), "warehouse"]
        )

    # Load configuration
    config_path = AgentFactory.get_config_path(args_cli.algorithm)
    config = AgentFactory.load_config(config_path)
    config["trainer"]["timesteps"] = args_cli.max_timesteps

    print(f"[INFO] Production training: {args_cli.algorithm}")
    print(f"[INFO] Curriculum learning: {args_cli.curriculum}")
    print(f"[INFO] Target timesteps: {args_cli.max_timesteps:,}")

    # Initialize curriculum manager
    curriculum = CurriculumManager() if args_cli.curriculum else None

    # Create environment
    env_cfg = WarehouseEnvCfg()
    env_cfg.scene.num_envs = 32  # Production scale
    env = gym.make("Warehouse-v0", cfg=env_cfg, render_mode="rgb_array")
    env = wrap_env(env, wrapper="isaaclab")

    device = env.device
    agent = AgentFactory.create_agent(args_cli.algorithm, env, config_path, device)

    # Production training setup
    trainer = SequentialTrainer(cfg=config["trainer"], env=env, agents=agent)

    # Training loop with curriculum
    start_time = time.time()
    last_log_time = start_time

    print("[INFO] Starting production training...")

    # For curriculum learning, we need to implement custom training loop
    if args_cli.curriculum and curriculum:
        # Custom training loop with curriculum updates
        timestep = 0
        episode_count = 0

        while timestep < args_cli.max_timesteps:
            # Update curriculum
            current_stage = curriculum.update_environment(env, timestep)

            # Train for one batch
            trainer.train()

            # Periodic logging
            current_time = time.time()
            if current_time - last_log_time > 60:  # Log every minute
                elapsed = current_time - start_time
                progress = timestep / args_cli.max_timesteps

                print(f"[TRAINING] Timestep: {timestep:,}/{args_cli.max_timesteps:,} "
                      f"({progress:.1%}) | Elapsed: {elapsed/3600:.1f}h | "
                      f"Stage: {current_stage['name']}")

                if WANDB_AVAILABLE:
                    wandb.log({
                        "timestep": timestep,
                        "progress": progress,
                        "elapsed_hours": elapsed / 3600,
                        "current_stage": current_stage["name"],
                        "max_distance": current_stage["max_distance"]
                    })

                last_log_time = current_time

            timestep += config["trainer"].get("timesteps_per_train", 1)
            episode_count += 1
    else:
        # Standard training
        trainer.train()

    # Final logging
    total_time = time.time() - start_time
    print(f"[INFO] Training completed in {total_time/3600:.1f} hours")

    if WANDB_AVAILABLE:
        wandb.log({
            "training_completed": True,
            "total_time_hours": total_time / 3600,
            "final_timestep": args_cli.max_timesteps,
            "curriculum_enabled": args_cli.curriculum
        })
        wandb.finish()

    print("[INFO] Production training completed successfully!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL TRAINING ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()