"""
Evaluation script for trained RL agents.

This script loads a trained checkpoint and evaluates it on the warehouse environment,
collecting detailed metrics and generating visualizations.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Weights & Biases integration
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

# 1. LAUNCH APP FIRST
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description="Evaluate a trained RL agent.")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--checkpoint", type=str, default=None,
                   help="Path to checkpoint file (if None, finds latest)")
parser.add_argument("--algorithm", type=str, default=None,
                   choices=["PPO", "SAC", "TD3"],
                   help="Algorithm type (if None, infers from checkpoint)")
parser.add_argument("--num-episodes", type=int, default=100,
                   help="Number of episodes to evaluate (default: 100)")
parser.add_argument("--output-dir", type=str, default=None,
                   help="Output directory for results (default: auto-generated)")
parser.add_argument("--deterministic", action="store_true", default=True,
                   help="Use deterministic actions (default: True)")
args_cli = parser.parse_args()

# Validate command-line arguments
if args_cli.num_episodes < 1 or args_cli.num_episodes > 10000:
    print(f"[ERROR] --num-episodes must be between 1 and 10000, got {args_cli.num_episodes}")
    sys.exit(1)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
import numpy as np

from skrl.envs.wrappers.torch import wrap_env

import source.envs
from source.envs.warehouse_env import WarehouseEnvCfg
from source.agents.factory import AgentFactory
from source.utils.metrics import MetricsCollector
from source.utils.path_validator import validate_checkpoint_path, validate_output_path, PathValidationError


def find_latest_checkpoint():
    """
    Find the latest checkpoint in the runs directory.

    Returns:
        tuple: (checkpoint_path, algorithm)

    Raises:
        FileNotFoundError: If no checkpoints found
    """
    runs_dir = Path("runs")
    if not runs_dir.exists():
        raise FileNotFoundError("No runs directory found")

    # Find all checkpoint directories
    checkpoint_dirs = []
    for run_dir in runs_dir.iterdir():
        if run_dir.is_dir():
            checkpoints_dir = run_dir / "checkpoints"
            if checkpoints_dir.exists():
                checkpoint_dirs.append((run_dir, checkpoints_dir))

    if not checkpoint_dirs:
        raise FileNotFoundError("No checkpoint directories found in runs/")

    # Sort by modification time
    checkpoint_dirs.sort(key=lambda x: x[0].stat().st_mtime, reverse=True)
    latest_run_dir, latest_checkpoints_dir = checkpoint_dirs[0]

    # Try to find best_agent.pt first
    best_checkpoint = latest_checkpoints_dir / "best_agent.pt"
    if best_checkpoint.exists():
        checkpoint_path = best_checkpoint
    else:
        # Find latest checkpoint by number
        checkpoint_files = sorted(latest_checkpoints_dir.glob("agent_*.pt"),
                                key=lambda x: int(x.stem.split('_')[1]),
                                reverse=True)
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found in {latest_checkpoints_dir}")
        checkpoint_path = checkpoint_files[0]

    # Infer algorithm from run directory name
    run_name = latest_run_dir.name
    algorithm = None
    for algo in ["PPO", "SAC", "TD3"]:
        if algo in run_name.upper():
            algorithm = algo
            break

    if algorithm is None:
        # Default to PPO
        algorithm = "PPO"
        print(f"[WARN] Could not infer algorithm from run name, defaulting to {algorithm}")

    print(f"[INFO] Found latest checkpoint: {checkpoint_path}")
    print(f"[INFO] Inferred algorithm: {algorithm}")

    return str(checkpoint_path), algorithm


def evaluate_agent(agent, env, num_episodes, deterministic=True):
    """
    Evaluate agent on environment.

    Args:
        agent: Trained agent
        env: Wrapped environment
        num_episodes (int): Number of episodes to evaluate
        deterministic (bool): Use deterministic actions

    Returns:
        MetricsCollector: Collector with episode metrics
    """
    print(f"\n[INFO] Starting evaluation for {num_episodes} episodes...")
    print(f"[INFO] Deterministic: {deterministic}")

    # Set agent to evaluation mode
    agent.set_mode("eval")

    # Create metrics collector
    collector = MetricsCollector()

    # Reset environment
    obs, _ = env.reset()

    episode_count = 0
    step_count = 0
    episodes_data = [[] for _ in range(env.num_envs)]  # Track data per parallel env

    # For tracking episode metrics
    episode_rewards = torch.zeros(env.num_envs, device=env.device)
    episode_steps = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)

    print(f"[INFO] Running evaluation with {env.num_envs} parallel environments...")

    while episode_count < num_episodes:
        # Get actions
        with torch.no_grad():
            if deterministic:
                # Use mean action (no sampling)
                actions = agent.act(obs, timestep=step_count, timesteps=999999)[0]
            else:
                # Sample from distribution
                actions = agent.act(obs, timestep=step_count, timesteps=999999)[0]

        # Step environment
        obs, reward, terminated, truncated, info = env.step(actions)

        # Accumulate episode data
        episode_rewards += reward.squeeze()
        episode_steps += 1

        # Store step data for each parallel environment
        for env_idx in range(env.num_envs):
            episodes_data[env_idx].append({
                "reward": reward[env_idx].item(),
                "action": actions[env_idx].cpu().numpy(),
                "terminated": terminated[env_idx].item(),
                "truncated": truncated[env_idx].item(),
            })

        step_count += 1

        # Check for episode completion
        done = terminated | truncated

        if done.any():
            # Process finished episodes
            for env_idx in range(env.num_envs):
                if done[env_idx] and episode_count < num_episodes:
                    # Extract episode data
                    ep_data = episodes_data[env_idx]

                    # Update metrics collector
                    for step_data in ep_data:
                        # Extract velocities from info if available (simplified)
                        lin_vel = 0.0
                        ang_vel = 0.0
                        collision = step_data.get("terminated", False)

                        collector.update_step(
                            reward=step_data["reward"],
                            linear_velocity=lin_vel,
                            angular_velocity=ang_vel,
                            action=step_data["action"],
                            collision=collision
                        )

                    # Finish episode
                    collector.finish_episode(episode_count)

                    episode_count += 1

                    # Reset tracking for this environment
                    episode_rewards[env_idx] = 0
                    episode_steps[env_idx] = 0
                    episodes_data[env_idx] = []

                    # Progress reporting
                    if episode_count % 10 == 0:
                        print(f"[INFO] Completed {episode_count}/{num_episodes} episodes")

    print(f"[INFO] Evaluation complete!")
    return collector


def main():
    # Determine checkpoint and algorithm
    if args_cli.checkpoint:
        checkpoint_path_str = args_cli.checkpoint
        # Validate checkpoint path for security
        try:
            checkpoint_path = validate_checkpoint_path(checkpoint_path_str)
            checkpoint_path = str(checkpoint_path)  # Convert back to string for compatibility
        except PathValidationError as e:
            print(f"[ERROR] Invalid checkpoint path: {e}")
            return

        if args_cli.algorithm:
            algorithm = args_cli.algorithm.upper()
        else:
            # Try to infer from path
            algorithm = None
            for algo in ["PPO", "SAC", "TD3"]:
                if algo in checkpoint_path.upper():
                    algorithm = algo
                    break
            if algorithm is None:
                print("[ERROR] Could not infer algorithm. Please specify --algorithm")
                return
    else:
        # Find latest checkpoint
        checkpoint_path, algorithm = find_latest_checkpoint()

    # Determine output directory
    if args_cli.output_dir:
        try:
            output_dir = validate_output_path(args_cli.output_dir, base_dir="results/evaluation")
        except PathValidationError as e:
            print(f"[ERROR] Invalid output directory: {e}")
            return
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = Path(checkpoint_path).stem
        output_dir = Path("results") / "evaluation" / f"{checkpoint_name}_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("EVALUATION CONFIGURATION")
    print("=" * 80)
    print(f"Checkpoint:    {checkpoint_path}")
    print(f"Algorithm:     {algorithm}")
    print(f"Episodes:      {args_cli.num_episodes}")
    print(f"Deterministic: {args_cli.deterministic}")
    print(f"Output Dir:    {output_dir}")
    print("=" * 80)

    # Create environment (use fewer parallel envs for evaluation)
    print("\n[INFO] Creating environment...")
    env_cfg = WarehouseEnvCfg()
    env_cfg.scene.num_envs = 16  # Balance speed vs independence

    env = gym.make("Warehouse-v0", cfg=env_cfg, render_mode="rgb_array")
    env = wrap_env(env, wrapper="isaaclab")
    device = env.device

    print(f"[INFO] Environment created with {env.num_envs} parallel environments")

    # Get config path
    config_path = AgentFactory.get_config_path(algorithm)

    # Create agent
    print(f"\n[INFO] Creating {algorithm} agent...")
    agent = AgentFactory.create_agent(algorithm, env, config_path, device)

    # Load checkpoint
    print(f"[INFO] Loading checkpoint from {checkpoint_path}...")
    try:
        agent.load(checkpoint_path)
        print("[INFO] Checkpoint loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return

    # Evaluate agent
    collector = evaluate_agent(
        agent,
        env,
        args_cli.num_episodes,
        deterministic=args_cli.deterministic
    )

    # Print summary
    collector.print_summary()

    # Save results
    print(f"\n[INFO] Saving results to {output_dir}...")
    collector.save_to_json(str(output_dir / "metrics.json"))
    collector.save_statistics(str(output_dir / "statistics.json"))
    collector.save_summary_txt(str(output_dir / "summary.txt"))

    # Generate plots
    try:
        from source.utils.plotting import EvaluationPlotter
        print("[INFO] Generating plots...")
        plots_dir = output_dir / "plots"
        plotter = EvaluationPlotter(str(plots_dir))
        plotter.plot_all(collector.get_episode_data())
        print(f"  - Plots: {plots_dir}")
    except ImportError:
        print("[INFO] Skipping plots (not available)")

    print(f"\n[INFO] Evaluation complete! Results saved to {output_dir}")
    print(f"[INFO] Summary:")
    print(f"  - Metrics: {output_dir / 'metrics.json'}")
    print(f"  - Statistics: {output_dir / 'statistics.json'}")
    print(f"  - Summary: {output_dir / 'summary.txt'}")

    # Log evaluation results to wandb
    if WANDB_AVAILABLE and wandb is not None:
        try:
            # Initialize wandb for evaluation
            wandb.init(
                project="WarehouseBenchmark-Evaluation",
                name=f"eval_{algorithm}_{Path(checkpoint_path).stem}",
                config={
                    "checkpoint": checkpoint_path,
                    "algorithm": algorithm,
                    "num_episodes": args_cli.num_episodes,
                    "deterministic": args_cli.deterministic
                },
                tags=["evaluation", "warehouse", algorithm.lower()]
            )

            # Log evaluation metrics
            try:
                summary_stats = collector.get_summary_statistics()
            except (AttributeError, KeyError, TypeError) as e:
                # Fallback if method doesn't exist or data is incomplete
                import logging
                logging.getLogger(__name__).debug(f"Could not get summary statistics: {e}")
                summary_stats = {}

            wandb.log({
                "evaluation_completed": True,
                "num_episodes": args_cli.num_episodes,
                "mean_reward": summary_stats.get("mean_reward", 0),
                "std_reward": summary_stats.get("std_reward", 0),
                "min_reward": summary_stats.get("min_reward", 0),
                "max_reward": summary_stats.get("max_reward", 0),
                "mean_episode_length": summary_stats.get("mean_episode_length", 0),
                "success_rate": summary_stats.get("success_rate", 0),
                "checkpoint_path": checkpoint_path,
                "algorithm": algorithm
            })

            print(f"[INFO] Evaluation results logged to wandb: {wandb.run.url if hasattr(wandb, 'run') and wandb.run else 'N/A'}")
            wandb.finish()

        except Exception as e:
            print(f"[WARNING] Failed to log to wandb: {e}")

    # Close environment
    env.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Evaluation interrupted by user")
    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
