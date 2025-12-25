"""
Benchmarking script for comparing multiple RL algorithms.

This script evaluates and compares multiple trained agents (PPO, SAC, TD3)
on the warehouse environment, generating comprehensive comparison reports.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
import json
import pandas as pd

# 1. LAUNCH APP FIRST
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description="Benchmark multiple RL algorithms.")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--algorithms", nargs="+", default=["PPO", "SAC", "TD3"],
                   help="Algorithms to benchmark (default: PPO SAC TD3)")
parser.add_argument("--num-episodes", type=int, default=100,
                   help="Episodes per algorithm evaluation (default: 100)")
parser.add_argument("--run-dirs", nargs="+", default=None,
                   help="Specific run directories to evaluate")
parser.add_argument("--output-dir", type=str, default=None,
                   help="Output directory for benchmark results")
args_cli = parser.parse_args()
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
from source.utils.plotting import BenchmarkPlotter


def find_latest_run(algorithm):
    """
    Find the latest run directory for an algorithm.

    Args:
        algorithm (str): Algorithm name

    Returns:
        Path: Run directory path

    Raises:
        FileNotFoundError: If no runs found
    """
    runs_dir = Path("runs")
    pattern = f"*_{algorithm.upper()}"
    matching_runs = sorted(runs_dir.glob(pattern), key=lambda x: x.stat().st_mtime)

    if not matching_runs:
        raise FileNotFoundError(f"No runs found for algorithm {algorithm}")

    return matching_runs[-1]


def find_best_checkpoint(run_dir):
    """
    Find the best checkpoint in a run directory.

    Args:
        run_dir (Path): Run directory

    Returns:
        Path: Checkpoint file path

    Raises:
        FileNotFoundError: If no checkpoints found
    """
    checkpoints_dir = Path(run_dir) / "checkpoints"

    # Prefer best_agent.pt if it exists
    best_checkpoint = checkpoints_dir / "best_agent.pt"
    if best_checkpoint.exists():
        return best_checkpoint

    # Otherwise, get the latest checkpoint
    checkpoint_files = sorted(checkpoints_dir.glob("agent_*.pt"),
                            key=lambda x: int(x.stem.split('_')[1]))

    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")

    return checkpoint_files[-1]


def evaluate_algorithm(algorithm, checkpoint_path, env, num_episodes):
    """
    Evaluate a single algorithm.

    Args:
        algorithm (str): Algorithm name
        checkpoint_path (Path): Checkpoint file path
        env: Wrapped environment
        num_episodes (int): Number of episodes

    Returns:
        MetricsCollector: Collector with evaluation metrics
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING {algorithm}")
    print(f"{'='*80}")
    print(f"Checkpoint: {checkpoint_path}")

    device = env.device

    # Load config
    config_path = AgentFactory.get_config_path(algorithm)
    agent = AgentFactory.create_agent(algorithm, env, config_path, device)

    # Load checkpoint
    try:
        agent.load(str(checkpoint_path))
        agent.set_mode("eval")
        print("[INFO] Checkpoint loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load checkpoint: {e}")
        return None

    # Run evaluation
    collector = MetricsCollector()
    obs, _ = env.reset()

    episode_count = 0
    step_count = 0
    episodes_data = [[] for _ in range(env.num_envs)]

    episode_rewards = torch.zeros(env.num_envs, device=device)
    episode_steps = torch.zeros(env.num_envs, device=device, dtype=torch.int32)

    print(f"[INFO] Running evaluation for {num_episodes} episodes...")

    while episode_count < num_episodes:
        # Get actions
        with torch.no_grad():
            actions = agent.act(obs, timestep=step_count, timesteps=999999)[0]

        # Step environment
        obs, reward, terminated, truncated, info = env.step(actions)

        # Accumulate episode data
        episode_rewards += reward
        episode_steps += 1

        # Store step data
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
            for env_idx in range(env.num_envs):
                if done[env_idx] and episode_count < num_episodes:
                    # Extract episode data
                    ep_data = episodes_data[env_idx]

                    # Update metrics collector
                    for step_data in ep_data:
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

                    # Reset tracking
                    episode_rewards[env_idx] = 0
                    episode_steps[env_idx] = 0
                    episodes_data[env_idx] = []

                    # Progress reporting
                    if episode_count % 20 == 0:
                        print(f"[INFO] Completed {episode_count}/{num_episodes} episodes")

    # Compute statistics
    stats = collector.compute_statistics()
    print(f"\n[INFO] {algorithm} Evaluation Complete:")
    print(f"  Mean Reward: {stats.mean_reward:.2f} +/- {stats.std_reward:.2f}")
    print(f"  Success Rate: {stats.success_rate:.1f}%")
    print(f"  Avg Collisions: {stats.avg_collision_count:.2f}")

    return collector


def main():
    print("=" * 80)
    print("WAREHOUSE BENCHMARK")
    print("=" * 80)
    print(f"Algorithms: {', '.join(args_cli.algorithms)}")
    print(f"Episodes per algorithm: {args_cli.num_episodes}")
    print("=" * 80)

    # Create output directory
    if args_cli.output_dir:
        output_dir = Path(args_cli.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("results") / "benchmarks" / f"warehouse_v0_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[INFO] Output directory: {output_dir}")

    # Setup environment (use fewer envs for evaluation)
    print("\n[INFO] Creating environment...")
    env_cfg = WarehouseEnvCfg()
    env_cfg.scene.num_envs = 16  # Balance speed vs statistics

    env = gym.make("Warehouse-v0", cfg=env_cfg, render_mode="rgb_array")
    env = wrap_env(env, wrapper="isaaclab")

    print(f"[INFO] Environment created with {env.num_envs} parallel environments")

    # Benchmark each algorithm
    results = {}
    collectors = {}

    for algorithm in args_cli.algorithms:
        try:
            # Find run directory
            if args_cli.run_dirs:
                # Use provided run directory
                run_dir = next((Path(d) for d in args_cli.run_dirs
                              if algorithm.upper() in str(d).upper()), None)
                if not run_dir:
                    print(f"\n[WARN] No run directory provided for {algorithm}, skipping")
                    continue
            else:
                # Find latest run
                run_dir = find_latest_run(algorithm)

            print(f"\n[INFO] Processing {algorithm} from {run_dir}")

            # Find checkpoint
            checkpoint_path = find_best_checkpoint(run_dir)

            # Evaluate
            collector = evaluate_algorithm(
                algorithm,
                checkpoint_path,
                env,
                args_cli.num_episodes
            )

            if collector:
                stats = collector.compute_statistics()
                results[algorithm] = stats.to_dict()
                collectors[algorithm] = collector

                # Save individual results
                algo_output_dir = output_dir / algorithm.lower()
                algo_output_dir.mkdir(exist_ok=True)
                collector.save_to_json(algo_output_dir / "metrics.json")
                collector.save_statistics(algo_output_dir / "statistics.json")
                collector.save_summary_txt(algo_output_dir / "summary.txt")

        except FileNotFoundError as e:
            print(f"\n[ERROR] {algorithm}: {e}")
            continue
        except Exception as e:
            print(f"\n[ERROR] Failed to benchmark {algorithm}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Close environment
    env.close()

    # Generate comparison results
    if results:
        print(f"\n{'='*80}")
        print("GENERATING COMPARISON REPORT")
        print(f"{'='*80}")

        # Save comparison JSON
        comparison_file = output_dir / "comparison.json"
        with open(comparison_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"[INFO] Saved comparison.json")

        # Save comparison CSV
        df = pd.DataFrame(results).T
        csv_file = output_dir / "comparison.csv"
        df.to_csv(csv_file)
        print(f"[INFO] Saved comparison.csv")

        # Generate summary report
        summary_file = output_dir / "summary.txt"
        with open(summary_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("BENCHMARK SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Environment: Warehouse-v0\n")
            f.write(f"Episodes per algorithm: {args_cli.num_episodes}\n\n")

            f.write("-" * 80 + "\n")
            f.write("ALGORITHM RESULTS\n")
            f.write("-" * 80 + "\n\n")

            for algorithm, metrics in results.items():
                f.write(f"{algorithm}:\n")
                f.write(f"  Mean Reward:    {metrics['mean_reward']:8.2f} � {metrics['std_reward']:.2f}\n")
                f.write(f"  Median Reward:  {metrics['median_reward']:8.2f}\n")
                f.write(f"  Min Reward:     {metrics['min_reward']:8.2f}\n")
                f.write(f"  Max Reward:     {metrics['max_reward']:8.2f}\n")
                f.write(f"  Success Rate:   {metrics['success_rate']:8.1f}%\n")
                f.write(f"  Avg Collisions: {metrics['avg_collision_count']:8.2f}\n")
                f.write(f"  Avg Length:     {metrics['mean_length']:8.1f} steps\n")
                f.write("\n")

            # Ranking
            sorted_algos = sorted(results.items(),
                                key=lambda x: x[1]['mean_reward'],
                                reverse=True)

            f.write("-" * 80 + "\n")
            f.write("RANKING (by mean reward)\n")
            f.write("-" * 80 + "\n\n")

            for i, (algo, metrics) in enumerate(sorted_algos, 1):
                f.write(f"  {i}. {algo:6s}: {metrics['mean_reward']:8.2f} "
                       f"(Success: {metrics['success_rate']:.1f}%)\n")

            f.write("\n" + "=" * 80 + "\n")

        print(f"[INFO] Saved summary.txt")

        # Print summary to console
        print(f"\n{'='*80}")
        print("BENCHMARK RESULTS")
        print(f"{'='*80}\n")

        for algorithm, metrics in results.items():
            print(f"{algorithm}:")
            print(f"  Mean Reward: {metrics['mean_reward']:.2f} � {metrics['std_reward']:.2f}")
            print(f"  Success Rate: {metrics['success_rate']:.1f}%")
            print()

        print("RANKING (by mean reward):")
        for i, (algo, metrics) in enumerate(sorted_algos, 1):
            print(f"  {i}. {algo}: {metrics['mean_reward']:.2f}")

        print(f"\n{'='*80}")

        # Generate plots
        print("\n[INFO] Generating comparison plots...")
        plots_dir = output_dir / "plots"
        plotter = BenchmarkPlotter(plots_dir)

        try:
            plotter.plot_all(results)
            print(f"[INFO] Plots saved to {plots_dir}")
        except Exception as e:
            print(f"[WARN] Failed to generate some plots: {e}")

        print(f"\n{'='*80}")
        print("BENCHMARK COMPLETE")
        print(f"{'='*80}")
        print(f"Results saved to: {output_dir}")
        print(f"  - Comparison: {comparison_file}")
        print(f"  - CSV: {csv_file}")
        print(f"  - Summary: {summary_file}")
        print(f"  - Plots: {plots_dir}")
        print(f"{'='*80}\n")

    else:
        print("\n[ERROR] No algorithms successfully benchmarked")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Benchmark interrupted by user")
    except Exception as e:
        print(f"[ERROR] Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
