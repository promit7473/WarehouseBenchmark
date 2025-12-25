"""
Metrics collection and statistical analysis for RL evaluation.

This module provides classes for collecting episode-level metrics during evaluation
and computing aggregated statistics across multiple episodes.
"""

import numpy as np
import json
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode."""

    episode_id: int
    total_reward: float
    episode_length: int
    success: bool  # No collision
    collision_count: int
    avg_linear_velocity: float
    avg_angular_velocity: float
    velocity_tracking_error: float
    action_smoothness: float  # L2 norm of action differences
    survival_time: float

    def to_dict(self):
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class AggregatedMetrics:
    """Aggregated statistics across episodes."""

    num_episodes: int

    # Reward statistics
    mean_reward: float
    std_reward: float
    min_reward: float
    max_reward: float
    median_reward: float

    # Episode length
    mean_length: float
    std_length: float

    # Success metrics
    success_rate: float  # Percentage
    avg_collision_count: float

    # Velocity metrics
    mean_linear_velocity: float
    mean_angular_velocity: float
    mean_tracking_error: float

    # Action quality
    mean_action_smoothness: float

    # Confidence intervals (95%)
    reward_ci_lower: float
    reward_ci_upper: float

    def to_dict(self):
        """Convert to dictionary."""
        return asdict(self)


class MetricsCollector:
    """Collect and aggregate metrics during evaluation."""

    def __init__(self):
        self.episodes: List[EpisodeMetrics] = []
        self.current_episode_data = None
        self.reset_episode()

    def reset_episode(self):
        """Initialize tracking for new episode."""
        self.current_episode_data = {
            "rewards": [],
            "linear_velocities": [],
            "angular_velocities": [],
            "actions": [],
            "collisions": 0,
            "steps": 0
        }

    def update_step(self, reward, linear_velocity, angular_velocity, action, collision):
        """
        Update metrics for current step.

        Args:
            reward (float): Step reward
            linear_velocity (float): Linear velocity
            angular_velocity (float): Angular velocity
            action (array): Action taken
            collision (bool): Whether collision occurred
        """
        self.current_episode_data["rewards"].append(float(reward))
        self.current_episode_data["linear_velocities"].append(float(linear_velocity))
        self.current_episode_data["angular_velocities"].append(float(angular_velocity))
        self.current_episode_data["actions"].append(action)
        self.current_episode_data["collisions"] += int(collision)
        self.current_episode_data["steps"] += 1

    def finish_episode(self, episode_id: int):
        """
        Compute and store episode metrics.

        Args:
            episode_id (int): Episode identifier

        Returns:
            EpisodeMetrics: Computed metrics for the episode
        """
        data = self.current_episode_data

        if len(data["rewards"]) == 0:
            # Empty episode, skip
            self.reset_episode()
            return None

        # Compute action smoothness
        actions = np.array(data["actions"])
        if len(actions) > 1:
            action_diffs = np.diff(actions, axis=0)
            smoothness = float(np.mean(np.linalg.norm(action_diffs, axis=1)))
        else:
            smoothness = 0.0

        # Compute velocity tracking error (simplified - assume target is mean)
        target_lin_vel = 1.0  # Typical target
        target_ang_vel = 0.0  # Typical target
        lin_vel_error = np.abs(np.array(data["linear_velocities"]) - target_lin_vel)
        ang_vel_error = np.abs(np.array(data["angular_velocities"]) - target_ang_vel)
        tracking_error = float(np.mean(lin_vel_error + ang_vel_error))

        # Create episode metrics
        metrics = EpisodeMetrics(
            episode_id=episode_id,
            total_reward=float(sum(data["rewards"])),
            episode_length=data["steps"],
            success=(data["collisions"] == 0),
            collision_count=data["collisions"],
            avg_linear_velocity=float(np.mean(data["linear_velocities"])) if data["linear_velocities"] else 0.0,
            avg_angular_velocity=float(np.mean(data["angular_velocities"])) if data["angular_velocities"] else 0.0,
            velocity_tracking_error=tracking_error,
            action_smoothness=smoothness,
            survival_time=data["steps"] * 0.02  # Assuming 50Hz (20ms per step)
        )

        self.episodes.append(metrics)
        self.reset_episode()

        return metrics

    def compute_statistics(self) -> AggregatedMetrics:
        """
        Compute aggregated statistics.

        Returns:
            AggregatedMetrics: Aggregated statistics across all episodes

        Raises:
            ValueError: If no episodes have been collected
        """
        if not self.episodes:
            raise ValueError("No episodes collected")

        rewards = [ep.total_reward for ep in self.episodes]
        lengths = [ep.episode_length for ep in self.episodes]

        # Confidence interval (95%)
        reward_std = np.std(rewards)
        reward_ci = 1.96 * reward_std / np.sqrt(len(rewards))

        return AggregatedMetrics(
            num_episodes=len(self.episodes),
            mean_reward=float(np.mean(rewards)),
            std_reward=float(reward_std),
            min_reward=float(np.min(rewards)),
            max_reward=float(np.max(rewards)),
            median_reward=float(np.median(rewards)),
            mean_length=float(np.mean(lengths)),
            std_length=float(np.std(lengths)),
            success_rate=100.0 * sum(ep.success for ep in self.episodes) / len(self.episodes),
            avg_collision_count=float(np.mean([ep.collision_count for ep in self.episodes])),
            mean_linear_velocity=float(np.mean([ep.avg_linear_velocity for ep in self.episodes])),
            mean_angular_velocity=float(np.mean([ep.avg_angular_velocity for ep in self.episodes])),
            mean_tracking_error=float(np.mean([ep.velocity_tracking_error for ep in self.episodes])),
            mean_action_smoothness=float(np.mean([ep.action_smoothness for ep in self.episodes])),
            reward_ci_lower=float(np.mean(rewards) - reward_ci),
            reward_ci_upper=float(np.mean(rewards) + reward_ci)
        )

    def save_to_json(self, filepath: str):
        """
        Save all episode metrics to JSON.

        Args:
            filepath (str): Path to output JSON file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "episodes": [ep.to_dict() for ep in self.episodes],
            "statistics": self.compute_statistics().to_dict()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"[INFO] Metrics saved to {filepath}")

    def save_statistics(self, filepath: str):
        """
        Save only aggregated statistics to JSON.

        Args:
            filepath (str): Path to output JSON file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        stats = self.compute_statistics()

        with open(filepath, 'w') as f:
            json.dump(stats.to_dict(), f, indent=2)

        print(f"[INFO] Statistics saved to {filepath}")

    def save_summary_txt(self, filepath: str):
        """
        Save human-readable summary to text file.

        Args:
            filepath (str): Path to output text file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        stats = self.compute_statistics()

        with open(filepath, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("EVALUATION SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Episodes Evaluated: {stats.num_episodes}\n\n")

            f.write("REWARD STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Mean:   {stats.mean_reward:8.2f} ± {stats.std_reward:.2f}\n")
            f.write(f"  Median: {stats.median_reward:8.2f}\n")
            f.write(f"  Min:    {stats.min_reward:8.2f}\n")
            f.write(f"  Max:    {stats.max_reward:8.2f}\n")
            f.write(f"  95% CI: [{stats.reward_ci_lower:.2f}, {stats.reward_ci_upper:.2f}]\n\n")

            f.write("SUCCESS METRICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Success Rate:         {stats.success_rate:6.2f}%\n")
            f.write(f"  Avg Collision Count:  {stats.avg_collision_count:6.2f}\n\n")

            f.write("EPISODE LENGTH\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Mean:   {stats.mean_length:8.2f} ± {stats.std_length:.2f} steps\n\n")

            f.write("VELOCITY METRICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Avg Linear Velocity:   {stats.mean_linear_velocity:6.3f} m/s\n")
            f.write(f"  Avg Angular Velocity:  {stats.mean_angular_velocity:6.3f} rad/s\n")
            f.write(f"  Tracking Error:        {stats.mean_tracking_error:6.3f}\n\n")

            f.write("ACTION QUALITY\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Action Smoothness:     {stats.mean_action_smoothness:6.3f}\n\n")

            f.write("=" * 80 + "\n")

        print(f"[INFO] Summary saved to {filepath}")

    def get_episode_data(self) -> List[Dict]:
        """
        Return list of episode dictionaries for plotting.

        Returns:
            list: List of episode data dictionaries
        """
        return [ep.to_dict() for ep in self.episodes]

    def print_summary(self):
        """Print summary statistics to console."""
        stats = self.compute_statistics()

        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        print(f"\nEpisodes: {stats.num_episodes}")
        print(f"Mean Reward: {stats.mean_reward:.2f} ± {stats.std_reward:.2f}")
        print(f"Success Rate: {stats.success_rate:.1f}%")
        print(f"Avg Collisions: {stats.avg_collision_count:.2f}")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    # Example usage
    collector = MetricsCollector()

    # Simulate 3 episodes
    for ep in range(3):
        for step in range(10):
            collector.update_step(
                reward=1.0,
                linear_velocity=0.5,
                angular_velocity=0.1,
                action=[0.2, 0.3],
                collision=False
            )
        collector.finish_episode(ep)

    # Print summary
    collector.print_summary()

    # Show episode data
    print("Episode data:")
    for ep_data in collector.get_episode_data():
        print(f"  Episode {ep_data['episode_id']}: Reward={ep_data['total_reward']:.2f}")
