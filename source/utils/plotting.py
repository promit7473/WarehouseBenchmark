"""
Visualization utilities for RL evaluation and benchmarking.

This module provides plotting classes for generating publication-quality visualizations
of evaluation metrics and benchmark comparisons.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

# Set style
sns.set_style("whitegrid")
sns.set_palette("colorblind")


class EvaluationPlotter:
    """Generate plots for single-agent evaluation."""

    def __init__(self, output_dir: str):
        """
        Initialize plotter.

        Args:
            output_dir (str): Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_reward_distribution(self, episodes: List[Dict], filename="reward_distribution.png"):
        """
        Histogram of episode rewards.

        Args:
            episodes (list): List of episode data dictionaries
            filename (str): Output filename
        """
        rewards = [ep["total_reward"] for ep in episodes]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(rewards, bins=30, alpha=0.7, edgecolor='black', color='steelblue')
        ax.axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(rewards):.2f}')
        ax.axvline(np.median(rewards), color='green', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(rewards):.2f}')
        ax.set_xlabel('Episode Reward', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Episode Rewards', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_episode_length_histogram(self, episodes: List[Dict], filename="episode_length_hist.png"):
        """
        Histogram of episode lengths.

        Args:
            episodes (list): List of episode data dictionaries
            filename (str): Output filename
        """
        lengths = [ep["episode_length"] for ep in episodes]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(lengths, bins=30, alpha=0.7, edgecolor='black', color='coral')
        ax.axvline(np.mean(lengths), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(lengths):.1f}')
        ax.set_xlabel('Episode Length (steps)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Episode Lengths', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_velocity_tracking(self, episodes: List[Dict], filename="velocity_tracking.png"):
        """
        Box plots of velocity metrics.

        Args:
            episodes (list): List of episode data dictionaries
            filename (str): Output filename
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Linear velocity
        linear_vels = [ep["avg_linear_velocity"] for ep in episodes]
        bp1 = axes[0].boxplot([linear_vels], widths=0.5, patch_artist=True)
        bp1['boxes'][0].set_facecolor('lightblue')
        axes[0].set_ylabel('Linear Velocity (m/s)', fontsize=12)
        axes[0].set_title('Average Linear Velocity per Episode', fontsize=13, fontweight='bold')
        axes[0].set_xticklabels([''])
        axes[0].grid(True, alpha=0.3, axis='y')

        # Angular velocity
        angular_vels = [ep["avg_angular_velocity"] for ep in episodes]
        bp2 = axes[1].boxplot([angular_vels], widths=0.5, patch_artist=True)
        bp2['boxes'][0].set_facecolor('lightcoral')
        axes[1].set_ylabel('Angular Velocity (rad/s)', fontsize=12)
        axes[1].set_title('Average Angular Velocity per Episode', fontsize=13, fontweight='bold')
        axes[1].set_xticklabels([''])
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_success_metrics(self, episodes: List[Dict], filename="success_metrics.png"):
        """
        Bar chart of success rate and collision statistics.

        Args:
            episodes (list): List of episode data dictionaries
            filename (str): Output filename
        """
        success_rate = 100 * sum(ep["success"] for ep in episodes) / len(episodes)
        avg_collisions = np.mean([ep["collision_count"] for ep in episodes])

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Success rate
        axes[0].bar(['Success', 'Collision'],
                    [success_rate, 100 - success_rate],
                    color=['green', 'red'], alpha=0.7, edgecolor='black', linewidth=1.5)
        axes[0].set_ylabel('Percentage (%)', fontsize=12)
        axes[0].set_title(f'Success Rate: {success_rate:.1f}%', fontsize=13, fontweight='bold')
        axes[0].set_ylim([0, 100])
        axes[0].grid(True, alpha=0.3, axis='y')

        # Collision histogram
        collision_counts = [ep["collision_count"] for ep in episodes]
        max_collisions = max(collision_counts) if collision_counts else 1
        axes[1].hist(collision_counts, bins=range(max_collisions + 2),
                     alpha=0.7, edgecolor='black', color='indianred')
        axes[1].set_xlabel('Number of Collisions', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title(f'Collision Distribution (Avg: {avg_collisions:.2f})',
                         fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_all(self, episodes: List[Dict]):
        """
        Generate all evaluation plots.

        Args:
            episodes (list): List of episode data dictionaries
        """
        print(f"[INFO] Generating evaluation plots in {self.output_dir}...")
        self.plot_reward_distribution(episodes)
        self.plot_episode_length_histogram(episodes)
        self.plot_velocity_tracking(episodes)
        self.plot_success_metrics(episodes)
        print(f"[INFO] All plots saved to {self.output_dir}")


class BenchmarkPlotter:
    """Generate comparison plots for multiple algorithms."""

    def __init__(self, output_dir: str):
        """
        Initialize plotter.

        Args:
            output_dir (str): Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_learning_curves(self, data: Dict[str, pd.DataFrame],
                            filename="learning_curves.png"):
        """
        Plot training curves from TensorBoard logs.

        Args:
            data (dict): Dictionary mapping algorithm names to DataFrames with 'step' and 'reward' columns
            filename (str): Output filename
        """
        if not data:
            print("[WARN] No learning curve data available")
            return

        fig, ax = plt.subplots(figsize=(12, 7))

        for algorithm, df in data.items():
            # Smooth the curve with rolling average
            window = max(10, len(df) // 50)
            if len(df) > window:
                smoothed = df['reward'].rolling(window=window, center=True).mean()
                ax.plot(df['step'], smoothed, label=algorithm, linewidth=2)

                # Add confidence band
                rolling_min = df['reward'].rolling(window=window, center=True).min()
                rolling_max = df['reward'].rolling(window=window, center=True).max()
                ax.fill_between(df['step'], rolling_min, rolling_max, alpha=0.2)
            else:
                ax.plot(df['step'], df['reward'], label=algorithm, linewidth=2, marker='o')

        ax.set_xlabel('Training Steps', fontsize=12)
        ax.set_ylabel('Episode Reward', fontsize=12)
        ax.set_title('Learning Curves Comparison', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_final_performance(self, results: Dict[str, Dict],
                              filename="final_performance.png"):
        """
        Bar chart comparing final performance metrics.

        Args:
            results (dict): Dictionary mapping algorithm names to metrics dictionaries
            filename (str): Output filename
        """
        algorithms = list(results.keys())
        mean_rewards = [results[alg]['mean_reward'] for alg in algorithms]
        std_rewards = [results[alg]['std_reward'] for alg in algorithms]

        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(algorithms))
        bars = ax.bar(x, mean_rewards, yerr=std_rewards,
                     capsize=5, alpha=0.7, edgecolor='black', linewidth=1.5)

        # Color bars by performance
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(algorithms)))
        sorted_indices = np.argsort(mean_rewards)
        for i, bar in enumerate(bars):
            bar.set_color(colors[np.where(sorted_indices == i)[0][0]])

        ax.set_xticks(x)
        ax.set_xticklabels(algorithms, fontsize=11)
        ax.set_ylabel('Mean Episode Reward', fontsize=12)
        ax.set_title('Final Performance Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for i, (mean, std) in enumerate(zip(mean_rewards, std_rewards)):
            ax.text(i, mean + std + max(mean_rewards) * 0.02,
                   f'{mean:.1f}±{std:.1f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_radar_chart(self, results: Dict[str, Dict],
                        metrics: List[str] = None,
                        filename="radar_comparison.png"):
        """
        Radar chart comparing multiple metrics.

        Args:
            results (dict): Dictionary mapping algorithm names to metrics dictionaries
            metrics (list): List of metric names to compare
            filename (str): Output filename
        """
        if metrics is None:
            metrics = ['mean_reward', 'success_rate', 'mean_length',
                      'mean_linear_velocity', 'mean_action_smoothness']

        # Filter metrics that exist in results
        available_metrics = []
        for metric in metrics:
            if all(metric in results[alg] for alg in results.keys()):
                available_metrics.append(metric)

        if not available_metrics:
            print("[WARN] No common metrics available for radar chart")
            return

        algorithms = list(results.keys())
        num_metrics = len(available_metrics)

        # Normalize metrics to [0, 1]
        normalized_data = {}
        for metric in available_metrics:
            values = [results[alg].get(metric, 0) for alg in algorithms]
            min_val, max_val = min(values), max(values)
            if max_val > min_val:
                normalized_data[metric] = [(v - min_val) / (max_val - min_val)
                                          for v in values]
            else:
                normalized_data[metric] = [0.5] * len(values)

        # Setup radar chart
        angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        for i, algorithm in enumerate(algorithms):
            values = [normalized_data[metric][i] for metric in available_metrics]
            values += values[:1]  # Complete the circle

            ax.plot(angles, values, 'o-', linewidth=2, label=algorithm)
            ax.fill(angles, values, alpha=0.15)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(available_metrics, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_title('Multi-Metric Comparison', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_all(self, results: Dict[str, Dict], learning_curves: Dict = None):
        """
        Generate all benchmark plots.

        Args:
            results (dict): Dictionary mapping algorithm names to metrics dictionaries
            learning_curves (dict): Optional dictionary of learning curve DataFrames
        """
        print(f"[INFO] Generating benchmark plots in {self.output_dir}...")
        self.plot_final_performance(results)
        self.plot_radar_chart(results)
        if learning_curves:
            self.plot_learning_curves(learning_curves)
        print(f"[INFO] All plots saved to {self.output_dir}")


if __name__ == "__main__":
    # Example usage for EvaluationPlotter
    print("Testing EvaluationPlotter...")

    # Create dummy episode data
    episodes = []
    for i in range(50):
        episodes.append({
            "episode_id": i,
            "total_reward": np.random.normal(100, 20),
            "episode_length": int(np.random.normal(200, 30)),
            "success": np.random.random() > 0.3,
            "collision_count": int(np.random.poisson(1)),
            "avg_linear_velocity": np.random.normal(0.5, 0.1),
            "avg_angular_velocity": np.random.normal(0.0, 0.1),
        })

    plotter = EvaluationPlotter("test_plots/evaluation")
    plotter.plot_all(episodes)

    print("\nTesting BenchmarkPlotter...")

    # Create dummy results
    results = {
        "PPO": {"mean_reward": 120, "std_reward": 15, "success_rate": 85,
               "mean_length": 200, "mean_linear_velocity": 0.5, "mean_action_smoothness": 0.1},
        "SAC": {"mean_reward": 110, "std_reward": 18, "success_rate": 80,
               "mean_length": 190, "mean_linear_velocity": 0.48, "mean_action_smoothness": 0.12},
        "TD3": {"mean_reward": 115, "std_reward": 16, "success_rate": 82,
               "mean_length": 195, "mean_linear_velocity": 0.49, "mean_action_smoothness": 0.11},
    }

    bench_plotter = BenchmarkPlotter("test_plots/benchmark")
    bench_plotter.plot_all(results)

    print("\nTest plots generated in test_plots/")
