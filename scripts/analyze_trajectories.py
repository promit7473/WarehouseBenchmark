#!/usr/bin/env python3
"""
Trajectory Analysis and Comparison Tool for Warehouse Benchmark

This script analyzes trajectories from trained agents and provides comparison metrics
for research papers. It generates path traces, performance statistics, and visualizations.
"""

import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple
import torch

def analyze_trajectories(runs_dir: str, algorithms: List[str] = None) -> Dict:
    """
    Analyze trajectories from different algorithm runs.

    Args:
        runs_dir: Directory containing algorithm run folders
        algorithms: List of algorithms to analyze (if None, analyze all)

    Returns:
        Dictionary containing analysis results
    """
    runs_path = Path(runs_dir)
    results = {}

    # Find all algorithm runs
    run_folders = [f for f in runs_path.iterdir() if f.is_dir() and any(alg in f.name for alg in ['PPO', 'SAC', 'TD3'])]

    if algorithms:
        run_folders = [f for f in run_folders if any(alg in f.name for alg in algorithms)]

    for run_folder in run_folders:
        algorithm = 'PPO' if 'PPO' in run_folder.name else ('SAC' if 'SAC' in run_folder.name else 'TD3')
        print(f"Analyzing {algorithm} run: {run_folder.name}")

        # Find the best checkpoint (highest step count)
        checkpoints = list(run_folder.glob("*.pt"))
        if not checkpoints:
            print(f"No checkpoints found for {algorithm}")
            continue

        best_checkpoint = max(checkpoints, key=lambda x: int(x.stem.split('_')[-1]) if x.stem.split('_')[-1].isdigit() else 0)

        # Analyze this checkpoint
        analysis = analyze_single_run(run_folder, best_checkpoint, algorithm)
        results[algorithm] = analysis

    return results

def analyze_single_run(run_folder: Path, checkpoint_path: Path, algorithm: str) -> Dict:
    """
    Analyze a single algorithm run.
    """
    # This would load the checkpoint and run evaluation episodes
    # For now, return placeholder analysis

    analysis = {
        'algorithm': algorithm,
        'checkpoint': str(checkpoint_path),
        'metrics': {
            'average_reward': 0.0,
            'success_rate': 0.0,
            'path_efficiency': 0.0,
            'completion_time': 0.0,
            'waypoint_reach_times': [],
            'total_distance': 0.0,
            'oscillation_penalty': 0.0,
            # Enhanced metrics for comprehensive evaluation
            'collision_rate': 0.0,
            'energy_efficiency': 0.0,
            'waypoint_progression_rate': 0.0,
            'aisle_adherence_score': 0.0,
            'camera_utilization_score': 0.0,  # For vision-based agents
            'real_time_factor': 0.0,  # Simulation speed vs real-time
            'stability_score': 0.0,  # Based on velocity/acceleration variance
        },
        'trajectory_data': {
            'positions': [],
            'velocities': [],
            'waypoint_reaches': [],
            'camera_features': [],  # For vision-based analysis
            'collision_events': [],
            'energy_consumption': []
        }
    }

    return analysis

def generate_comparison_report(results: Dict, output_dir: str = "./analysis"):
    """
    Generate comparison report and visualizations.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Create summary table
    summary_data = []
    for alg, data in results.items():
        metrics = data['metrics']
        summary_data.append({
            'Algorithm': alg,
            'Avg Reward': metrics['average_reward'],
            'Success Rate': metrics['success_rate'],
            'Path Efficiency': metrics['path_efficiency'],
            'Completion Time': metrics['completion_time'],
            'Total Distance': metrics['total_distance']
        })

    df = pd.DataFrame(summary_data)
    df.to_csv(output_path / "algorithm_comparison.csv", index=False)

    # Generate plots
    generate_comparison_plots(df, output_path)

    # Save detailed results
    with open(output_path / "detailed_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Analysis report saved to {output_path}")

def generate_comparison_plots(df: pd.DataFrame, output_path: Path):
    """
    Generate comparison plots.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Algorithm Comparison - Warehouse Navigation Benchmark')

    metrics = ['Avg Reward', 'Success Rate', 'Path Efficiency', 'Completion Time', 'Total Distance']

    for i, metric in enumerate(metrics):
        ax = axes[i // 3, i % 3]
        bars = ax.bar(df['Algorithm'], df[metric])
        ax.set_title(metric)
        ax.set_ylabel(metric)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   '.2f' if isinstance(height, float) else str(height),
                   ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_path / "algorithm_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_trajectory_visualization(results: Dict, output_dir: str = "./analysis"):
    """
    Create trajectory visualization showing paths taken by different algorithms.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Define waypoint positions
    waypoints = np.array([
        [5, 5],
        [-5, 5],
        [-5, -5],
        [5, -5]
    ])

    plt.figure(figsize=(12, 10))

    # Plot waypoints
    colors = ['green', 'yellow', 'blue', 'magenta']
    for i, (wp, color) in enumerate(zip(waypoints, colors)):
        plt.scatter(wp[0], wp[1], c=color, s=200, marker='o', edgecolors='black', linewidth=2, zorder=5)
        plt.text(wp[0], wp[1] + 0.5, f'Goal {i+1}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Plot trajectories for each algorithm
    algorithm_colors = {'PPO': 'red', 'SAC': 'blue', 'TD3': 'green'}
    algorithm_styles = {'PPO': '-', 'SAC': '--', 'TD3': ':'}

    for alg, data in results.items():
        trajectory = data.get('trajectory_data', {}).get('positions', [])

        if trajectory:
            trajectory = np.array(trajectory)
            plt.plot(trajectory[:, 0], trajectory[:, 1],
                    color=algorithm_colors.get(alg, 'black'),
                    linestyle=algorithm_styles.get(alg, '-'),
                    linewidth=3, alpha=0.8, label=f'{alg} Path',
                    zorder=3)

            # Mark start and end points
            if len(trajectory) > 0:
                plt.scatter(trajectory[0, 0], trajectory[0, 1],
                           color=algorithm_colors.get(alg, 'black'),
                           marker='^', s=150, label=f'{alg} Start', zorder=4)
                plt.scatter(trajectory[-1, 0], trajectory[-1, 1],
                           color=algorithm_colors.get(alg, 'black'),
                           marker='s', s=150, label=f'{alg} End', zorder=4)

    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Algorithm Trajectory Comparison - Warehouse Navigation')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    # Add warehouse boundary
    plt.axhline(y=10, color='black', linestyle='-', alpha=0.5)
    plt.axhline(y=-10, color='black', linestyle='-', alpha=0.5)
    plt.axvline(x=10, color='black', linestyle='-', alpha=0.5)
    plt.axvline(x=-10, color='black', linestyle='-', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path / "trajectory_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze and compare algorithm trajectories")
    parser.add_argument("--runs_dir", type=str, default="./runs",
                       help="Directory containing algorithm run folders")
    parser.add_argument("--algorithms", nargs='+', choices=['PPO', 'SAC', 'TD3'],
                       help="Algorithms to analyze (default: all)")
    parser.add_argument("--output_dir", type=str, default="./analysis",
                       help="Output directory for analysis results")

    args = parser.parse_args()

    print("🔍 Starting trajectory analysis...")
    print(f"📁 Analyzing runs in: {args.runs_dir}")
    print(f"🤖 Algorithms: {args.algorithms or 'All'}")
    print(f"📊 Output directory: {args.output_dir}")

    # Analyze trajectories
    results = analyze_trajectories(args.runs_dir, args.algorithms)

    if not results:
        print("❌ No valid runs found for analysis")
        return

    print(f"✅ Analyzed {len(results)} algorithms")

    # Generate comparison report
    generate_comparison_report(results, args.output_dir)

    # Create trajectory visualization
    create_trajectory_visualization(results, args.output_dir)

    print("🎉 Analysis complete!")
    print(f"📈 Check {args.output_dir} for detailed results and visualizations")

if __name__ == "__main__":
    main()