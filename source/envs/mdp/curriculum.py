# source/envs/mdp/curriculum.py
"""Curriculum learning functions for warehouse navigation."""

import torch
from isaaclab.envs import ManagerBasedRLEnv


def waypoint_distance_curriculum(env: ManagerBasedRLEnv, env_ids: torch.Tensor, min_dist: float, max_dist: float):
    """
    Progressive waypoint distance curriculum.

    Gradually increases maximum waypoint distance as training progresses:
    - 0-20% training: 3-8m (easy, nearby goals)
    - 20-40%: 3-12m (medium distance)
    - 40-60%: 3-16m (longer distances)
    - 60-80%: 3-20m (warehouse scale)
    - 80-100%: 3-25m (full warehouse)

    Args:
        env: The environment instance
        env_ids: Environment IDs to update (unused, applies globally)
        min_dist: Minimum waypoint distance (default: 3.0m)
        max_dist: Maximum waypoint distance at full curriculum (default: 25.0m)
    """
    # Get current training progress (0.0 to 1.0)
    # Use common_step_counter which tracks total steps across all envs
    if hasattr(env, 'common_step_counter'):
        current_step = env.common_step_counter
    elif hasattr(env, 'episode_length_buf'):
        # Fallback: estimate from episode buffer
        current_step = env.episode_length_buf.sum().item()
    else:
        current_step = 0

    # Get total training steps from config
    # Default: 50k steps for fast benchmarking
    total_steps = 50_000
    if hasattr(env, 'max_episode_length'):
        # Rough estimate if max_episode_length is available
        total_steps = getattr(env, 'total_training_steps', 50_000)

    # Calculate progress (0.0 to 1.0)
    progress = min(1.0, current_step / total_steps)

    # 5-stage curriculum progression
    if progress < 0.2:
        # Stage 1: Easy (0-20% training)
        current_max_dist = 8.0
    elif progress < 0.4:
        # Stage 2: Medium (20-40% training)
        current_max_dist = 12.0
    elif progress < 0.6:
        # Stage 3: Longer (40-60% training)
        current_max_dist = 16.0
    elif progress < 0.8:
        # Stage 4: Warehouse-scale (60-80% training)
        current_max_dist = 20.0
    else:
        # Stage 5: Full warehouse (80-100% training)
        current_max_dist = max_dist

    # Store in environment for waypoint generation to use
    if not hasattr(env.unwrapped, 'curriculum_max_distance'):
        env.unwrapped.curriculum_max_distance = current_max_dist
    else:
        # Only update if changed (avoid spamming logs)
        old_dist = env.unwrapped.curriculum_max_distance
        if abs(old_dist - current_max_dist) > 0.1:
            env.unwrapped.curriculum_max_distance = current_max_dist
            print(f"[CURRICULUM] Progress: {progress*100:.1f}% | Max waypoint distance: {current_max_dist:.1f}m")
