import argparse
import sys
import os
import glob
import subprocess
from pathlib import Path

# Add project root to Python path BEFORE importing isaaclab
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Set PYTHONPATH environment variable
os.environ['PYTHONPATH'] = project_root + ':' + os.environ.get('PYTHONPATH', '')

from isaaclab.app import AppLauncher

# Kill any previous Isaac Sim processes to avoid conflicts
try:
    subprocess.run(["pkill", "-f", "isaac-sim"], check=False)
    print("[INFO] Killed previous Isaac Sim processes")
except Exception as e:
    print(f"[WARNING] Could not kill previous processes: {e}")

# 1. Launch App with GUI (No --headless)
parser = argparse.ArgumentParser(description="Play a trained RL agent.")
parser.add_argument("--algorithm", type=str, default=None,
                     help="Algorithm to use (PPO, SAC, TD3). Auto-detected if not specified.")
parser.add_argument("--checkpoint", type=str, default=None,
                     help="Path to checkpoint file. Uses latest if not specified.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
import source.envs
from source.agents.factory import AgentFactory
from skrl.envs.wrappers.torch import wrap_env
from source.envs.warehouse_env import WarehouseEnvCfg
try:
    from omni.isaac.debug_draw import _debug_draw
    DEBUG_DRAW_AVAILABLE = True
    print("[INFO] Debug draw available - enabling command arrows")
    # Try to enable debug drawing
    try:
        _debug_draw.enable(True)
        print("[INFO] Debug drawing enabled")
    except (AttributeError, RuntimeError, ImportError) as e:
        print(f"[WARN] Could not enable debug drawing automatically: {e}")
except ImportError:
    DEBUG_DRAW_AVAILABLE = False
    print("[WARN] Debug draw not available - command arrows may not show")


def find_latest_checkpoint():
    """Find the most recent checkpoint file."""
    checkpoint_files = glob.glob("./runs/*/checkpoints/*.pt")
    
    if not checkpoint_files:
        return None
    
    # Sort by modification time and get the latest
    return max(checkpoint_files, key=os.path.getmtime)


def detect_algorithm_from_checkpoint(checkpoint_path):
    """Detect algorithm from checkpoint directory name."""
    # Path format: ./runs/25-12-03_15-05-20-390692_SAC/checkpoints/agent_100000.pt
    path_parts = Path(checkpoint_path).parts
    for part in path_parts:
        if '_' in part:
            possible_algo = part.split('_')[-1].upper()
            if possible_algo in AgentFactory.SUPPORTED_ALGORITHMS:
                return possible_algo
    return None


def main():
    print("[INFO] Setting up environment for inference...")
    
    # 1. Find checkpoint
    checkpoint_path = args_cli.checkpoint
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint()
        if checkpoint_path is None:
            print("[ERROR] No checkpoints found in ./runs/")
            print("[INFO] Please train an agent first using:")
            print("       ~/IsaacLab/isaaclab.sh -p scripts/train.py --algorithm SAC --headless")
            simulation_app.close()
            return
        print(f"[INFO] Auto-detected latest checkpoint: {checkpoint_path}")
    
    # 2. Detect algorithm
    algorithm = args_cli.algorithm
    if algorithm is None:
        algorithm = detect_algorithm_from_checkpoint(checkpoint_path)
        if algorithm is None:
            print("[ERROR] Could not detect algorithm from checkpoint path")
            print("[INFO] Please specify algorithm with --algorithm PPO|SAC|TD3")
            simulation_app.close()
            return
        print(f"[INFO] Auto-detected algorithm: {algorithm}")
    else:
        algorithm = algorithm.upper()
    
    if algorithm not in AgentFactory.SUPPORTED_ALGORITHMS:
        print(f"[ERROR] Unsupported algorithm: {algorithm}")
        print(f"[INFO] Supported: {AgentFactory.SUPPORTED_ALGORITHMS}")
        simulation_app.close()
        return
    
    # 3. Configure Environment (match training setup)
    env_cfg = WarehouseEnvCfg()
    env_cfg.scene.num_envs = 1  # Match training: 4 parallel environments
    env_cfg.episode_length_s = 60.0  # Longer episodes for better observation (60 seconds)

    # 4. Create Environment
    env = gym.make("Warehouse-v0", cfg=env_cfg, render_mode="rgb_array")
    env = wrap_env(env, wrapper="isaaclab")
    device = env.device

    # Debug: Print robot position after reset
    obs, _ = env.reset()
    if hasattr(env.unwrapped, 'scene') and hasattr(env.unwrapped.scene, 'robot'):
        robot_pos = env.unwrapped.scene.robot.data.root_state_w[:, :3]
        print(f"[DEBUG] Robot position after reset: {robot_pos}")
    else:
        print("[DEBUG] Cannot access robot position directly")

    # Initialize debug drawing for path tracing
    if DEBUG_DRAW_AVAILABLE:
        draw = _debug_draw.acquire_debug_draw_interface()
        paths = [[] for _ in range(env_cfg.scene.num_envs)]
    else:
        draw = None
        paths = None
    
    # 5. Create agent using factory (matches training configuration)
    config_path = AgentFactory.get_config_path(algorithm)
    print(f"[INFO] Loading config from: {config_path}")
    
    try:
        agent = AgentFactory.create_agent(algorithm, env, config_path, device)
    except Exception as e:
        print(f"[ERROR] Failed to create agent: {e}")
        import traceback
        traceback.print_exc()
        simulation_app.close()
        return
    
    # 6. Load checkpoint
    try:
        print(f"[INFO] Loading checkpoint from: {checkpoint_path}")
        agent.load(checkpoint_path)
        print("[INFO] Checkpoint loaded successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to load checkpoint: {e}")
        import traceback
        traceback.print_exc()
        simulation_app.close()
        return
    
    # 7. Run Inference Loop
    print("[INFO] Starting inference loop...")
    print("[INFO] Close the Isaac Sim window to stop")
    print("[INFO] TIP: Press 'F' to focus camera on robot, or use mouse to navigate")
    
    obs, _ = env.reset()
    
    # Try to set camera view to look at origin where robot should be
    try:
        from omni.isaac.core.utils.viewports import set_camera_view
        set_camera_view(eye=[5, 5, 10], target=[0, 0, 0])
        print("[INFO] Camera positioned to look at origin")
    except (ImportError, AttributeError, RuntimeError) as e:
        print(f"[WARN] Could not set camera view automatically: {e}")
    episode_count = 0
    step_count = 0
    total_reward = 0.0
    
    # Set agent to evaluation mode (deterministic, no learning)
    agent.set_mode("eval")
    
    # Disable reward logging to focus on performance
    print("[INFO] Agent in EVALUATION MODE - showing trained performance only")
    
    try:
        # Main loop - integrates with Isaac Sim rendering
        while simulation_app.is_running():
            # Get actions from trained agent (deterministic for evaluation)
            with torch.no_grad():
                actions = agent.act(obs, timestep=step_count, timesteps=999999)[0]
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(actions)

            total_reward += reward.mean().item()
            step_count += 1

            # Path tracing: update paths and draw red lines
            if DEBUG_DRAW_AVAILABLE and draw and paths:
                draw.clear()
                positions = env.unwrapped.robot.data.root_state_w[:, :3].cpu().numpy()
                for i in range(env_cfg.scene.num_envs):
                    paths[i].append(positions[i].tolist())
                for i in range(env_cfg.scene.num_envs):
                    if len(paths[i]) > 1:
                        points = paths[i]
                        lines = []
                        for j in range(len(points)-1):
                            lines.append((points[j], points[j+1]))
                        if lines:
                            draw.draw_lines(lines, [(1,0,0,1)] * len(lines), [2] * len(lines))

            # Print progress
            if step_count % 100 == 0:
                avg_reward = total_reward / step_count
                print(f"[INFO] Steps: {step_count} | Avg Reward: {avg_reward:.4f} | Episodes: {episode_count}")

            # Handle resets
            if terminated.any() or truncated.any():
                reset_mask = terminated | truncated
                if DEBUG_DRAW_AVAILABLE and paths:
                    for i in range(env_cfg.scene.num_envs):
                        if reset_mask[i]:
                            paths[i] = []
                episode_count += terminated.sum().item() + truncated.sum().item()

            # Update simulation (refreshes GUI)
            simulation_app.update()
                    
    except KeyboardInterrupt:
        print("\n[INFO] Inference interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Runtime error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        avg_reward = total_reward / max(step_count, 1)
        print(f"\n[INFO] Final Statistics:")
        print(f"  Algorithm: {algorithm}")
        print(f"  Total steps: {step_count}")
        print(f"  Total episodes: {episode_count}")
        print(f"  Average reward: {avg_reward:.4f}")
        env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()