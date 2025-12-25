import argparse
import os
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record trained agent video.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
parser.add_argument("--algorithm", type=str, default=None, help="Algorithm (PPO, SAC, TD3). Auto-detected if not specified.")
parser.add_argument("--num_episodes", type=int, default=5, help="Number of episodes to record")
parser.add_argument("--output_dir", type=str, default="./videos", help="Output directory for videos")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
import numpy as np
from PIL import Image
import source.envs
from skrl.envs.wrappers.torch import wrap_env
from source.envs.warehouse_env import WarehouseEnvCfg
from source.agents.factory import AgentFactory

# Removed hardcoded model definitions - now using factory


def save_video_from_frames(frames, output_path, fps=30):
    """Save frames as MP4 video using imageio"""
    import imageio
    print(f"[INFO] Saving video to: {output_path}")
    imageio.mimsave(output_path, frames, fps=fps)
    print(f"[INFO] Video saved! ({len(frames)} frames)")


def main():
    print("\n" + "="*70)
    print("VIDEO RECORDING FOR GITHUB")
    print("="*70 + "\n")
    
    # Create output directory
    os.makedirs(args_cli.output_dir, exist_ok=True)
    
    # Setup environment (single robot for cleaner video)
    print("[INFO] Setting up environment...")
    env_cfg = WarehouseEnvCfg()
    env_cfg.scene.num_envs = 1  # Single robot for clean video
    
    env = gym.make("Warehouse-v0", cfg=env_cfg, render_mode="rgb_array")
    env = wrap_env(env, wrapper="isaaclab")
    device = env.device

    # Detect algorithm from checkpoint path
    algorithm = args_cli.algorithm
    if algorithm is None:
        # Try to detect from checkpoint path
        checkpoint_path = args_cli.checkpoint
        if 'SAC' in checkpoint_path.upper():
            algorithm = 'SAC'
        elif 'TD3' in checkpoint_path.upper():
            algorithm = 'TD3'
        else:
            algorithm = 'PPO'  # Default

    print(f"[INFO] Detected algorithm: {algorithm}")

    # Setup agent using factory
    print("[INFO] Initializing agent...")
    config_path = AgentFactory.get_config_path(algorithm)
    agent = AgentFactory.create_agent(algorithm, env, config_path, device)
    
    # Load checkpoint
    print(f"[INFO] Loading checkpoint: {args_cli.checkpoint}")
    try:
        agent.load(args_cli.checkpoint)
        print("[INFO] ✓ Checkpoint loaded successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to load checkpoint: {e}")
        simulation_app.close()
        return
    
    agent.set_mode("eval")
    
    # Record episodes
    print(f"\n[INFO] Recording {args_cli.num_episodes} episodes...")
    print("="*70)
    
    all_frames = []
    episode_rewards = []
    episode_lengths = []
    
    obs, _ = env.reset()
    episode_reward = 0.0
    episode_length = 0
    episodes_recorded = 0
    step_count = 0

    print("[INFO] Starting frame capture...")

    try:
        while episodes_recorded < args_cli.num_episodes and simulation_app.is_running():
            # Get action from agent
            with torch.no_grad():
                actions = agent.act(obs, timestep=0, timesteps=999999)[0]

            # Step environment
            obs, reward, terminated, truncated, info = env.step(actions)

            # Capture frame from Isaac Lab environment
            # Isaac Lab environments have render_mode="rgb_array" support
            try:
                # Get RGB frame from environment
                frame = env.unwrapped.sim.render()

                if frame is not None and len(frame.shape) == 3:
                    # Convert from float [0,1] to uint8 [0,255] if needed
                    if frame.dtype == np.float32 or frame.dtype == np.float64:
                        frame = (frame * 255).astype(np.uint8)

                    # Ensure it's in the right format (H, W, C)
                    if frame.shape[2] == 4:  # RGBA
                        frame = frame[:, :, :3]  # Convert to RGB

                    all_frames.append(frame)

                    # Progress indicator every 100 frames
                    if len(all_frames) % 100 == 0:
                        print(f"[INFO] Captured {len(all_frames)} frames...", end='\r')

            except Exception as e:
                # First frame might fail, that's okay
                if step_count > 10:
                    print(f"\n[WARN] Frame capture error at step {step_count}: {e}")

            episode_reward += reward.item()
            episode_length += 1
            step_count += 1
            
            # Check for episode end
            if terminated.any() or truncated.any():
                episodes_recorded += 1
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
                print(f"[EPISODE {episodes_recorded}] "
                      f"Reward: {episode_reward:7.2f} | "
                      f"Length: {episode_length:4d} steps")
                
                episode_reward = 0.0
                episode_length = 0
                
                # Reset if we haven't recorded enough episodes yet
                if episodes_recorded < args_cli.num_episodes:
                    obs, _ = env.reset()
    
    except KeyboardInterrupt:
        print("\n[INFO] Recording interrupted by user")
    
    # Save video
    if all_frames:
        checkpoint_name = os.path.basename(args_cli.checkpoint).replace('.pt', '')
        video_path = os.path.join(args_cli.output_dir, f"agent_{checkpoint_name}.mp4")
        save_video_from_frames(all_frames, video_path, fps=30)
    else:
        print("[WARN] No frames captured!")
    
    # Print statistics
    print("\n" + "="*70)
    print("RECORDING SUMMARY")
    print("="*70)
    print(f"Episodes recorded:  {episodes_recorded}")
    print(f"Total frames:       {len(all_frames)}")
    print(f"Average reward:     {np.mean(episode_rewards):.4f}")
    print(f"Average length:     {np.mean(episode_lengths):.1f} steps")
    print(f"Video saved to:     {video_path if all_frames else 'N/A'}")
    print("="*70 + "\n")
    
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()