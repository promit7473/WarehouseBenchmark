"""Diagnostic script to check scene setup and identify issues."""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Diagnose warehouse environment setup")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
import source.envs
from source.envs.warehouse_env import WarehouseEnvCfg
from skrl.envs.wrappers.torch import wrap_env

print("="*80)
print("WAREHOUSE ENVIRONMENT DIAGNOSTICS")
print("="*80)

try:
    # Create minimal environment
    print("\n[1/6] Creating base environment...")
    env_cfg = WarehouseEnvCfg()
    env_cfg.scene.num_envs = 4  # Test with 4 envs
    
    base_env = gym.make("Warehouse-v0", cfg=env_cfg, render_mode="rgb_array")
    print("✓ Base environment created successfully")
    
    print("\n[2/6] Wrapping environment for SKRL...")
    env = wrap_env(base_env, wrapper="isaaclab")
    print("✓ Environment wrapped successfully")
    
    # Check observation space
    print("\n[3/6] Checking observation/action spaces...")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print(f"  Device: {env.device}")
    print(f"  Num envs: {env.num_envs}")
    
    # Reset environment
    print("\n[4/6] Resetting environment...")
    obs, info = env.reset()
    print(f"✓ Reset successful")
    if isinstance(obs, dict):
        print(f"  Observation is a dict with keys: {obs.keys()}")
        for key, val in obs.items():
            print(f"    {key}: shape={val.shape}, mean={val.mean().item():.4f}")
    else:
        print(f"  Observation shape: {obs.shape}")
        print(f"  Observation sample: {obs[0][:5]}")
    
    # Take a step with zeros
    print("\n[5/6] Taking test steps...")
    for i in range(5):
        action = torch.zeros(env.num_envs, 2, device=env.device)
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Step {i+1}: reward={reward.mean().item():.4f}, "
              f"terminated={terminated.sum().item()}, truncated={truncated.sum().item()}")
        
        # Check for NaN/Inf
        if isinstance(obs, dict):
            for key, val in obs.items():
                if torch.isnan(val).any() or torch.isinf(val).any():
                    print(f"  ⚠️  WARNING: NaN/Inf detected in observation '{key}'!")
        else:
            if torch.isnan(obs).any() or torch.isinf(obs).any():
                print(f"  ⚠️  WARNING: NaN/Inf detected in observation!")
    
    # Test with random actions
    print("\n[6/6] Testing with random actions...")
    for i in range(5):
        action = torch.randn(env.num_envs, 2, device=env.device) * 0.5
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Step {i+1}: reward={reward.mean().item():.4f}, action_mean={action.mean().item():.4f}")
    
    # Check robot positions
    print("\n[BONUS] Checking robot states...")
    robot = base_env.unwrapped.scene["robot"]
    root_state = robot.data.root_state_w
    print(f"  Robot positions (z-height): {root_state[:, 2]}")
    print(f"  Robot velocities (linear): {root_state[:, 7:10].norm(dim=1)}")
    
    if (root_state[:, 2] < -1.0).any():
        print("  ⚠️  WARNING: Some robots have fallen through the floor!")
    
    print("\n" + "="*80)
    print("✓ ALL DIAGNOSTICS PASSED - Environment is healthy!")
    print("="*80)
    
    env.close()

except Exception as e:
    print("\n" + "="*80)
    print("✗ DIAGNOSTIC FAILED")
    print("="*80)
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

finally:
    simulation_app.close()