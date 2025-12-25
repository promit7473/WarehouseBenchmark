import argparse
import sys
import os
import warnings

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Suppress Python warnings related to USD
warnings.filterwarnings("ignore", message=".*OrthogonalizeBasis.*")
warnings.filterwarnings("ignore", message=".*orthonormal.*")
warnings.filterwarnings("ignore", category=UserWarning)

# Filter stderr to suppress USD orthonormal warnings from C++ layer
class WarningFilter:
    """Filter to suppress specific USD warnings from stderr (C++ warnings)."""
    def __init__(self, stream):
        self.stream = stream
        self.suppress_patterns = [
            "OrthogonalizeBasis did not converge",
            "matrix may not be orthonormal",
            "background_color",
            "not available in the MDL",
            "Orthonormalize at line",
            "/builds/omniverse/usd-ci/USD",
            "[Warning] [omni.usd]",
        ]

    def write(self, message):
        # Suppress messages containing any of the patterns
        if any(pattern in message for pattern in self.suppress_patterns):
            return  # Suppress this message
        self.stream.write(message)

    def flush(self):
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)

# Apply the filter to stderr BEFORE any USD imports
sys.stderr = WarningFilter(sys.stderr)

# Suppress USD warnings via environment variables (before Isaac Sim loads)
os.environ["OMNI_LOG_LEVEL"] = "error"
os.environ["USD_DIAGNOSTICS_ENABLE"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PXR_LOG_LEVEL"] = "error"  # USD/Pixar logging level
os.environ["CARB_LOG_LEVEL"] = "error"  # Carbonite logging level

# 1. LAUNCH APP FIRST
from isaaclab.app import AppLauncher

# Fixed learning rates for consistent benchmarking
# No adaptive learning rates - we use consistent fixed rates

parser = argparse.ArgumentParser(description="Train an RL agent with multi-algorithm support.")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--algorithm", type=str, default="PPO",
                    choices=["PPO", "SAC", "TD3", "PPO_ENHANCED"],
                    help="RL algorithm to use (default: PPO)")
parser.add_argument("--config", type=str, default=None,
                    help="Path to config file (overrides --algorithm default)")
parser.add_argument("--num-envs", type=int, default=None,
                    help="Number of parallel environments (default: 64 for headless, 4 for GUI)")


args_cli = parser.parse_args()

# Validate command-line arguments
if args_cli.num_envs is not None:
    if args_cli.num_envs < 1 or args_cli.num_envs > 1024:
        print(f"[ERROR] --num-envs must be between 1 and 1024, got {args_cli.num_envs}")
        sys.exit(1)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 2. SUPPRESS WARNINGS IMMEDIATELY AFTER APP LAUNCH
import carb
try:
    settings = carb.settings.get_settings()

    # ============================================================
    # CRITICAL: Suppress OrthogonalizeBasis warnings from USD
    # These come from warehouse.usd having non-orthonormal matrices
    # ============================================================
    settings.set("/persistent/physics/warnOnNonOrthonormal", False)
    settings.set("/persistent/physx/warnOnNonOrthonormal", False)
    settings.set("/persistent/physxSimulation/warnOnNonOrthonormal", False)

    # Suppress USD matrix decomposition warnings
    settings.set("/persistent/usd/warnOnNonOrthonormal", False)
    settings.set("/omni.usd/warnOnNonOrthonormal", False)

    # Suppress fabric warnings
    settings.set("/persistent/omnihydra/useSceneGraphInstancing", True)

    # Additional USD warnings
    settings.set("/persistent/simulation/minFrameRate", 15)
    settings.set("/app/player/useFixedTimeStepping", False)

    # Suppress material/shader warnings (stops the looping warnings)
    settings.set("/rtx/materialDb/syncLoads", True)
    settings.set("/rtx/hydra/disableMaterialLoading", False)
    settings.set("/app/asyncRendering", False)
    settings.set("/app/asyncRenderingLowLatency", False)

    # Disable material validation warnings
    settings.set("/rtx/mdl/printMaterialLoadWarnings", False)

    # Suppress ALL USD-level warnings using carb logging
    settings.set("/log/level", "error")  # Only show errors, not warnings
    settings.set("/log/fileLogLevel", "error")
    settings.set("/log/outputStreamLevel", "error")

    # Suppress specific USD warnings
    settings.set("/app/usd/warnOnMissingRelationshipTargets", False)
    settings.set("/persistent/usd/warnOnMissingRelationshipTargets", False)

    # Use carb logging filter to suppress orthonormal warnings
    carb_log = carb.log.get_log()
    if carb_log:
        carb_log.set_level_threshold(carb.log.LogLevel.ERROR)

    # Speed up rendering for GUI mode
    settings.set("/rtx/rendermode", "PathTracing")
    settings.set("/rtx/pathtracing/spp", 1)  # Samples per pixel - lower for speed
    settings.set("/rtx/pathtracing/totalSpp", 4)
    settings.set("/rtx/pathtracing/maxBounces", 2)

    print("[INFO] USD visualization warnings suppressed and rendering optimized")
except Exception as e:
    print(f"[WARNING] Could not suppress warnings: {e}")

import torch
import gymnasium as gym

from skrl.envs.wrappers.torch import wrap_env
from skrl.trainers.torch import SequentialTrainer

# Weights & Biases integration for experiment tracking
# Import moved to conditional block below

# Enhanced logging and experiment tracking
try:
    import wandb
    # Check if wandb has the required functions
    if hasattr(wandb, 'init') and hasattr(wandb, 'log') and hasattr(wandb, 'finish'):
        WANDB_AVAILABLE = True
        print("[INFO] wandb is fully available and functional")
    else:
        WANDB_AVAILABLE = False
        wandb = None
        print("[WARNING] wandb imported but missing required functions")
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False
    print("[WARNING] wandb not available. Install with: pip install wandb")

import source.envs
from source.envs.warehouse_env import WarehouseEnvCfg
from source.agents.factory import AgentFactory


def main():
    # Determine algorithm and config path
    algorithm = args_cli.algorithm.upper()
    config_path = args_cli.config or AgentFactory.get_config_path(algorithm)

    print(f"[INFO] Training with {algorithm} algorithm")
    print(f"[INFO] Loading config from: {config_path}")
    
    # Load configuration with error handling
    try:
        config = AgentFactory.load_config(config_path)
        print(f"[INFO] Loaded config with fixed learning rate: {config['agent']['learning_rate']:.2e}")
    except Exception as e:
        print(f"[ERROR] Failed to load config from {config_path}: {e}")
        return

    # Initialize wandb for experiment tracking
    if WANDB_AVAILABLE and wandb is not None:
        wandb.init(
            project="WarehouseBenchmark",
            name=f"{algorithm}_{torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}",
            config={
                "algorithm": algorithm,
                "config_path": config_path,
                "num_envs": args_cli.num_envs,
                "total_timesteps": config["trainer"]["timesteps"]
            },
            tags=["warehouse", "navigation", algorithm.lower()]
        )
        print("[INFO] Wandb logging enabled")
    else:
        print("[INFO] Wandb not available - proceeding without experiment tracking")

    # Curriculum Learning: Progressive difficulty increase
    total_timesteps = config["trainer"]["timesteps"]
    curriculum_stages = [
        (0.0, 8.0),      # Stage 1: 0-20% training, max distance 8m
        (0.2, 12.0),     # Stage 2: 20-40% training, max distance 12m
        (0.4, 16.0),     # Stage 3: 40-60% training, max distance 16m
        (0.6, 20.0),     # Stage 4: 60-80% training, max distance 20m
        (0.8, 25.0),     # Stage 5: 80-100% training, max distance 25m
    ]

    # Adjust config for GUI mode stability (fewer environments = more unstable)
    if hasattr(args_cli, 'headless') and not args_cli.headless:
        print("[INFO] GUI mode detected - adjusting config for stability")
        if algorithm in ["SAC", "TD3"]:
            # Increase learning_starts to collect more experience before learning
            config["agent"]["learning_starts"] = max(config["agent"].get("learning_starts", 1), 1000)
            # Reduce batch_size for stability with fewer environments
            config["agent"]["batch_size"] = min(config["agent"].get("batch_size", 128), 32)
            # Reduce memory size to avoid overfitting with limited experience
            config["agent"]["memory_size"] = min(config["agent"].get("memory_size", 100000), 10000)
            print(f"[INFO] Adjusted for GUI: learning_starts={config['agent']['learning_starts']}, "
                  f"batch_size={config['agent']['batch_size']}, memory_size={config['agent']['memory_size']}")

    # Create environment
    print("[INFO] Creating warehouse environment...")

    env_cfg = WarehouseEnvCfg()
    print("[INFO] Using warehouse environment (recommended for most GPUs)")

    # Set seed for reproducible training (addresses non-deterministic warning)
    import random
    import numpy as np
    import torch
    seed = 42  # Fixed seed for reproducible results
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    env_cfg.seed = seed
    print(f"[INFO] Set random seed to {seed} for reproducible training")

    # Determine number of environments and render mode
    if args_cli.num_envs is not None:
        # Warehouse environment - ensure environments fit within warehouse bounds
        # Navigable area: X=[-24.3, 3.5] (27.8m), Y=[-21.4, 28.6] (50m)
        warehouse_nav_x = 27.8  # meters (navigable width)
        warehouse_nav_y = 50.0  # meters (navigable height)
        env_cfg.scene.env_spacing = 3.5  # Closer spacing for more environments
        max_envs_x = max(1, int(warehouse_nav_x / env_cfg.scene.env_spacing))
        max_envs_y = max(1, int(warehouse_nav_y / env_cfg.scene.env_spacing))
        max_envs_total = max_envs_x * max_envs_y

        env_cfg.scene.num_envs = min(args_cli.num_envs, max_envs_total)
        render_mode = "rgb_array" if args_cli.headless else None

        if args_cli.num_envs > max_envs_total:
            print(f"[WARNING] Requested {args_cli.num_envs} environments exceeds warehouse capacity ({max_envs_total})")
            print(f"[INFO] Warehouse navigable area: {warehouse_nav_x:.1f}m × {warehouse_nav_y:.1f}m")
            print(f"[INFO] Using {env_cfg.scene.num_envs} environments ({max_envs_x}x{max_envs_y} grid) with {env_cfg.scene.env_spacing}m spacing")
        else:
            print(f"[INFO] Using {env_cfg.scene.num_envs} environments (user specified)")
    elif hasattr(args_cli, 'headless') and not args_cli.headless:
        env_cfg.scene.num_envs = 4  # Reduced for GUI stability
        render_mode = None  # GUI rendering
        print("[INFO] GUI mode detected - using 4 environments for stability")
    else:
        # Warehouse environment - ensure environments fit within warehouse bounds
        # Navigable area: X=[-24.3, 3.5] (27.8m), Y=[-21.4, 28.6] (50m)
        warehouse_nav_x = 27.8  # meters (navigable width)
        warehouse_nav_y = 50.0  # meters (navigable height)
        env_cfg.scene.env_spacing = 3.5  # Closer spacing for more environments
        max_envs_x = max(1, int(warehouse_nav_x / env_cfg.scene.env_spacing))
        max_envs_y = max(1, int(warehouse_nav_y / env_cfg.scene.env_spacing))
        max_envs_total = max_envs_x * max_envs_y
        env_cfg.scene.num_envs = min(args_cli.num_envs or 32, max_envs_total)
        print(f"[INFO] Warehouse environment - navigable area: {warehouse_nav_x:.1f}m × {warehouse_nav_y:.1f}m")
        print(f"[INFO] Using {env_cfg.scene.num_envs} environments ({max_envs_x}x{max_envs_y} grid) with {env_cfg.scene.env_spacing}m spacing")
        if args_cli.num_envs and args_cli.num_envs > max_envs_total:
            print(f"[WARNING] Requested {args_cli.num_envs} environments exceeds warehouse capacity ({max_envs_total})")
        render_mode = "rgb_array"  # Offscreen rendering

    env = gym.make("Warehouse-v0", cfg=env_cfg, render_mode=render_mode)
    print(f"[DEBUG] After gym.make: action_space = {env.action_space}, shape = {env.action_space.shape}")

    env = wrap_env(env, wrapper="isaaclab")

    # Adjust environment origins to center on warehouse bounds
    # This ensures all environments are placed within warehouse walls
    from source.envs.warehouse_env import adjust_env_origins_for_warehouse
    adjust_env_origins_for_warehouse(env)


    device = env.device

    print(f"[INFO] Environment created with {env.num_envs} parallel environments")
    print(f"[INFO] Observation space: {env.observation_space}")
    print(f"[INFO] Action space: {env.action_space}")
    print(f"[INFO] Device: {device}")

    # Create agent using factory
    agent = AgentFactory.create_agent(algorithm, env, config_path, device)



    # Start training
    print("[INFO] Starting training loop...")
    trainer = SequentialTrainer(cfg=config["trainer"], env=env, agents=agent)
    trainer.train()

    print("[INFO] Training completed successfully!")

    # Log final results to wandb
    if WANDB_AVAILABLE and wandb is not None:
        # Log training completion and summary
        final_metrics = {
            "training_completed": True,
            "total_timesteps": config["trainer"]["timesteps"],
            "algorithm": algorithm,
            "device": str(device),
            "num_envs": len(env.envs) if hasattr(env, 'envs') else 'unknown'
        }

        # Try to get basic training statistics
        try:
            # Log some basic info about the training
            final_metrics.update({
                "config_used": config_path,
                "learning_rate": config["agent"]["learning_rate"],
                "batch_size": config["agent"].get("batch_size", "N/A"),
            })
        except (KeyError, AttributeError, TypeError) as e:
            import logging
            logging.getLogger(__name__).debug(f"Could not log full training metrics: {e}")

        wandb.log(final_metrics)
        print(f"[INFO] Training results logged to wandb: {wandb.run.url if hasattr(wandb, 'run') and wandb.run else 'N/A'}")
        wandb.finish()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user")
    except Exception as e:
        print(f"FATAL TRAINING ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure proper cleanup
        print("[INFO] Cleaning up...")
        try:
            if 'simulation_app' in locals():
                simulation_app.close()
                print("[INFO] Simulation app closed successfully")
        except Exception as e:
            print(f"[WARNING] Error closing simulation app: {e}")
        
        # Graceful cleanup of any lingering Isaac Lab processes
        # Note: Aggressive process killing removed to prevent GPU memory corruption
        try:
            import subprocess
            import os
            current_pid = os.getpid()

            # Find Isaac Lab/Isaac Sim processes belonging to current session
            result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
            lines = result.stdout.split('\n')
            pids = []
            for line in lines:
                if 'isaac' in line.lower() and 'grep' not in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        pid = parts[1]
                        # Don't kill the current process or its parent
                        if pid != str(current_pid) and pid != str(os.getppid()):
                            pids.append(pid)

            if pids:
                # First try graceful termination (SIGTERM)
                try:
                    term_cmd = ["kill", "-15"] + pids
                    subprocess.run(term_cmd, capture_output=True, text=True, timeout=5)
                    print(f"[INFO] Sent SIGTERM to Isaac Lab processes: {pids}")

                    # Wait briefly for graceful shutdown
                    import time
                    time.sleep(2)

                    # Check if processes are still running
                    result = subprocess.run(["ps", "-p", ",".join(pids)], capture_output=True, text=True)
                    if result.returncode == 0:
                        # Some processes still running - force kill as last resort
                        kill_cmd = ["kill", "-9"] + pids
                        subprocess.run(kill_cmd, capture_output=True, text=True)
                        print(f"[WARNING] Force killed remaining processes: {pids}")
                except subprocess.TimeoutExpired:
                    print("[WARNING] Graceful shutdown timed out")
            else:
                print("[DEBUG] No orphan Isaac Lab processes found")
        except Exception as e:
            print(f"[DEBUG] Could not cleanup processes: {e}")
        
        print("[INFO] Cleanup complete")