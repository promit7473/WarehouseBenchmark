# WarehouseBenchmark

A Deep Reinforcement Learning benchmark for warehouse robot navigation using NVIDIA Isaac Lab.

## Overview

WarehouseBenchmark provides a realistic simulation environment for training autonomous mobile robots to navigate warehouse environments. The benchmark uses a Clearpath Jackal robot equipped with LiDAR sensors to navigate through a full-scale warehouse while avoiding obstacles and reaching waypoint goals.

### Key Features

- **Real LiDAR Perception**: 2D LiDAR with 180 rays providing 360° coverage
- **Realistic Localization**: Simulated GPS/SLAM noise (15cm position, 3° heading uncertainty)
- **Professional Robot Platform**: Clearpath Jackal differential drive robot
- **Curriculum Learning**: Progressive difficulty increase from 8m to 25m waypoint distances
- **Domain Randomization**: Mass, friction, and actuator gain variations for sim-to-real transfer
- **Multiple RL Algorithms**: PPO, SAC, and TD3 implementations via skrl
- **Warehouse-Scale Environment**: 31.8m × 54m navigable area

## Installation

### Prerequisites

- NVIDIA Isaac Lab ([installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html))
- Python 3.10+
- CUDA-compatible GPU (RTX 3070 or better recommended)

### Setup

1. Clone the repository:
```bash
git clone <repository-url> WarehouseBenchmark
cd WarehouseBenchmark
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify warehouse assets:
```bash
ls -lh assets/warehouse/full_warehouse.usd
```

## Usage

### Training

Train an agent using PPO (recommended):
```bash
~/IsaacLab/isaaclab.sh -p scripts/train.py \
    --algorithm PPO \
    --num-envs 128 \
    --headless
```

For visualization during training:
```bash
~/IsaacLab/isaaclab.sh -p scripts/train.py \
    --algorithm PPO \
    --num-envs 4
```

### Evaluation

Evaluate a trained agent:
```bash
~/IsaacLab/isaaclab.sh -p scripts/evaluate.py \
    --algorithm PPO \
    --checkpoint runs/latest/checkpoints/best_agent.pt \
    --num-episodes 100 \
    --deterministic
```

### Benchmarking

Compare multiple algorithms:
```bash
~/IsaacLab/isaaclab.sh -p scripts/benchmark.py \
    --algorithms PPO SAC TD3 \
    --num-episodes 100
```

### Video Recording

Record agent behavior:
```bash
~/IsaacLab/isaaclab.sh -p scripts/record_video.py \
    --checkpoint runs/PPO/checkpoints/best_agent.pt \
    --algorithm PPO \
    --num_episodes 3 \
    --output_dir ./videos
```

## Environment Specifications

### Observation Space (108 dimensions)

| Component | Dimensions | Description |
|-----------|------------|-------------|
| Linear Velocity | 3 | Robot's 3D velocity |
| Angular Velocity | 3 | Robot's 3D rotation rate |
| Gravity Vector | 3 | Gravity direction from IMU |
| Last Action | 4 | Previous wheel velocities |
| Waypoint Command | 4 | Goal position + heading (with 15cm noise) |
| Aisle Alignment | 2 | Warehouse grid alignment |
| LiDAR Scan | 89 | Distance measurements |

### Action Space (4 dimensions)

Continuous wheel velocities for differential drive control:
- Range: [-1, 1] (scaled internally by 10.0)
- Controls: `[front_left, front_right, rear_left, rear_right]`

### Reward Structure

| Component | Weight | Description |
|-----------|--------|-------------|
| Waypoint Reached | +10.0 | Sparse reward for reaching goal |
| Waypoint Progress | +3.0 | Dense reward for approach |
| Path Efficiency | +1.0 | Bonus for direct paths |
| Aisle Navigation | +0.5 | Warehouse-aware navigation |
| Collision Avoidance | -0.5 | LiDAR-based proximity penalty |
| Action Smoothness | -0.1 | Penalize jerky motions |
| Alive Bonus | +0.1 | Survival reward |

### Warehouse Environment

- **Total Size**: 31.8m × 54.0m (wall-to-wall)
- **Navigable Area**: 27.8m × 50.0m (with 2m safety margins)
- **Floor Material**: Concrete (static_friction=1.0, dynamic_friction=0.8)
- **Obstacles**: Shelves, walls, and storage areas from USD geometry

### Robot Platform

- **Model**: Clearpath Jackal
- **Type**: 4-wheel differential drive
- **Spawn Height**: 0.5m
- **Max Linear Velocity**: 10.0 m/s
- **Max Angular Velocity**: 10.0 rad/s

### Sensors

**LiDAR**:
- Type: 2D planar
- Rays: 89 (~4° resolution)
- Coverage: 360° horizontal
- Range: 10m
- Mount: 0.2m above base
- Update Rate: 25Hz

**Proprioceptive**:
- IMU (linear/angular velocities, gravity)
- Odometry (with realistic noise)
- Contact sensors (collision detection)

### Episode Configuration

- Duration: 60 seconds
- Simulation Frequency: 60 Hz
- Control Frequency: 10 Hz
- Success Threshold: 1.5m from waypoint
- Max Parallel Environments: 128 (headless), 4 (GUI)

### Curriculum Learning

The environment implements **progressive difficulty scaling** to improve training efficiency:

| Training Progress | Waypoint Distance Range | Difficulty |
|------------------|-------------------------|------------|
| 0-20% | 3-8m | Easy (nearby goals) |
| 20-40% | 3-12m | Medium |
| 40-60% | 3-16m | Longer distances |
| 60-80% | 3-20m | Warehouse-scale |
| 80-100% | 3-25m | Full warehouse |

**How it works:**
- The curriculum automatically increases maximum waypoint distance as training progresses
- Agents learn basic navigation on nearby goals before tackling full warehouse distances
- Progress is tracked and logged during training: `[CURRICULUM] Progress: 25.0% | Max waypoint distance: 12.0m`
- Implementation: `source/envs/mdp/curriculum.py`

**Benefits:**
- Faster convergence (agents master basics first)
- More stable training (gradual difficulty increase)
- Higher final performance (solid foundation before complex tasks)

## Supported Algorithms

### PPO (Proximal Policy Optimization)

Recommended for stable training on navigation tasks.

Configuration: `configs/ppo_warehouse.yaml`

Key hyperparameters:
- Learning rate: 1e-4
- Batch size: 256 (rollouts × mini_batches)
- Rollouts: 24
- Discount factor: 0.99
- GAE lambda: 0.95

### SAC (Soft Actor-Critic)

Off-policy algorithm with automatic entropy tuning.

Configuration: `configs/sac_warehouse.yaml`

Key hyperparameters:
- Learning rate: 1e-4
- Batch size: 128
- Memory size: 100k
- Discount factor: 0.99

### TD3 (Twin Delayed DDPG)

Deterministic policy gradient method.

Configuration: `configs/td3_warehouse.yaml`

Key hyperparameters:
- Learning rate: 1e-4
- Batch size: 128
- Memory size: 100k
- Policy delay: 2

## Project Structure

```
WarehouseBenchmark/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── setup.py                           # Package setup
├── pyproject.toml                     # Build configuration
│
├── assets/
│   └── warehouse/
│       └── full_warehouse.usd         # Warehouse environment
│
├── source/
│   ├── __init__.py
│   ├── envs/
│   │   ├── warehouse_env.py           # Environment configuration
│   │   ├── warehouse_constants.py     # Centralized constants
│   │   ├── config/
│   │   │   └── warehouse_static_cfg.py  # Scene configuration
│   │   └── mdp/
│   │       ├── observations.py        # Observation functions
│   │       ├── rewards.py             # Reward functions
│   │       ├── terminations.py        # Termination conditions
│   │       └── events.py              # Reset and event handlers
│   │
│   ├── agents/
│   │   ├── factory.py                 # Agent creation
│   │   └── models.py                  # Neural network architectures
│   │
│   └── utils/
│       ├── metrics.py                 # Evaluation metrics
│       └── plotting.py                # Visualization
│
├── scripts/
│   ├── train.py                       # Training script
│   ├── evaluate.py                    # Evaluation script
│   ├── benchmark.py                   # Multi-algorithm comparison
│   ├── play.py                        # Interactive testing
│   └── record_video.py                # Video recording
│
└── configs/
    ├── ppo_warehouse.yaml             # PPO configuration
    ├── sac_warehouse.yaml             # SAC configuration
    └── td3_warehouse.yaml             # TD3 configuration
```

## Configuration

All algorithms are configured via YAML files in the `configs/` directory. Key parameters:

- `agent.learning_rate`: Learning rate for optimizer
- `agent.batch_size`: Batch size for training
- `agent.rollouts`: Number of rollout steps (PPO only)
- `trainer.timesteps`: Total training timesteps

Training runs are logged to `runs/[timestamp]_[algorithm]/` with checkpoints saved in `checkpoints/`.

## Evaluation Metrics

The benchmark tracks:

- **Success Rate**: Percentage of episodes reaching waypoint
- **Collision Rate**: Percentage of episodes with collisions
- **Path Efficiency**: Ratio of actual to optimal path length
- **Average Reward**: Mean cumulative reward per episode
- **Episode Length**: Steps taken to reach goal

Target performance:
- Success rate > 80%
- Collision rate < 5%
- Path efficiency > 0.85

## Troubleshooting

### Robots spawn outside warehouse

Verify warehouse bounds in `source/envs/warehouse_constants.py` match the USD file.

### GPU memory errors

Reduce number of parallel environments:
```bash
--num-envs 64  # instead of 128
```

### Training crashes

Check observation space size in config matches actual observations (108 dimensions).

### Checkpoint not found

Use absolute path or list available checkpoints:
```bash
ls runs/*/checkpoints/
```

## License

MIT License - see LICENSE file for details.

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@software{warehousebenchmark2024,
  title={WarehouseBenchmark: A Deep RL Benchmark for Warehouse Navigation},
  year={2024},
  url={https://github.com/yourusername/WarehouseBenchmark}
}
```

## Acknowledgments

Built using:
- [NVIDIA Isaac Lab](https://isaac-sim.github.io/IsaacLab/)
- [Isaac Sim](https://developer.nvidia.com/isaac-sim)
- [skrl](https://skrl.readthedocs.io/)
