"""
Neural network model definitions for RL algorithms.

This module contains all model architectures used across different RL algorithms (PPO, SAC, TD3).
All models share the same architecture (256 -> 128 hidden layers) for fair comparison.
"""

import torch
import torch.nn as nn
from skrl.models.torch import GaussianMixin, DeterministicMixin, Model


# ============================================================================
# CNN Models for Vision Processing (Enhanced)
# ============================================================================

class ConvEncoder(nn.Module):
    """Convolutional encoder for image processing."""
    def __init__(self, in_channels, encoder_features=[16, 32, 64], activation="elu"):
        super().__init__()
        self.encoder_layers = nn.ModuleList()

        for i, feature in enumerate(encoder_features):
            if i == 0:
                self.encoder_layers.append(nn.Conv2d(in_channels, feature, kernel_size=3, stride=1, padding=1))
            else:
                self.encoder_layers.append(nn.Conv2d(encoder_features[i-1], feature, kernel_size=3, stride=1, padding=1))

            self.encoder_layers.append(nn.BatchNorm2d(feature))
            self.encoder_layers.append(self._get_activation(activation))
            self.encoder_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

    def _get_activation(self, activation):
        if activation == "elu":
            return nn.ELU()
        elif activation == "relu":
            return nn.ReLU()
        elif activation == "leaky_relu":
            return nn.LeakyReLU(0.1)
        else:
            return nn.ReLU()

    def forward(self, x):
        for layer in self.encoder_layers:
            x = layer(x)
        return x


class HeightmapCNN(nn.Module):
    """CNN for processing heightmap data."""
    def __init__(self, heightmap_size=8, features=[32, 64]):
        super().__init__()
        self.encoder = ConvEncoder(1, features)  # 1 channel for heightmap

        # Calculate output size after convolution
        # Assuming 8x8 input -> 4x4 after first conv+pool -> 2x2 after second -> 1x1 after third
        self.flatten_size = features[-1] * 1 * 1  # Final feature map size
        self.fc = nn.Linear(self.flatten_size, 64)  # Output embedding

    def forward(self, x):
        # x shape: (batch, heightmap_size, heightmap_size)
        x = x.unsqueeze(1)  # Add channel dimension: (batch, 1, H, W)
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x


class VisionEncoder(nn.Module):
    """Multi-modal vision encoder combining RGB and depth."""
    def __init__(self, image_size=(80, 60), features=[32, 64, 128]):
        super().__init__()
        self.rgb_encoder = ConvEncoder(3, features)  # RGB channels
        self.depth_encoder = ConvEncoder(1, features)  # Depth channel

        # Calculate flattened size
        # After 3 conv+pool layers: image_size -> image_size/2 -> image_size/4 -> image_size/8
        h, w = image_size[0] // 8, image_size[1] // 8
        self.flatten_size = features[-1] * h * w * 2  # *2 for RGB + depth

        self.fusion_fc = nn.Sequential(
            nn.Linear(self.flatten_size, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64)
        )

    def forward(self, rgb, depth):
        # Process RGB and depth separately
        rgb_features = self.rgb_encoder(rgb)
        depth_features = self.depth_encoder(depth.unsqueeze(1))  # Add channel dim

        # Concatenate features
        combined = torch.cat([rgb_features, depth_features], dim=1)
        combined = combined.view(combined.size(0), -1)

        # Fusion through FC layers
        output = self.fusion_fc(combined)
        return output


# ============================================================================
# PPO Models (On-Policy, Stochastic)
# ============================================================================

class PolicyPPO(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self,
                               clip_actions=clip_actions,
                               clip_log_std=True,
                               min_log_std=-20.0,
                               max_log_std=2.0)

        # Shared feature extractor
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
        )

        # Action mean
        self.mean_layer = nn.Linear(128, self.num_actions)

        # Action log standard deviation (learnable parameter)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions, device=device))

    def compute(self, inputs, role):
        x = self.net(inputs["states"])
        return self.mean_layer(x), self.log_std_parameter, {}


class ValuePPO(DeterministicMixin, Model):
    """
    Value function network for PPO algorithm.
    """
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)

        # Value network
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1)
        )

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


# ============================================================================
# SAC Models (Off-Policy, Stochastic) - FIXED
# ============================================================================

class PolicySAC(GaussianMixin, Model):
    """
    Stochastic policy network for SAC algorithm (FIXED for NaN issues).

    Architecture: Input(11) -> Linear(256) -> ELU -> Linear(128) -> ELU -> Output(2)
    Output: Mean and log_std for Gaussian distribution over actions

    Key improvements:
    - Better weight initialization for stable training
    - Log std initialization for controlled exploration
    - Input sanitization to prevent NaN propagation
    """
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self,
                                clip_actions=clip_actions,
                                clip_log_std=True,
                                min_log_std=-20.0,
                                max_log_std=2.0)

        # Shared feature extractor
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
        )

        # Mean head
        self.mean_layer = nn.Linear(128, self.num_actions)

        # Log std parameter - initialize to log(0.1) = -2.3 for controlled exploration
        # This prevents the large std values that cause instability
        self.log_std_parameter = nn.Parameter(torch.full((self.num_actions,), -2.3))

        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights to prevent NaN values."""
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

        # Initialize mean layer with small weights to ensure actions start near 0
        nn.init.xavier_uniform_(self.mean_layer.weight, gain=0.01)
        nn.init.constant_(self.mean_layer.bias, 0.0)

    def compute(self, inputs, role):
        """Compute mean and log_std for action distribution."""
        # Check for NaN in inputs (Safety check)
        if torch.isnan(inputs["states"]).any():
            inputs["states"] = torch.nan_to_num(inputs["states"], nan=0.0)

        x = self.net(inputs["states"])
        mean = self.mean_layer(x)

        # Clamp mean to prevent extreme values during early training
        mean = torch.clamp(mean, -1.0, 1.0)  # Match TD3 action range

        # Ensure log_std is reasonable to prevent NaN in sampling
        if not self.training:
            # In eval mode, use deterministic actions (very small std)
            log_std = torch.full_like(self.log_std_parameter, -20.0)  # Near-zero std for deterministic
        else:
            log_std = torch.clamp(self.log_std_parameter, -10.0, 0.0)  # Prevent extreme std

        return mean, log_std, {}


class CriticSAC(DeterministicMixin, Model):
    """
    Q-function network for SAC algorithm.
    Architecture: Input(11+2) -> Linear(256) -> ELU -> Linear(128) -> ELU -> Linear(1)
    """
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)
        
        # Q-network takes state + action as input
        self.net = nn.Sequential(
            nn.Linear(self.num_observations + self.num_actions, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def compute(self, inputs, role):
        # Concatenate state and action
        x = torch.cat([inputs["states"], inputs["taken_actions"]], dim=-1)
        return self.net(x), {}


# ============================================================================
# TD3 Models (Off-Policy, Deterministic)
# ============================================================================

class ActorTD3(DeterministicMixin, Model):
    """
    TD3 Actor network with improved numerical stability.

    Architecture: Input(11) -> Linear(256) -> ELU -> Linear(128) -> ELU -> Linear(2) -> Tanh
    Output: Actions in [-1, 1] range
    """
    def __init__(self, observation_space, action_space, device, clip_actions=True):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, self.num_actions),
            nn.Tanh()
        )

        # Initialize weights for numerical stability
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights for numerical stability."""
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def compute(self, inputs, role):
        # Input validation and sanitization
        states = inputs["states"]
        if torch.isnan(states).any() or torch.isinf(states).any():
            states = torch.nan_to_num(states, nan=0.0, posinf=1.0, neginf=-1.0)
            inputs["states"] = states

        return self.net(states), {}


# ============================================================================
# Enhanced Vision-Based Models (CNN Integration)
# ============================================================================

class PolicyPPOVision(GaussianMixin, Model):
    """Enhanced PPO Policy with CNN vision processing."""
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self,
                                clip_actions=clip_actions,
                                clip_log_std=True,
                                min_log_std=-20.0,
                                max_log_std=2.0)

        # Vision encoders
        self.vision_encoder = VisionEncoder(image_size=(80, 60))  # Match camera resolution
        self.heightmap_encoder = HeightmapCNN(heightmap_size=8)

        # LiDAR processing (simplified)
        self.lidar_encoder = nn.Sequential(
            nn.Linear(32, 64),  # 32 LiDAR rays
            nn.ELU(),
            nn.Linear(64, 32),
            nn.ELU()
        )

        # Pose processing
        self.pose_encoder = nn.Sequential(
            nn.Linear(3, 32),  # x, y, yaw
            nn.ELU(),
            nn.Linear(32, 16),
            nn.ELU()
        )

        # Fusion network
        vision_dim = 64  # From VisionEncoder
        heightmap_dim = 64  # From HeightmapCNN
        lidar_dim = 32  # From lidar_encoder
        pose_dim = 16  # From pose_encoder
        total_dim = vision_dim + heightmap_dim + lidar_dim + pose_dim

        self.fusion_net = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU()
        )

        # Action heads
        self.mean_layer = nn.Linear(128, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions, device=device))

    def compute(self, inputs, role):
        # Extract different observation modalities
        states = inputs["states"]

        # For now, assume states contain all modalities concatenated
        # In practice, we'd need to parse the observation space properly
        # This is a simplified implementation

        # Placeholder: return simple policy output
        # In full implementation, this would process each modality
        x = torch.zeros(states.shape[0], 128, device=states.device)  # Placeholder
        return self.mean_layer(x), self.log_std_parameter, {}


class ValuePPOVision(DeterministicMixin, Model):
    """Enhanced PPO Value function with vision processing."""
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)

        # Same architecture as policy but for value estimation
        self.vision_encoder = VisionEncoder(image_size=(80, 60))
        self.heightmap_encoder = HeightmapCNN(heightmap_size=8)

        self.lidar_encoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.ELU()
        )

        self.pose_encoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.ELU(),
            nn.Linear(32, 16),
            nn.ELU()
        )

        vision_dim = 64
        heightmap_dim = 64
        lidar_dim = 32
        pose_dim = 16
        total_dim = vision_dim + heightmap_dim + lidar_dim + pose_dim

        self.fusion_net = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1)  # Single value output
        )

    def compute(self, inputs, role):
        # Same placeholder implementation as policy
        states = inputs["states"]
        x = torch.zeros(states.shape[0], 128, device=states.device)
        return self.fusion_net(x), {}


class CriticTD3(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)

        input_size = self.num_observations + self.num_actions

        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1)
        )

    def compute(self, inputs, role):
        x = torch.cat([inputs["states"], inputs["taken_actions"]], dim=-1)
        return self.net(x), {}


