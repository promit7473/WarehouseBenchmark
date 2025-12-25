"""Utility functions for warehouse environment."""

import torch


def quaternion_to_yaw(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Extract yaw angle from quaternion.

    Args:
        quaternion: Quaternion tensor of shape (..., 4) in format [qw, qx, qy, qz]
                   or full state tensor where quaternion starts at index 3

    Returns:
        Yaw angle in radians
    """
    # If quaternion is part of full state [x, y, z, qw, qx, qy, qz, ...]
    # Extract the quaternion part
    if quaternion.shape[-1] > 4:
        quat = quaternion[..., 3:7]  # Extract [qw, qx, qy, qz]
    else:
        quat = quaternion

    qw = quat[..., 0]
    qx = quat[..., 1]
    qy = quat[..., 2]
    qz = quat[..., 3]

    # Convert quaternion to yaw (rotation around Z-axis)
    # yaw = atan2(2(qw*qz + qx*qy), 1 - 2(qy^2 + qz^2))
    yaw = torch.atan2(2.0 * (qw * qz + qx * qy),
                      1.0 - 2.0 * (qy * qy + qz * qz))

    return yaw
