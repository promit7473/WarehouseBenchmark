# source/envs/config/warehouse_dynamic_cfg.py

from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp_utils

@configclass
class CommandsCfg:
    """Enhanced command terms for warehouse waypoint navigation."""
    # Enhanced waypoint navigation with validation and dynamic resampling
    waypoint_nav = mdp_utils.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=True,  # Use direction towards target
        resampling_time_range=(1000.0, 1000.0),  # Disable automatic resampling, handle manually
        debug_vis=True,  # Enable waypoint arrows for visualization
        ranges=mdp_utils.UniformPose2dCommandCfg.Ranges(
            pos_x=(-25.0, 25.0),  # Increased range for more challenging navigation
                pos_y=(-25.0, 25.0),
                heading=(-3.14, 3.14),
        ),
    )