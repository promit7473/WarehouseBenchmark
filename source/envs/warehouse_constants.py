"""
Centralized warehouse environment constants.
All warehouse dimensions, bounds, and physical parameters are defined here.
"""

# ============================================================================
# WAREHOUSE BOUNDS - Measured from full_warehouse.usd (WALLS ONLY)
# ============================================================================
# IMPORTANT: These are WALL bounds, not roof bounds (roof is larger but robots can't reach it)

# Raw warehouse bounds (wall-to-wall) - excludes roof overhang
WAREHOUSE_MIN_X = -26.30  # Western wall boundary (meters)
WAREHOUSE_MAX_X = 5.50    # Eastern wall boundary (meters)
WAREHOUSE_MIN_Y = -23.40  # Southern wall boundary (meters)
WAREHOUSE_MAX_Y = 30.60   # Northern wall boundary (meters)

# Calculated dimensions
WAREHOUSE_WIDTH_M = WAREHOUSE_MAX_X - WAREHOUSE_MIN_X   # 31.8m
WAREHOUSE_HEIGHT_M = WAREHOUSE_MAX_Y - WAREHOUSE_MIN_Y  # 54.0m
WAREHOUSE_AREA_M2 = WAREHOUSE_WIDTH_M * WAREHOUSE_HEIGHT_M  # ~1717 m²

# Warehouse bounds tuple (min_x, min_y, max_x, max_y)
WAREHOUSE_BOUNDS = (WAREHOUSE_MIN_X, WAREHOUSE_MIN_Y, WAREHOUSE_MAX_X, WAREHOUSE_MAX_Y)

# ============================================================================
# NAVIGABLE AREA - Warehouse bounds with safety margins
# ============================================================================
# Safe navigation area with margins from walls to avoid collisions

WAREHOUSE_WALL_MARGIN_M = 2.0  # Safety margin from walls (meters)

# Navigable area (with safety margins from walls)
WAREHOUSE_NAVIGABLE_X_MIN = WAREHOUSE_MIN_X + WAREHOUSE_WALL_MARGIN_M  # -24.3m
WAREHOUSE_NAVIGABLE_X_MAX = WAREHOUSE_MAX_X - WAREHOUSE_WALL_MARGIN_M  # 3.5m
WAREHOUSE_NAVIGABLE_Y_MIN = WAREHOUSE_MIN_Y + WAREHOUSE_WALL_MARGIN_M  # -21.4m
WAREHOUSE_NAVIGABLE_Y_MAX = WAREHOUSE_MAX_Y - WAREHOUSE_WALL_MARGIN_M  # 28.6m

# Navigable bounds tuple (min_x, min_y, max_x, max_y)
WAREHOUSE_NAVIGABLE_BOUNDS = (
    WAREHOUSE_NAVIGABLE_X_MIN,
    WAREHOUSE_NAVIGABLE_Y_MIN,
    WAREHOUSE_NAVIGABLE_X_MAX,
    WAREHOUSE_NAVIGABLE_Y_MAX
)

# Navigable dimensions
WAREHOUSE_NAVIGABLE_WIDTH_M = WAREHOUSE_NAVIGABLE_X_MAX - WAREHOUSE_NAVIGABLE_X_MIN   # 27.8m
WAREHOUSE_NAVIGABLE_HEIGHT_M = WAREHOUSE_NAVIGABLE_Y_MAX - WAREHOUSE_NAVIGABLE_Y_MIN  # 50.0m

# ============================================================================
# WAREHOUSE LAYOUT CONSTANTS
# ============================================================================

WAREHOUSE_AISLE_WIDTH_M = 3.5  # Standard warehouse aisle width (meters)
WAREHOUSE_CENTER_X = (WAREHOUSE_MIN_X + WAREHOUSE_MAX_X) / 2  # -10.0m
WAREHOUSE_CENTER_Y = (WAREHOUSE_MIN_Y + WAREHOUSE_MAX_Y) / 2  # -4.0m

# ============================================================================
# ROBOT SPAWN CONSTANTS
# ============================================================================

ROBOT_SPAWN_HEIGHT_M = 0.5  # Height above floor for robot spawn (meters)
ROBOT_SPAWN_SAFETY_MARGIN_M = 1.0  # Minimum distance from obstacles when spawning (meters)

# ============================================================================
# WAYPOINT CONFIGURATION
# ============================================================================

WAYPOINT_SUCCESS_THRESHOLD_M = 1.5  # Distance threshold to consider waypoint reached (meters)
WAYPOINT_MIN_OBSTACLE_DISTANCE_M = 2.0  # Minimum distance waypoints should be from obstacles (meters)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def is_within_bounds(x: float, y: float, use_navigable: bool = True) -> bool:
    """
    Check if a position (x, y) is within warehouse bounds.

    Args:
        x: X coordinate (meters)
        y: Y coordinate (meters)
        use_navigable: If True, use navigable bounds (with margins), else use raw bounds

    Returns:
        True if position is within bounds, False otherwise
    """
    if use_navigable:
        return (WAREHOUSE_NAVIGABLE_X_MIN <= x <= WAREHOUSE_NAVIGABLE_X_MAX and
                WAREHOUSE_NAVIGABLE_Y_MIN <= y <= WAREHOUSE_NAVIGABLE_Y_MAX)
    else:
        return (WAREHOUSE_MIN_X <= x <= WAREHOUSE_MAX_X and
                WAREHOUSE_MIN_Y <= y <= WAREHOUSE_MAX_Y)


def get_warehouse_info() -> dict:
    """
    Get comprehensive warehouse information as a dictionary.

    Returns:
        Dictionary with all warehouse constants
    """
    return {
        "raw_bounds": {
            "min_x": WAREHOUSE_MIN_X,
            "max_x": WAREHOUSE_MAX_X,
            "min_y": WAREHOUSE_MIN_Y,
            "max_y": WAREHOUSE_MAX_Y,
            "width_m": WAREHOUSE_WIDTH_M,
            "height_m": WAREHOUSE_HEIGHT_M,
            "area_m2": WAREHOUSE_AREA_M2,
        },
        "navigable_bounds": {
            "min_x": WAREHOUSE_NAVIGABLE_X_MIN,
            "max_x": WAREHOUSE_NAVIGABLE_X_MAX,
            "min_y": WAREHOUSE_NAVIGABLE_Y_MIN,
            "max_y": WAREHOUSE_NAVIGABLE_Y_MAX,
            "width_m": WAREHOUSE_NAVIGABLE_WIDTH_M,
            "height_m": WAREHOUSE_NAVIGABLE_HEIGHT_M,
            "wall_margin_m": WAREHOUSE_WALL_MARGIN_M,
        },
        "layout": {
            "aisle_width_m": WAREHOUSE_AISLE_WIDTH_M,
            "center_x": WAREHOUSE_CENTER_X,
            "center_y": WAREHOUSE_CENTER_Y,
        },
        "robot": {
            "spawn_height_m": ROBOT_SPAWN_HEIGHT_M,
            "spawn_safety_margin_m": ROBOT_SPAWN_SAFETY_MARGIN_M,
        },
        "waypoints": {
            "success_threshold_m": WAYPOINT_SUCCESS_THRESHOLD_M,
            "min_obstacle_distance_m": WAYPOINT_MIN_OBSTACLE_DISTANCE_M,
        }
    }
