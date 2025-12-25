"""Get USD file bounds - run with isaaclab.sh"""
from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--usd-path", type=str, required=True, help="Path to USD file")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from pxr import Usd, UsdGeom
import re

usd_path = args.usd_path
stage = Usd.Stage.Open(usd_path)

root = stage.GetPseudoRoot()
bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'])

# Smart boundary detection: find walls/floors, exclude roofs
wall_bounds = []
floor_bounds = []

def traverse_prims(prim):
    """Recursively traverse prims to find walls and floors."""
    if not prim.IsValid():
        return

    # Check if this prim has geometry (Mesh)
    if prim.IsA(UsdGeom.Mesh):
        prim_name = prim.GetName()
        prim_path = str(prim.GetPath())

        # Get bounding box
        bbox = bbox_cache.ComputeWorldBound(prim)
        if bbox:
            range_box = bbox.ComputeAlignedRange()
            if range_box:
                min_pt = range_box.GetMin()
                max_pt = range_box.GetMax()

                # Classify prims based on name/path and dimensions
                is_wall = False
                is_floor = False
                is_roof = False

                # Check for roof indicators
                if any(keyword in prim_name.lower() or keyword in prim_path.lower()
                       for keyword in ['roof', 'ceiling', 'top']):
                    is_roof = True

                # Check for wall indicators
                elif any(keyword in prim_name.lower() or keyword in prim_path.lower()
                        for keyword in ['wall', 'side', 'boundary']):
                    is_wall = True

                # Check for floor indicators or very thin horizontal geometry
                elif any(keyword in prim_name.lower() or keyword in prim_path.lower()
                        for keyword in ['floor', 'ground', 'base']):
                    is_floor = True

                # Heuristic: very thin geometry in Z direction is likely floor/walls
                thickness = max_pt[2] - min_pt[2]
                if thickness < 0.5:  # Less than 50cm thick
                    # Check if it's more horizontal (floor) or vertical (wall)
                    width_x = max_pt[0] - min_pt[0]
                    width_y = max_pt[1] - min_pt[1]

                    if width_x > width_y * 2 or width_y > width_x * 2:  # Elongated
                        if not is_roof:
                            is_wall = True
                    elif thickness < 0.1:  # Very thin horizontal
                        is_floor = True

                # Store bounds
                if is_wall:
                    wall_bounds.append((min_pt, max_pt))
                    print(f"Found wall: {prim_path} -> X:[{min_pt[0]:.1f}, {max_pt[0]:.1f}] Y:[{min_pt[1]:.1f}, {max_pt[1]:.1f}] Z:[{min_pt[2]:.1f}, {max_pt[2]:.1f}]")
                elif is_floor:
                    floor_bounds.append((min_pt, max_pt))
                    print(f"Found floor: {prim_path} -> X:[{min_pt[0]:.1f}, {max_pt[0]:.1f}] Y:[{min_pt[1]:.1f}, {max_pt[1]:.1f}] Z:[{min_pt[2]:.1f}, {max_pt[2]:.1f}]")
                elif is_roof:
                    print(f"Skipping roof: {prim_path}")

    # Recurse on children
    for child in prim.GetChildren():
        traverse_prims(child)

print("🔍 Analyzing warehouse geometry...")
traverse_prims(root)

# Compute bounds from walls and floors only
if wall_bounds or floor_bounds:
    all_bounds = wall_bounds + floor_bounds

    # Find min/max across all wall and floor bounds
    global_min = [float('inf')] * 3
    global_max = [float('-inf')] * 3

    for min_pt, max_pt in all_bounds:
        for i in range(3):
            global_min[i] = min(global_min[i], min_pt[i])
            global_max[i] = max(global_max[i], max_pt[i])

    min_pt = global_min
    max_pt = global_max
else:
    # Fallback to full scene bounds if no walls/floors found
    print("⚠️  No walls/floors detected, using full scene bounds")
    world_bbox = bbox_cache.ComputeWorldBound(root)
    world_range = world_bbox.ComputeAlignedRange()
    min_pt = world_range.GetMin()
    max_pt = world_range.GetMax()

# Print results
print("\n" + "=" * 80)
print(f"WAREHOUSE BOUNDS: {usd_path}")
print("=" * 80)
print(f"MIN: X={min_pt[0]:.2f}, Y={min_pt[1]:.2f}, Z={min_pt[2]:.2f}")
print(f"MAX: X={max_pt[0]:.2f}, Y={max_pt[1]:.2f}, Z={max_pt[2]:.2f}")
print(f"SIZE: X={max_pt[0]-min_pt[0]:.2f}m, Y={max_pt[1]-min_pt[1]:.2f}m, Z={max_pt[2]-min_pt[2]:.2f}m")
print(f"CENTER: X={(min_pt[0]+max_pt[0])/2:.2f}, Y={(min_pt[1]+max_pt[1])/2:.2f}")
print("=" * 80)
print("\nRecommended spawn bounds (with 2m wall margin):")
print(f"  WAREHOUSE_MIN_X = {min_pt[0]:.2f}")
print(f"  WAREHOUSE_MAX_X = {max_pt[0]:.2f}")
print(f"  WAREHOUSE_MIN_Y = {min_pt[1]:.2f}")
print(f"  WAREHOUSE_MAX_Y = {max_pt[1]:.2f}")
print(f"  NAVIGABLE_X_MIN = {min_pt[0]+2.0:.2f}")
print(f"  NAVIGABLE_X_MAX = {max_pt[0]-2.0:.2f}")
print(f"  NAVIGABLE_Y_MIN = {min_pt[1]+2.0:.2f}")
print(f"  NAVIGABLE_Y_MAX = {max_pt[1]-2.0:.2f}")
print("=" * 80)

simulation_app.close()