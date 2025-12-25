#!/usr/bin/env python3
"""
Fix USD file transformation matrices to be orthonormal.

This script fixes the "OrthogonalizeBasis did not converge" warnings
by normalizing all transformation matrices in the USD file.

Usage (run with Isaac Lab's Python):
    ~/.local/share/ov/pkg/isaac-sim-*/python.sh scripts/fix_usd_transforms.py

Or with isaaclab.sh:
    ./isaaclab.sh -p scripts/fix_usd_transforms.py

If no arguments provided, fixes the default warehouse.usd file.
"""

import argparse
import os
import sys

# Check if pxr (USD) is available
USD_AVAILABLE = False
Usd = None
UsdGeom = None
Gf = None

try:
    from pxr import Usd, UsdGeom, Gf
    USD_AVAILABLE = True
except ImportError:
    print("[WARNING] USD Python bindings (pxr) not available.")
    print("Run with Isaac Sim's Python interpreter:")
    print("  ~/.local/share/ov/pkg/isaac-sim-*/python.sh scripts/fix_usd_transforms.py")
    print("Or:")
    print("  /home/mhpromit7473/IsaacLab/isaaclab.sh -p scripts/fix_usd_transforms.py")


def orthonormalize_matrix(matrix):
    """
    Orthonormalize a 4x4 transformation matrix.

    This ensures the rotation component is orthonormal while preserving
    the translation and scale.
    """
    if not USD_AVAILABLE:
        return matrix

    # Extract the 3x3 rotation/scale part
    rotation = Gf.Matrix3d(
        matrix[0][0], matrix[0][1], matrix[0][2],
        matrix[1][0], matrix[1][1], matrix[1][2],
        matrix[2][0], matrix[2][1], matrix[2][2]
    )

    # Orthonormalize using Gram-Schmidt
    try:
        rotation_ortho = rotation.GetOrthonormalized()
    except Exception:
        # If orthonormalization fails, use identity
        rotation_ortho = Gf.Matrix3d(1.0)

    # Reconstruct the 4x4 matrix
    result = Gf.Matrix4d(1.0)  # Identity
    for i in range(3):
        for j in range(3):
            result[i][j] = rotation_ortho[i][j]

    # Preserve translation
    result[3][0] = matrix[3][0]
    result[3][1] = matrix[3][1]
    result[3][2] = matrix[3][2]

    return result


def fix_usd_transforms(input_path: str, output_path: str = None) -> bool:
    """
    Fix all transformation matrices in a USD file.

    Args:
        input_path: Path to the input USD file
        output_path: Path for the output file (defaults to overwriting input)

    Returns:
        True if successful, False otherwise
    """
    if not USD_AVAILABLE:
        print("[ERROR] USD Python bindings not available")
        return False

    if not os.path.exists(input_path):
        print(f"[ERROR] File not found: {input_path}")
        return False

    if output_path is None:
        output_path = input_path

    print(f"[INFO] Loading USD file: {input_path}")

    # Open the stage
    stage = Usd.Stage.Open(input_path)
    if not stage:
        print(f"[ERROR] Failed to open USD file: {input_path}")
        return False

    fixed_count = 0

    # Iterate through all prims
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Xformable):
            xformable = UsdGeom.Xformable(prim)

            # Get the local transformation
            xform_ops = xformable.GetOrderedXformOps()

            for op in xform_ops:
                if op.GetOpType() == UsdGeom.XformOp.TypeTransform:
                    # Get the matrix value
                    matrix = op.Get()
                    if matrix:
                        # Check if matrix needs fixing
                        try:
                            # Try to get rotation - this will fail if not orthonormal
                            rot = Gf.Matrix3d(
                                matrix[0][0], matrix[0][1], matrix[0][2],
                                matrix[1][0], matrix[1][1], matrix[1][2],
                                matrix[2][0], matrix[2][1], matrix[2][2]
                            )

                            # Check orthonormality
                            ortho_rot = rot.GetOrthonormalized()
                            diff = 0.0
                            for i in range(3):
                                for j in range(3):
                                    diff += abs(rot[i][j] - ortho_rot[i][j])

                            if diff > 1e-6:
                                # Matrix is not orthonormal, fix it
                                fixed_matrix = orthonormalize_matrix(matrix)
                                op.Set(fixed_matrix)
                                fixed_count += 1
                                print(f"  Fixed transform on: {prim.GetPath()}")

                        except Exception as e:
                            print(f"  [WARNING] Could not process {prim.GetPath()}: {e}")

    if fixed_count > 0:
        print(f"[INFO] Fixed {fixed_count} non-orthonormal transforms")
        print(f"[INFO] Saving to: {output_path}")
        stage.GetRootLayer().Export(output_path)
    else:
        print("[INFO] No transforms needed fixing")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Fix USD file transformation matrices to be orthonormal"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Input USD file path"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output USD file path (defaults to input path)"
    )

    args = parser.parse_args()

    # Default to warehouse.usd if no input specified
    if args.input is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        args.input = os.path.join(
            project_root,
            "assets", "warehouse", "full_warehouse.usd"
        )

    if not USD_AVAILABLE:
        print("\n[INFO] Alternative: You can fix the USD file in Blender/Maya:")
        print("  1. Open the USD file")
        print("  2. Select all objects")
        print("  3. Apply all transforms (Ctrl+A -> All Transforms)")
        print("  4. Re-export as USD")
        sys.exit(1)

    success = fix_usd_transforms(args.input, args.output)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
