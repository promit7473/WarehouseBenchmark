"""Path validation utilities for secure file operations."""

import os
from pathlib import Path
from typing import Union, Optional


class PathValidationError(Exception):
    """Raised when path validation fails."""
    pass


def validate_path(
    path: Union[str, Path],
    base_dir: Optional[Union[str, Path]] = None,
    must_exist: bool = False,
    must_be_file: bool = False,
    must_be_dir: bool = False,
    allowed_extensions: Optional[list] = None
) -> Path:
    """
    Validate a file/directory path for security and correctness.

    Args:
        path: Path to validate
        base_dir: If provided, ensure path is within this directory
        must_exist: If True, path must exist
        must_be_file: If True, path must be a file
        must_be_dir: If True, path must be a directory
        allowed_extensions: If provided, file extension must be in this list

    Returns:
        Path: Validated absolute path

    Raises:
        PathValidationError: If validation fails

    Examples:
        >>> validate_path("checkpoint.pt", must_exist=True, allowed_extensions=[".pt"])
        >>> validate_path("runs/ppo/checkpoint.pt", base_dir="runs")
    """
    if not path:
        raise PathValidationError("Path cannot be empty")

    # Convert to Path object
    path = Path(path)

    # Resolve to absolute path
    try:
        abs_path = path.resolve()
    except (OSError, RuntimeError) as e:
        raise PathValidationError(f"Invalid path: {e}")

    # Check for path traversal attempts
    if ".." in str(path):
        # Verify the resolved path is safe
        if base_dir:
            base_abs = Path(base_dir).resolve()
            try:
                abs_path.relative_to(base_abs)
            except ValueError:
                raise PathValidationError(
                    f"Path {path} attempts to escape base directory {base_dir}"
                )

    # Validate base directory constraint
    if base_dir:
        base_abs = Path(base_dir).resolve()
        try:
            abs_path.relative_to(base_abs)
        except ValueError:
            raise PathValidationError(
                f"Path {abs_path} is not within allowed directory {base_abs}"
            )

    # Existence checks
    if must_exist and not abs_path.exists():
        raise PathValidationError(f"Path does not exist: {abs_path}")

    if abs_path.exists():
        if must_be_file and not abs_path.is_file():
            raise PathValidationError(f"Path is not a file: {abs_path}")
        if must_be_dir and not abs_path.is_dir():
            raise PathValidationError(f"Path is not a directory: {abs_path}")

    # Extension validation
    if allowed_extensions and abs_path.suffix.lower() not in [ext.lower() for ext in allowed_extensions]:
        raise PathValidationError(
            f"Invalid file extension {abs_path.suffix}. "
            f"Allowed: {allowed_extensions}"
        )

    return abs_path


def validate_checkpoint_path(checkpoint_path: Union[str, Path]) -> Path:
    """
    Validate a checkpoint path specifically.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Path: Validated checkpoint path

    Raises:
        PathValidationError: If validation fails
    """
    return validate_path(
        checkpoint_path,
        must_exist=True,
        must_be_file=True,
        allowed_extensions=[".pt", ".pth"]
    )


def validate_output_path(output_path: Union[str, Path], base_dir: Union[str, Path] = "results") -> Path:
    """
    Validate an output path for writing results.

    Args:
        output_path: Path for output
        base_dir: Base directory for outputs (default: "results")

    Returns:
        Path: Validated output path

    Raises:
        PathValidationError: If validation fails
    """
    # Ensure output is within results directory for safety
    abs_path = Path(output_path).resolve()
    base_abs = Path(base_dir).resolve()

    # Create base directory if it doesn't exist
    base_abs.mkdir(parents=True, exist_ok=True)

    # Check if path is within base directory
    try:
        abs_path.relative_to(base_abs)
    except ValueError:
        # If not within base_dir, create within it
        abs_path = base_abs / abs_path.name

    return abs_path
