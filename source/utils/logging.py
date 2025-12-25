#!/usr/bin/env python3
"""
Warehouse Benchmark Logging System

Provides structured logging for training, evaluation, and analysis.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

class WarehouseLogger:
    """Centralized logging system for the warehouse benchmark."""

    def __init__(self, name: str = "warehouse_benchmark", log_level: str = "INFO"):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))

        # Remove any existing handlers
        self.logger.handlers.clear()

        # Create formatters
        self.file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        self.console_formatter = logging.Formatter(
            '%(levelname)s - %(message)s'
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(self.console_formatter)
        self.logger.addHandler(console_handler)

        # File handler (will be set when log directory is known)
        self.file_handler = None

    def set_log_file(self, log_dir: str, filename: str = "warehouse_benchmark.log"):
        """Set up file logging."""
        if self.file_handler:
            self.logger.removeHandler(self.file_handler)

        log_path = Path(log_dir) / filename
        log_path.parent.mkdir(parents=True, exist_ok=True)

        self.file_handler = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=10*1024*1024, backupCount=5
        )
        self.file_handler.setLevel(logging.DEBUG)
        self.file_handler.setFormatter(self.file_formatter)
        self.logger.addHandler(self.file_handler)

    def get_logger(self) -> logging.Logger:
        """Get the configured logger."""
        return self.logger

# Global logger instance
_logger_instance: Optional[WarehouseLogger] = None

def get_logger(name: str = "warehouse_benchmark", log_level: str = "INFO") -> logging.Logger:
    """Get or create the global logger instance."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = WarehouseLogger(name, log_level)
    return _logger_instance.get_logger()

def setup_experiment_logging(experiment_dir: str, experiment_name: str):
    """Set up logging for a specific experiment."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = WarehouseLogger()

    log_dir = Path(experiment_dir) / "logs"
    _logger_instance.set_log_file(str(log_dir), f"{experiment_name}.log")

    logger = _logger_instance.get_logger()
    logger.info(f"Experiment logging initialized: {experiment_name}")
    logger.info(f"Log directory: {log_dir}")

    return logger

# Convenience functions
def log_training_start(algorithm: str, config: dict):
    """Log training start with configuration."""
    logger = get_logger()
    logger.info(f"🚀 Starting {algorithm} training")
    logger.info(f"Configuration: {config}")

def log_training_progress(epoch: int, total_epochs: int, reward: float, loss: float = None):
    """Log training progress."""
    logger = get_logger()
    progress = f"Epoch {epoch}/{total_epochs} | Reward: {reward:.4f}"
    if loss is not None:
        progress += f" | Loss: {loss:.4f}"
    logger.info(progress)

def log_evaluation_results(algorithm: str, metrics: dict):
    """Log evaluation results."""
    logger = get_logger()
    logger.info(f"📊 {algorithm} Evaluation Results:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value}")

def log_error(message: str, exc_info: bool = True):
    """Log error with optional exception info."""
    logger = get_logger()
    logger.error(message, exc_info=exc_info)

def log_warning(message: str):
    """Log warning message."""
    logger = get_logger()
    logger.warning(message)

def log_info(message: str):
    """Log info message."""
    logger = get_logger()
    logger.info(message)

def log_debug(message: str):
    """Log debug message."""
    logger = get_logger()
    logger.debug(message)