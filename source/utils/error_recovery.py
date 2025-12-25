#!/usr/bin/env python3
"""
Error Recovery and Resilience Utilities for Warehouse Benchmark

Provides robust error handling, retry mechanisms, and graceful degradation.
"""

import time
import logging
from typing import Callable, Any, Optional, Dict, List
from functools import wraps
import traceback

class TrainingResilienceManager:
    """Manages training resilience and error recovery."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.failure_count = 0
        self.max_failures = 5
        self.backoff_factor = 2.0
        self.max_backoff = 60.0  # seconds

    def retry_with_backoff(self, func: Callable, max_attempts: int = 3,
                          backoff_factor: float = 2.0, max_backoff: float = 30.0):
        """
        Retry a function with exponential backoff.

        Args:
            func: Function to retry
            max_attempts: Maximum number of attempts
            backoff_factor: Exponential backoff multiplier
            max_backoff: Maximum backoff time in seconds

        Returns:
            Function result or raises last exception
        """
        last_exception = None

        for attempt in range(max_attempts):
            try:
                return func()
            except Exception as e:
                last_exception = e
                wait_time = min(backoff_factor ** attempt, max_backoff)

                if attempt < max_attempts - 1:
                    self.logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {wait_time:.1f} seconds..."
                    )
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"All {max_attempts} attempts failed. Last error: {e}")

        raise last_exception

    def graceful_checkpoint_save(self, agent, checkpoint_path: str) -> bool:
        """
        Save checkpoint with error handling and validation.

        Args:
            agent: Agent to save
            checkpoint_path: Path to save checkpoint

        Returns:
            True if successful, False otherwise
        """
        try:
            # Attempt to save
            agent.save(checkpoint_path)
            self.logger.info(f"Checkpoint saved successfully: {checkpoint_path}")

            # Validate checkpoint exists and is readable
            import os
            if os.path.exists(checkpoint_path):
                file_size = os.path.getsize(checkpoint_path)
                self.logger.debug(f"Checkpoint file size: {file_size} bytes")
                return True
            else:
                self.logger.error(f"Checkpoint file not found after save: {checkpoint_path}")
                return False

        except Exception as e:
            self.logger.error(f"Failed to save checkpoint {checkpoint_path}: {e}")
            return False

    def safe_environment_reset(self, env) -> bool:
        """
        Safely reset environment with error handling.

        Args:
            env: Environment to reset

        Returns:
            True if successful, False otherwise
        """
        try:
            obs, info = env.reset()
            self.logger.debug("Environment reset successful")
            return True
        except Exception as e:
            self.logger.error(f"Environment reset failed: {e}")
            return False

    def monitor_training_stability(self, metrics_history: List[Dict]) -> Dict[str, Any]:
        """
        Monitor training stability and detect issues.

        Args:
            metrics_history: List of recent training metrics

        Returns:
            Stability analysis results
        """
        if len(metrics_history) < 5:
            return {"status": "insufficient_data", "issues": []}

        recent_metrics = metrics_history[-10:]  # Last 10 measurements

        issues = []
        analysis = {"status": "stable", "issues": issues}

        # Check for reward instability
        if 'reward' in recent_metrics[0]:
            rewards = [m.get('reward', 0) for m in recent_metrics]
            reward_std = np.std(rewards)
            reward_mean = np.mean(rewards)

            # High variance relative to mean
            if reward_std > abs(reward_mean) * 0.5:
                issues.append("High reward variance detected")
                analysis["status"] = "unstable"

        # Check for NaN values
        for i, metrics in enumerate(recent_metrics):
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and (np.isnan(value) or np.isinf(value)):
                    issues.append(f"NaN/Inf detected in {key} at step {i}")
                    analysis["status"] = "critical"

        # Check for loss spikes
        if 'loss' in recent_metrics[0]:
            losses = [m.get('loss', 0) for m in recent_metrics]
            recent_avg = np.mean(losses[-3:])
            overall_avg = np.mean(losses)

            if recent_avg > overall_avg * 2:
                issues.append("Recent loss spike detected")
                analysis["status"] = "warning"

        return analysis

# Global resilience manager
_resilience_manager = None

def get_resilience_manager() -> TrainingResilienceManager:
    """Get the global resilience manager instance."""
    global _resilience_manager
    if _resilience_manager is None:
        _resilience_manager = TrainingResilienceManager()
    return _resilience_manager

# Decorators for resilient operations
def resilient_operation(max_attempts: int = 3, backoff_factor: float = 2.0):
    """Decorator for resilient operations with retry logic."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            manager = get_resilience_manager()
            return manager.retry_with_backoff(
                lambda: func(*args, **kwargs),
                max_attempts=max_attempts,
                backoff_factor=backoff_factor
            )
        return wrapper
    return decorator

def safe_checkpoint_operation(operation_type: str = "save"):
    """Decorator for safe checkpoint operations."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            manager = get_resilience_manager()
            logger = logging.getLogger(__name__)

            try:
                result = func(*args, **kwargs)
                logger.debug(f"Checkpoint {operation_type} operation successful")
                return result
            except Exception as e:
                logger.error(f"Checkpoint {operation_type} operation failed: {e}")
                # Could implement recovery strategies here
                raise e
        return wrapper
    return decorator

# Training stability monitoring
class TrainingStabilityMonitor:
    """Monitors training stability and provides early warning signals."""

    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.metrics_history = []
        self.logger = logging.getLogger(__name__)

    def add_metrics(self, metrics: Dict[str, Any]):
        """Add new metrics to the monitoring history."""
        self.metrics_history.append(metrics)

        # Keep only recent history
        if len(self.metrics_history) > self.window_size:
            self.metrics_history = self.metrics_history[-self.window_size:]

    def check_stability(self) -> Dict[str, Any]:
        """Check training stability and return analysis."""
        if len(self.metrics_history) < 10:
            return {"status": "warming_up", "confidence": 0.0}

        manager = get_resilience_manager()
        analysis = manager.monitor_training_stability(self.metrics_history)

        # Add confidence score
        analysis["confidence"] = min(1.0, len(self.metrics_history) / self.window_size)

        return analysis

    def get_stability_score(self) -> float:
        """
        Get a numerical stability score (0.0 = unstable, 1.0 = stable).

        Returns:
            Stability score between 0 and 1
        """
        analysis = self.check_stability()

        status_scores = {
            "critical": 0.0,
            "unstable": 0.3,
            "warning": 0.7,
            "stable": 1.0,
            "warming_up": 0.5
        }

        base_score = status_scores.get(analysis["status"], 0.5)
        confidence = analysis.get("confidence", 1.0)

        return base_score * confidence

# Graceful degradation utilities
class GracefulDegradationManager:
    """Manages graceful degradation when resources are constrained."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.degradation_level = 0
        self.max_degradation_level = 3

    def assess_system_health(self) -> Dict[str, Any]:
        """Assess overall system health."""
        health_status = {
            "memory_ok": True,
            "gpu_ok": True,
            "disk_ok": True,
            "network_ok": True
        }

        # Check memory
        try:
            import psutil
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                health_status["memory_ok"] = False
        except ImportError:
            pass

        # Check GPU
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                if gpu_memory > 0.9:
                    health_status["gpu_ok"] = False
        except Exception:
            health_status["gpu_ok"] = False

        return health_status

    def apply_degradation(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply graceful degradation to configuration based on system health.

        Args:
            config: Original configuration

        Returns:
            Degraded configuration for stability
        """
        health = self.assess_system_health()
        degraded_config = config.copy()

        if not health["memory_ok"] or not health["gpu_ok"]:
            self.degradation_level = min(self.degradation_level + 1, self.max_degradation_level)

            # Apply degradation strategies
            if self.degradation_level >= 1:
                # Reduce batch size
                if 'batch_size' in degraded_config.get('agent', {}):
                    degraded_config['agent']['batch_size'] = max(
                        degraded_config['agent']['batch_size'] // 2, 16
                    )

            if self.degradation_level >= 2:
                # Reduce learning rate
                if 'learning_rate' in degraded_config.get('agent', {}):
                    degraded_config['agent']['learning_rate'] *= 0.5

            if self.degradation_level >= 3:
                # Reduce environments
                if 'num_envs' in degraded_config.get('env', {}):
                    degraded_config['env']['num_envs'] = max(
                        degraded_config['env']['num_envs'] // 2, 1
                    )

            self.logger.warning(
                f"Applied degradation level {self.degradation_level} due to system constraints"
            )

        return degraded_config

    def can_recover(self) -> bool:
        """Check if system can recover from degradation."""
        health = self.assess_system_health()

        if all(health.values()):
            self.degradation_level = max(0, self.degradation_level - 1)
            if self.degradation_level == 0:
                self.logger.info("System health recovered, degradation lifted")
            return True

        return False

# Convenience functions
def retry_on_failure(func: Callable, max_attempts: int = 3):
    """Convenience function for retrying operations."""
    manager = get_resilience_manager()
    return manager.retry_with_backoff(func, max_attempts)

def check_training_stability(metrics_history: List[Dict]) -> Dict[str, Any]:
    """Check training stability from metrics history."""
    manager = get_resilience_manager()
    return manager.monitor_training_stability(metrics_history)