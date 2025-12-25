#!/usr/bin/env python3
"""
Memory Management Utilities for Warehouse Benchmark

Provides memory monitoring, cleanup, and optimization utilities.
"""

import torch
import gc
import psutil
import os
from typing import Dict, Any
import logging

class MemoryManager:
    """Manages GPU and CPU memory for training stability."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.initial_memory = self.get_memory_usage()

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        memory_info = {}

        # GPU memory (if available)
        if torch.cuda.is_available():
            memory_info['gpu_allocated'] = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_info['gpu_reserved'] = torch.cuda.memory_reserved() / 1024**3    # GB
            memory_info['gpu_max_allocated'] = torch.cuda.max_memory_allocated() / 1024**3

        # CPU memory
        process = psutil.Process(os.getpid())
        memory_info['cpu_rss'] = process.memory_info().rss / 1024**3  # GB
        memory_info['cpu_vms'] = process.memory_info().vms / 1024**3  # GB

        # System memory
        system_memory = psutil.virtual_memory()
        memory_info['system_total'] = system_memory.total / 1024**3
        memory_info['system_available'] = system_memory.available / 1024**3
        memory_info['system_percent'] = system_memory.percent

        return memory_info

    def log_memory_usage(self, prefix: str = ""):
        """Log current memory usage."""
        memory = self.get_memory_usage()

        log_message = f"{prefix}Memory Usage:"
        if 'gpu_allocated' in memory:
            log_message += f" GPU: {memory['gpu_allocated']:.2f}GB allocated, {memory['gpu_reserved']:.2f}GB reserved"
        log_message += f" CPU: {memory['cpu_rss']:.2f}GB RSS, System: {memory['system_percent']:.1f}% used"

        self.logger.info(log_message)

    def cleanup_memory(self):
        """Perform memory cleanup operations."""
        # PyTorch GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Python garbage collection
        gc.collect()

        self.logger.debug("Memory cleanup performed")

    def check_memory_thresholds(self, gpu_threshold: float = 0.9, cpu_threshold: float = 0.8) -> bool:
        """
        Check if memory usage exceeds thresholds.

        Args:
            gpu_threshold: Maximum GPU memory usage (0-1)
            cpu_threshold: Maximum CPU memory usage (0-1)

        Returns:
            True if within limits, False if exceeded
        """
        memory = self.get_memory_usage()
        warnings = []

        if 'gpu_allocated' in memory and 'gpu_reserved' in memory:
            gpu_usage = memory['gpu_reserved'] / (torch.cuda.get_device_properties(0).total_memory / 1024**3)
            if gpu_usage > gpu_threshold:
                warnings.append(f"GPU memory usage {gpu_usage:.2%} exceeds threshold {gpu_threshold:.2%}")

        if memory['system_percent'] / 100 > cpu_threshold:
            warnings.append(f"System memory usage {memory['system_percent']:.1f}% exceeds threshold {cpu_threshold:.1f}%")

        if warnings:
            for warning in warnings:
                self.logger.warning(warning)
            return False

        return True

    def optimize_memory_settings(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize memory-related configuration settings.

        Args:
            config: Original configuration dictionary

        Returns:
            Optimized configuration
        """
        optimized_config = config.copy()

        # Reduce batch sizes if GPU memory is limited
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3

            # Adjust batch size based on GPU memory
            if gpu_memory_gb < 8:  # Low memory GPUs
                if 'batch_size' in optimized_config.get('agent', {}):
                    original_batch = optimized_config['agent']['batch_size']
                    optimized_config['agent']['batch_size'] = min(original_batch, 64)
                    if optimized_config['agent']['batch_size'] != original_batch:
                        self.logger.info(f"Reduced batch size from {original_batch} to {optimized_config['agent']['batch_size']} for low GPU memory")

        # Adjust memory buffer sizes for stability
        if 'memory_size' in optimized_config.get('agent', {}):
            memory_size = optimized_config['agent']['memory_size']
            # Ensure reasonable bounds
            optimized_config['agent']['memory_size'] = max(1000, min(memory_size, 500000))

        return optimized_config

# Global memory manager instance
_memory_manager = None

def get_memory_manager() -> MemoryManager:
    """Get the global memory manager instance."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager

def log_memory_usage(prefix: str = ""):
    """Convenience function to log memory usage."""
    get_memory_manager().log_memory_usage(prefix)

def cleanup_memory():
    """Convenience function to cleanup memory."""
    get_memory_manager().cleanup_memory()

def check_memory_health() -> bool:
    """Check if memory usage is within healthy limits."""
    return get_memory_manager().check_memory_thresholds()

def optimize_config_for_memory(config: Dict[str, Any]) -> Dict[str, Any]:
    """Optimize configuration for available memory."""
    return get_memory_manager().optimize_memory_settings(config)

# Memory monitoring decorator
def monitor_memory(func):
    """Decorator to monitor memory usage of a function."""
    def wrapper(*args, **kwargs):
        manager = get_memory_manager()
        manager.log_memory_usage(f"Before {func.__name__}: ")

        try:
            result = func(*args, **kwargs)
            manager.log_memory_usage(f"After {func.__name__}: ")
            return result
        except Exception as e:
            manager.log_memory_usage(f"Error in {func.__name__}: ")
            raise e
        finally:
            manager.cleanup_memory()

    return wrapper

# Memory-aware training utilities
class MemoryAwareTrainer:
    """Training utilities that adapt to memory constraints."""

    def __init__(self):
        self.memory_manager = get_memory_manager()

    def get_optimal_batch_size(self, base_batch_size: int, num_envs: int) -> int:
        """Calculate optimal batch size based on available memory."""
        if not torch.cuda.is_available():
            return base_batch_size

        # Estimate memory per sample
        memory_per_sample = 0.001  # Rough estimate in GB
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3

        # Reserve 20% for overhead
        available_memory = gpu_memory_gb * 0.8

        # Calculate max batch size
        max_batch = int(available_memory / (memory_per_sample * num_envs))

        optimal_batch = min(base_batch_size, max_batch, 512)  # Cap at 512
        optimal_batch = max(optimal_batch, 16)  # Minimum 16

        if optimal_batch != base_batch_size:
            self.memory_manager.logger.info(
                f"Adjusted batch size from {base_batch_size} to {optimal_batch} "
                f"for {gpu_memory_gb:.1f}GB GPU"
            )

        return optimal_batch

    def should_reduce_frequency(self) -> bool:
        """Check if operations should be performed less frequently due to memory."""
        memory = self.memory_manager.get_memory_usage()

        if 'gpu_reserved' in memory:
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            usage_ratio = memory['gpu_reserved'] / gpu_memory_gb

            # If using more than 80% GPU memory, reduce frequency
            return usage_ratio > 0.8

        return False