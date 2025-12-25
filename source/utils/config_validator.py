#!/usr/bin/env python3
"""
Configuration Validation for Warehouse Benchmark

Validates algorithm configurations and provides helpful error messages.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Tuple
import yaml

class ConfigValidator:
    """Validates configuration files for the warehouse benchmark."""

    def __init__(self):
        self.required_fields = {
            'ppo': ['rollouts', 'learning_epochs', 'mini_batches', 'learning_rate'],
            'ppo_enhanced': ['rollouts', 'learning_epochs', 'mini_batches', 'learning_rate'],
            'sac': ['memory_size', 'learning_rate', 'batch_size'],
            'td3': ['memory_size', 'learning_rate', 'batch_size']
        }

        self.valid_ranges = {
            'learning_rate': (1e-6, 1e-2),
            'batch_size': (16, 2048),
            'memory_size': (1000, 1000000),
            'rollouts': (1, 100),
            'learning_epochs': (1, 50),
            'mini_batches': (1, 20)
        }

    def validate_config(self, config_path: str, algorithm: str) -> Tuple[bool, List[str]]:
        """
        Validate a configuration file.

        Args:
            config_path: Path to the YAML config file
            algorithm: Algorithm name ('ppo', 'sac', 'td3')

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check if file exists
        if not Path(config_path).exists():
            return False, [f"Configuration file not found: {config_path}"]

        try:
            # Load config
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            if config is None:
                return False, ["Configuration file is empty or invalid YAML"]

            # Validate algorithm-specific requirements
            if algorithm.lower() not in self.required_fields:
                return False, [f"Unsupported algorithm: {algorithm}"]

            # Check required fields
            agent_config = config.get('agent', {})
            required = self.required_fields[algorithm.lower()]

            for field in required:
                if field not in agent_config:
                    errors.append(f"Missing required field 'agent.{field}' for {algorithm}")

            # Validate value ranges
            for field, (min_val, max_val) in self.valid_ranges.items():
                if field in agent_config:
                    value = agent_config[field]
                    if isinstance(value, (int, float)):
                        if not (min_val <= value <= max_val):
                            errors.append(
                                f"Field 'agent.{field}' value {value} is outside valid range "
                                f"[{min_val}, {max_val}]"
                            )

            # Algorithm-specific validations
            if algorithm.lower() == 'ppo':
                self._validate_ppo_config(agent_config, errors)
            elif algorithm.lower() == 'sac':
                self._validate_sac_config(agent_config, errors)
            elif algorithm.lower() == 'td3':
                self._validate_td3_config(agent_config, errors)


            # Validate trainer config
            trainer_config = config.get('trainer', {})
            if 'timesteps' in trainer_config:
                timesteps = trainer_config['timesteps']
                if timesteps < 1000:
                    errors.append("Training timesteps too low (< 1000). Consider increasing for meaningful training.")

            return len(errors) == 0, errors

        except yaml.YAMLError as e:
            return False, [f"YAML parsing error: {e}"]
        except Exception as e:
            return False, [f"Configuration validation error: {e}"]

    def _validate_ppo_config(self, config: Dict[str, Any], errors: List[str]):
        """Validate PPO-specific configuration."""
        if 'lambda' in config:
            lambda_val = config['lambda']
            if not (0.8 <= lambda_val <= 1.0):
                errors.append(f"PPO lambda {lambda_val} should typically be between 0.8 and 1.0")

        if 'clip_range' in config:
            clip_range = config['clip_range']
            if not (0.1 <= clip_range <= 0.4):
                errors.append(f"PPO clip_range {clip_range} should typically be between 0.1 and 0.4")

    def _validate_sac_config(self, config: Dict[str, Any], errors: List[str]):
        """Validate SAC-specific configuration."""
        if 'learn_entropy' in config and config['learn_entropy']:
            if 'entropy_learning_rate' not in config:
                errors.append("SAC entropy learning enabled but 'entropy_learning_rate' not specified")

        if 'initial_log_std' in config:
            log_std = config['initial_log_std']
            if log_std < -5 or log_std > 0:
                errors.append(f"SAC initial_log_std {log_std} seems unusual (typical range: -5 to 0)")

    def _validate_td3_config(self, config: Dict[str, Any], errors: List[str]):
        """Validate TD3-specific configuration."""
        if 'policy_delay' in config:
            delay = config['policy_delay']
            if delay < 1:
                errors.append("TD3 policy_delay should be >= 1")

        if 'noise_std' in config:
            noise = config['noise_std']
            if noise > 1.0:
                errors.append(f"TD3 noise_std {noise} seems high (typical < 1.0)")



    def suggest_fixes(self, errors: List[str]) -> Dict[str, str]:
        """
        Provide suggested fixes for common configuration errors.
        """
        suggestions = {}

        for error in errors:
            if "Missing required field" in error:
                field = error.split("'")[1]
                suggestions[field] = f"Add '{field}' to your agent configuration"
            elif "outside valid range" in error:
                field = error.split("'")[1]
                if field in self.valid_ranges:
                    min_val, max_val = self.valid_ranges[field]
                    suggestions[field] = f"Set {field} to a value between {min_val} and {max_val}"
            elif "timesteps too low" in error:
                suggestions['timesteps'] = "Increase timesteps to at least 10000 for meaningful training"

        return suggestions

def validate_and_report(config_path: str, algorithm: str) -> bool:
    """
    Validate configuration and print detailed report.

    Returns:
        True if valid, False otherwise
    """
    validator = ConfigValidator()
    is_valid, errors = validator.validate_config(config_path, algorithm)

    print(f"🔍 Validating {algorithm.upper()} configuration: {config_path}")
    print(f"📊 Result: {'✅ VALID' if is_valid else '❌ INVALID'}")

    if errors:
        print("\n🚨 Issues found:")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")

        suggestions = validator.suggest_fixes(errors)
        if suggestions:
            print("\n💡 Suggested fixes:")
            for field, suggestion in suggestions.items():
                print(f"  • {field}: {suggestion}")

    print()
    return is_valid

def validate_all_configs(configs_dir: str = "./configs") -> bool:
    """
    Validate all configuration files in the configs directory.
    """
    configs_path = Path(configs_dir)
    if not configs_path.exists():
        print(f"❌ Configs directory not found: {configs_dir}")
        return False

    algorithms = ['ppo', 'ppo_enhanced', 'sac', 'td3']
    all_valid = True

    print("🔍 Validating all algorithm configurations...\n")

    for algorithm in algorithms:
        config_file = configs_path / f"{algorithm}_warehouse.yaml"
        if config_file.exists():
            valid = validate_and_report(str(config_file), algorithm)
            all_valid = all_valid and valid
        else:
            print(f"⚠️  Configuration file not found: {config_file}")
            all_valid = False

    return all_valid

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate warehouse benchmark configurations")
    parser.add_argument("--config", help="Path to specific config file")
    parser.add_argument("--algorithm", choices=['ppo', 'ppo_enhanced', 'sac', 'td3'], help="Algorithm for validation")
    parser.add_argument("--all", action="store_true", help="Validate all configurations")

    args = parser.parse_args()

    if args.all:
        success = validate_all_configs()
    elif args.config and args.algorithm:
        success = validate_and_report(args.config, args.algorithm)
    else:
        print("❌ Specify either --all or both --config and --algorithm")
        success = False

    exit(0 if success else 1)