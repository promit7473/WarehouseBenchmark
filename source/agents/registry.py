# source/agents/registry.py
"""Model registry for extensible neural network model creation.

This module provides a registry pattern (inspired by RLRoverLab) for
registering and retrieving neural network models by name.

Usage:
    from source.agents.registry import register_model, MODEL_REGISTRY

    @register_model("MyCustomPolicy")
    class MyCustomPolicy(GaussianMixin, Model):
        ...

    # Later, retrieve by name
    ModelClass = MODEL_REGISTRY["MyCustomPolicy"]
"""

from typing import Dict, Type, Any

# Global model registry
MODEL_REGISTRY: Dict[str, Type[Any]] = {}


def register_model(name: str):
    """
    Decorator to register a model class in the global registry.

    Args:
        name: The name to register the model under

    Returns:
        Decorator function that registers the class

    Raises:
        ValueError: If a model with the same name is already registered

    Example:
        @register_model("GaussianPolicyMLP")
        class GaussianPolicyMLP(GaussianMixin, Model):
            ...
    """
    def decorator(cls: Type[Any]) -> Type[Any]:
        if name in MODEL_REGISTRY:
            raise ValueError(
                f"Model '{name}' is already registered. "
                f"Choose a different name or unregister the existing model."
            )
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def unregister_model(name: str) -> None:
    """
    Remove a model from the registry.

    Args:
        name: The name of the model to unregister

    Raises:
        KeyError: If the model is not in the registry
    """
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Model '{name}' is not registered.")
    del MODEL_REGISTRY[name]


def get_model(name: str) -> Type[Any]:
    """
    Retrieve a model class from the registry.

    Args:
        name: The name of the model to retrieve

    Returns:
        The model class

    Raises:
        ValueError: If the model is not found in the registry
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Model '{name}' not found in registry. "
            f"Available models: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name]


def list_models() -> list:
    """
    List all registered model names.

    Returns:
        List of registered model names
    """
    return list(MODEL_REGISTRY.keys())


# Register built-in models when this module is imported
def _register_builtin_models():
    """Register the built-in model classes."""
    from .models import (
        PolicyPPO,
        ValuePPO,
        PolicySAC,
        CriticSAC,
        ActorTD3,
        CriticTD3,
        PolicyPPOVision,
        ValuePPOVision,
    )

    # Only register if not already registered
    builtin_models = {
        "PolicyPPO": PolicyPPO,
        "ValuePPO": ValuePPO,
        "GaussianPolicyMLP": PolicyPPO,  # Alias
        "DeterministicValueMLP": ValuePPO,  # Alias
        "PolicySAC": PolicySAC,
        "CriticSAC": CriticSAC,
        "ActorTD3": ActorTD3,
        "CriticTD3": CriticTD3,
        "PolicyPPOVision": PolicyPPOVision,
        "ValuePPOVision": ValuePPOVision,
    }

    for name, cls in builtin_models.items():
        if name not in MODEL_REGISTRY:
            MODEL_REGISTRY[name] = cls


# Auto-register built-in models on import
try:
    _register_builtin_models()
except ImportError:
    # Models not yet available (circular import protection)
    pass
