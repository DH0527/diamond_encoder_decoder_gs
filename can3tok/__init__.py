"""can3tok: Gaussian-splat point autoencoder with a fixed 32x64x64 latent."""

from .config import Can3TokConfig, channel_budget, describe_layout, patch_layout, validate_layout
from .model import Can3TokAE, build_model

__all__ = [
    "Can3TokConfig",
    "Can3TokAE",
    "build_model",
    "patch_layout",
    "channel_budget",
    "validate_layout",
    "describe_layout",
]
__version__ = "0.1.0"
