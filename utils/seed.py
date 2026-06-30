"""
Seed and reproducibility utilities for the NPP system.
Ensures deterministic behavior across all components.
"""

import os
import random
import numpy as np
import torch
from typing import Optional


def set_seed(seed: int) -> None:
    """
    Set random seed for all libraries to ensure reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def enable_deterministic_mode(deterministic: bool = True) -> None:
    """
    Enable or disable deterministic mode in PyTorch.
    
    Args:
        deterministic: Whether to enable deterministic mode
    """
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def setup_reproducibility(
    seed: int,
    deterministic: bool = True,
    disable_cudnn_benchmark: bool = True
) -> None:
    """
    Complete setup for reproducibility.
    
    Args:
        seed: Random seed value
        deterministic: Whether to enable deterministic mode
        disable_cudnn_benchmark: Whether to disable cuDNN benchmark
    """
    set_seed(seed)
    enable_deterministic_mode(deterministic)
    
    if disable_cudnn_benchmark:
        torch.backends.cudnn.benchmark = False