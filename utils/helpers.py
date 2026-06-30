"""
Helper utilities for the NPP system.
Provides common functions for data processing and model operations.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file not found
        yaml.YAMLError: If YAML parsing fails
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config or {}


def save_config(config: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save YAML file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def create_directories(dirs: List[Union[str, Path]]) -> None:
    """
    Create multiple directories if they don't exist.
    
    Args:
        dirs: List of directory paths to create
    """
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def get_device(use_cuda: bool = True) -> torch.device:
    """
    Get appropriate device (CUDA or CPU).
    
    Args:
        use_cuda: Whether to use CUDA if available
        
    Returns:
        torch.device instance
    """
    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_json(data: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """
    Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        output_path: Path to output JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(input_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load dictionary from JSON file.
    
    Args:
        input_path: Path to JSON file
        
    Returns:
        Loaded dictionary
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"JSON file not found: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count total trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_parameters(num_params: int) -> str:
    """
    Format parameter count in human-readable format.
    
    Args:
        num_params: Number of parameters
        
    Returns:
        Formatted string
    """
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    else:
        return str(num_params)


def move_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """
    Move batch to specified device.
    
    Args:
        batch: Batch dictionary with tensors
        device: Target device
        
    Returns:
        Batch moved to device
    """
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
    return batch


def set_requires_grad(model: torch.nn.Module, requires_grad: bool) -> None:
    """
    Set requires_grad for all parameters in a model.
    
    Args:
        model: PyTorch model
        requires_grad: Whether to require gradients
    """
    for param in model.parameters():
        param.requires_grad = requires_grad


def get_model_size(model: torch.nn.Module) -> float:
    """
    Get approximate model size in MB.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size in MB
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def seed_worker(worker_id: int) -> None:
    """
    Worker seed for DataLoader reproducibility.
    
    Args:
        worker_id: Worker ID from DataLoader
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """
    Flatten nested dictionary.
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Separator for nested keys
        
    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)