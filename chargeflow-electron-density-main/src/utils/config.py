"""
Configuration management utilities.

This module provides functions for loading and saving configuration files
in YAML and JSON formats.
"""

import json
import yaml
from pathlib import Path
from typing import Any, Dict, Union


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from a YAML or JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration parameters
        
    Raises:
        ValueError: If file format is not supported
        FileNotFoundError: If config file doesn't exist
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    suffix = config_path.suffix.lower()
    
    with open(config_path, 'r') as f:
        if suffix in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        elif suffix == '.json':
            config = json.load(f)
        else:
            raise ValueError(
                f"Unsupported configuration format: {suffix}. "
                "Supported formats: .yaml, .yml, .json"
            )
    
    return config if config is not None else {}


def save_config(config: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """
    Save configuration to a YAML or JSON file.
    
    Args:
        config: Configuration dictionary to save
        output_path: Path where to save the configuration
        
    Raises:
        ValueError: If file format is not supported
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    suffix = output_path.suffix.lower()
    
    with open(output_path, 'w') as f:
        if suffix in ['.yaml', '.yml']:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        elif suffix == '.json':
            json.dump(config, f, indent=2)
        else:
            raise ValueError(
                f"Unsupported configuration format: {suffix}. "
                "Supported formats: .yaml, .yml, .json"
            )


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries, with override_config taking precedence.
    
    Args:
        base_config: Base configuration dictionary
        override_config: Configuration to override base values
        
    Returns:
        Merged configuration dictionary
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged
