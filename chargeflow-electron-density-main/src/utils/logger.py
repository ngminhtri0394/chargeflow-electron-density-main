"""
Logging utilities for training and evaluation.

This module provides functions to set up and configure logging.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union


def setup_logger(
    name: str = __name__,
    log_file: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
    format_str: Optional[str] = None,
) -> logging.Logger:
    """
    Set up a logger with console and optional file output.
    
    Args:
        name: Name of the logger
        log_file: Optional path to log file. If None, only console output is used.
        level: Logging level (e.g., logging.INFO, logging.DEBUG)
        format_str: Custom format string. If None, uses default format.
        
    Returns:
        Configured logger instance
    """
    if format_str is None:
        format_str = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
    
    formatter = logging.Formatter(
        format_str,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Get or create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_metrics(
    logger: logging.Logger,
    metrics: dict,
    prefix: str = "",
    level: int = logging.INFO
) -> None:
    """
    Log a dictionary of metrics in a formatted way.
    
    Args:
        logger: Logger instance to use
        metrics: Dictionary of metric names and values
        prefix: Optional prefix to add before metric names
        level: Logging level to use
    """
    if prefix and not prefix.endswith("_"):
        prefix += "_"
    
    formatted_metrics = []
    for key, value in metrics.items():
        if isinstance(value, float):
            formatted_metrics.append(f"{prefix}{key}: {value:.6f}")
        else:
            formatted_metrics.append(f"{prefix}{key}: {value}")
    
    logger.log(level, " | ".join(formatted_metrics))
