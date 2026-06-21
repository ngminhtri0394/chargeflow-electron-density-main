"""
Metrics calculation utilities.

This module provides functions for calculating various metrics
used to evaluate model performance.
"""

import torch
import numpy as np
from typing import Tuple, Optional


def calculate_mae(
    prediction: torch.Tensor,
    target: torch.Tensor,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Calculate Mean Absolute Error.
    
    Args:
        prediction: Predicted values
        target: Target values
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        MAE value(s)
    """
    mae = torch.abs(prediction - target)
    
    if reduction == 'mean':
        return mae.mean()
    elif reduction == 'sum':
        return mae.sum()
    else:
        return mae


def calculate_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Calculate Mean Squared Error.
    
    Args:
        prediction: Predicted values
        target: Target values
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        MSE value(s)
    """
    mse = (prediction - target) ** 2
    
    if reduction == 'mean':
        return mse.mean()
    elif reduction == 'sum':
        return mse.sum()
    else:
        return mse


def calculate_rmse(
    prediction: torch.Tensor,
    target: torch.Tensor
) -> torch.Tensor:
    """
    Calculate Root Mean Squared Error.
    
    Args:
        prediction: Predicted values
        target: Target values
        
    Returns:
        RMSE value
    """
    return torch.sqrt(calculate_mse(prediction, target, reduction='mean'))


def calculate_normalized_mae(
    prediction: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1e-8
) -> torch.Tensor:
    """
    Calculate Normalized Mean Absolute Error.
    
    Normalized by the mean absolute value of the target.
    
    Args:
        prediction: Predicted values
        target: Target values
        epsilon: Small value to avoid division by zero
        
    Returns:
        Normalized MAE value
    """
    mae = calculate_mae(prediction, target, reduction='mean')
    target_norm = torch.abs(target).mean() + epsilon
    return mae / target_norm


def calculate_relative_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1e-8
) -> torch.Tensor:
    """
    Calculate relative error (element-wise).
    
    Args:
        prediction: Predicted values
        target: Target values
        epsilon: Small value to avoid division by zero
        
    Returns:
        Relative error
    """
    return torch.abs(prediction - target) / (torch.abs(target) + epsilon)


def calculate_r2_score(
    prediction: torch.Tensor,
    target: torch.Tensor
) -> float:
    """
    Calculate R² (coefficient of determination) score.
    
    Args:
        prediction: Predicted values
        target: Target values
        
    Returns:
        R² score
    """
    ss_res = ((target - prediction) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    
    return 1.0 - (ss_res / ss_tot).item()


def calculate_density_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor
) -> dict:
    """
    Calculate a comprehensive set of metrics for density prediction.
    
    Args:
        prediction: Predicted density values
        target: Target density values
        
    Returns:
        Dictionary containing various metrics
    """
    metrics = {
        'mae': calculate_mae(prediction, target).item(),
        'mse': calculate_mse(prediction, target).item(),
        'rmse': calculate_rmse(prediction, target).item(),
        'normalized_mae': calculate_normalized_mae(prediction, target).item(),
        'r2_score': calculate_r2_score(prediction, target),
    }
    
    return metrics
