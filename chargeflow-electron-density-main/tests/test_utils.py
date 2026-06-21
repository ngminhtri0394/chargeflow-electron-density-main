"""
Unit tests for utility functions.

Run tests with: pytest tests/test_utils.py
"""

import pytest
import torch
import tempfile
from pathlib import Path

from src.utils.config import load_config, save_config, merge_configs
from src.utils.metrics import (
    calculate_mae, calculate_mse, calculate_rmse,
    calculate_normalized_mae, calculate_r2_score,
    calculate_density_metrics
)


class TestConfigUtils:
    """Tests for configuration utilities."""
    
    def test_save_and_load_yaml(self):
        """Test saving and loading YAML config."""
        config = {
            'model': {'architecture': 'cifar10'},
            'training': {'epochs': 100, 'lr': 0.001},
        }
        
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
            config_path = Path(f.name)
        
        try:
            # Save and load
            save_config(config, config_path)
            loaded_config = load_config(config_path)
            
            assert loaded_config == config
        finally:
            config_path.unlink()
    
    def test_save_and_load_json(self):
        """Test saving and loading JSON config."""
        config = {
            'model': {'architecture': 'cifar10'},
            'training': {'epochs': 100, 'lr': 0.001},
        }
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            config_path = Path(f.name)
        
        try:
            # Save and load
            save_config(config, config_path)
            loaded_config = load_config(config_path)
            
            assert loaded_config == config
        finally:
            config_path.unlink()
    
    def test_merge_configs(self):
        """Test merging configurations."""
        base = {
            'model': {'architecture': 'cifar10', 'layers': 10},
            'training': {'epochs': 100},
        }
        
        override = {
            'model': {'layers': 20},
            'training': {'lr': 0.001},
        }
        
        merged = merge_configs(base, override)
        
        # Check merged values
        assert merged['model']['architecture'] == 'cifar10'
        assert merged['model']['layers'] == 20
        assert merged['training']['epochs'] == 100
        assert merged['training']['lr'] == 0.001


class TestMetrics:
    """Tests for metrics calculations."""
    
    def test_calculate_mae(self):
        """Test MAE calculation."""
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.5, 2.5, 3.5])
        
        mae = calculate_mae(pred, target)
        assert torch.isclose(mae, torch.tensor(0.5))
    
    def test_calculate_mse(self):
        """Test MSE calculation."""
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([2.0, 3.0, 4.0])
        
        mse = calculate_mse(pred, target)
        assert torch.isclose(mse, torch.tensor(1.0))
    
    def test_calculate_rmse(self):
        """Test RMSE calculation."""
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([2.0, 3.0, 4.0])
        
        rmse = calculate_rmse(pred, target)
        assert torch.isclose(rmse, torch.tensor(1.0))
    
    def test_calculate_normalized_mae(self):
        """Test normalized MAE calculation."""
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([2.0, 4.0, 6.0])
        
        norm_mae = calculate_normalized_mae(pred, target)
        
        # MAE = (1 + 2 + 3) / 3 = 2.0
        # Target norm = (2 + 4 + 6) / 3 = 4.0
        # Normalized MAE = 2.0 / 4.0 = 0.5
        assert torch.isclose(norm_mae, torch.tensor(0.5))
    
    def test_calculate_r2_score(self):
        """Test R² score calculation."""
        # Perfect prediction
        pred = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        target = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        
        r2 = calculate_r2_score(pred, target)
        assert r2 == 1.0
        
        # Poor prediction (constant)
        pred = torch.tensor([3.0, 3.0, 3.0, 3.0, 3.0])
        target = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        
        r2 = calculate_r2_score(pred, target)
        assert r2 == 0.0
    
    def test_calculate_density_metrics(self):
        """Test comprehensive density metrics."""
        pred = torch.randn(10, 10, 10)
        target = pred + torch.randn(10, 10, 10) * 0.1
        
        metrics = calculate_density_metrics(pred, target)
        
        # Check all expected keys are present
        assert 'mae' in metrics
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'normalized_mae' in metrics
        assert 'r2_score' in metrics
        
        # Check values are reasonable
        assert metrics['mae'] >= 0
        assert metrics['mse'] >= 0
        assert metrics['rmse'] >= 0
        assert metrics['normalized_mae'] >= 0
        assert -1 <= metrics['r2_score'] <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
