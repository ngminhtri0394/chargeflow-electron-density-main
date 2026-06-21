"""
Unit tests for dataset classes.

Run tests with: pytest tests/test_dataset.py
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import tempfile
import shutil

from src.data.dataset import RhoDataset, RhoDatasetCharge


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory with mock data."""
    temp_dir = tempfile.mkdtemp()
    
    # Create mock data files
    data_dir = Path(temp_dir) / "data"
    data_dir.mkdir()
    
    # Create sample density files
    for i in range(5):
        # Low-res data
        data = np.random.randn(32, 32, 32).astype(np.float32)
        np.save(data_dir / f"sample_{i:03d}_data.npy", data)
        
        # High-res label
        label = np.random.randn(64, 64, 64).astype(np.float32)
        np.save(data_dir / f"sample_{i:03d}_label_charge_{i-2}.npy", label)
        
        # Grid sizes
        with open(data_dir / f"sample_{i:03d}_data_gs.txt", 'w') as f:
            f.write("32 32 32\n")
        with open(data_dir / f"sample_{i:03d}_label_gs.txt", 'w') as f:
            f.write("64 64 64\n")
    
    # Create list files
    with open(data_dir / "list_data.txt", 'w') as f:
        for i in range(5):
            f.write(str(data_dir / f"sample_{i:03d}_data.npy") + "\n")
    
    with open(data_dir / "list_label.txt", 'w') as f:
        for i in range(5):
            f.write(str(data_dir / f"sample_{i:03d}_label_charge_{i-2}.npy") + "\n")
    
    with open(data_dir / "list_data_gs.txt", 'w') as f:
        for i in range(5):
            f.write(str(data_dir / f"sample_{i:03d}_data_gs.txt") + "\n")
    
    with open(data_dir / "list_label_gs.txt", 'w') as f:
        for i in range(5):
            f.write(str(data_dir / f"sample_{i:03d}_label_gs.txt") + "\n")
    
    yield data_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


class TestRhoDataset:
    """Tests for RhoDataset class."""
    
    def test_initialization(self, temp_data_dir):
        """Test dataset initialization."""
        dataset = RhoDataset(
            data_list_path=temp_data_dir / "list_data.txt",
            label_list_path=temp_data_dir / "list_label.txt",
            data_gridsize_path=temp_data_dir / "list_data_gs.txt",
            label_gridsize_path=temp_data_dir / "list_label_gs.txt",
            data_augmentation=False,
        )
        
        assert len(dataset) == 5
    
    def test_getitem(self, temp_data_dir):
        """Test dataset item retrieval."""
        dataset = RhoDataset(
            data_list_path=temp_data_dir / "list_data.txt",
            label_list_path=temp_data_dir / "list_label.txt",
            data_gridsize_path=temp_data_dir / "list_data_gs.txt",
            label_gridsize_path=temp_data_dir / "list_label_gs.txt",
            data_augmentation=False,
        )
        
        rho_input, rho_target = dataset[0]
        
        # Check types
        assert isinstance(rho_input, torch.Tensor)
        assert isinstance(rho_target, torch.Tensor)
        
        # Check shapes
        assert rho_input.shape == (1, 32, 32, 32)
        assert rho_target.shape == (1, 64, 64, 64)
        
        # Check dtype
        assert rho_input.dtype == torch.float32
        assert rho_target.dtype == torch.float32
    
    def test_downsampling(self, temp_data_dir):
        """Test downsampling functionality."""
        dataset = RhoDataset(
            data_list_path=temp_data_dir / "list_data.txt",
            label_list_path=temp_data_dir / "list_label.txt",
            data_gridsize_path=temp_data_dir / "list_data_gs.txt",
            label_gridsize_path=temp_data_dir / "list_label_gs.txt",
            data_augmentation=False,
            downsample_data=2,
            downsample_label=2,
        )
        
        rho_input, rho_target = dataset[0]
        
        # Check downsampled shapes
        assert rho_input.shape == (1, 16, 16, 16)
        assert rho_target.shape == (1, 32, 32, 32)
    
    def test_rotation(self, temp_data_dir):
        """Test that rotation doesn't crash (not deterministic)."""
        dataset = RhoDataset(
            data_list_path=temp_data_dir / "list_data.txt",
            label_list_path=temp_data_dir / "list_label.txt",
            data_gridsize_path=temp_data_dir / "list_data_gs.txt",
            label_gridsize_path=temp_data_dir / "list_label_gs.txt",
            data_augmentation=True,
        )
        
        # Should not raise an error
        for i in range(5):
            rho_input, rho_target = dataset[i]
            assert rho_input.shape == (1, 32, 32, 32)
            assert rho_target.shape == (1, 64, 64, 64)


class TestRhoDatasetCharge:
    """Tests for RhoDatasetCharge class."""
    
    def test_initialization(self, temp_data_dir):
        """Test charge dataset initialization."""
        dataset = RhoDatasetCharge(
            data_list_path=temp_data_dir / "list_data.txt",
            label_list_path=temp_data_dir / "list_label.txt",
            data_gridsize_path=temp_data_dir / "list_data_gs.txt",
            label_gridsize_path=temp_data_dir / "list_label_gs.txt",
            data_augmentation=False,
        )
        
        assert len(dataset) == 5
    
    def test_charge_extraction(self, temp_data_dir):
        """Test charge level extraction."""
        dataset = RhoDatasetCharge(
            data_list_path=temp_data_dir / "list_data.txt",
            label_list_path=temp_data_dir / "list_label.txt",
            data_gridsize_path=temp_data_dir / "list_data_gs.txt",
            label_gridsize_path=temp_data_dir / "list_label_gs.txt",
            data_augmentation=False,
        )
        
        # Test each sample
        for i in range(5):
            rho_input, rho_target, charge_level = dataset[i]
            
            # Check charge level extraction (i-2+3 = i+1)
            expected_charge = i + 1
            assert charge_level == expected_charge
            
            # Check types and shapes
            assert isinstance(rho_input, torch.Tensor)
            assert isinstance(rho_target, torch.Tensor)
            assert isinstance(charge_level, int)
            assert rho_input.shape == (1, 32, 32, 32)
            assert rho_target.shape == (1, 64, 64, 64)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
