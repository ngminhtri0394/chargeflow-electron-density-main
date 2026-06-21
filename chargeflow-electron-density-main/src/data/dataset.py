"""
Dataset classes for loading electron density data.

This module provides PyTorch Dataset classes for loading and preprocessing
electron density data for flow matching models.
"""

import re
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


class RhoDataset(Dataset):
    """
    Dataset for loading electron density pairs (low-res -> high-res).
    
    Args:
        data_list_path: Path to file containing paths to input density files
        label_list_path: Path to file containing paths to target density files
        data_gridsize_path: Path to file containing grid sizes for input densities
        label_gridsize_path: Path to file containing grid sizes for target densities
        data_augmentation: Whether to apply random rotations for data augmentation
        downsample_data: Downsampling factor for input data (default: 1, no downsampling)
        downsample_label: Downsampling factor for target labels (default: 1, no downsampling)
    """
    
    def __init__(
        self,
        data_list_path: Union[str, Path],
        label_list_path: Union[str, Path],
        data_gridsize_path: Union[str, Path],
        label_gridsize_path: Union[str, Path],
        data_augmentation: bool = True,
        downsample_data: int = 1,
        downsample_label: int = 1,
    ):
        self.ds_data = downsample_data
        self.ds_label = downsample_label
        self.data_augmentation = data_augmentation

        # Load file lists
        self.data_paths = np.genfromtxt(data_list_path, dtype=str)
        self.data_gridsizes = np.genfromtxt(data_gridsize_path, dtype=str)
        self.label_paths = np.genfromtxt(label_list_path, dtype=str)
        self.label_gridsizes = np.genfromtxt(label_gridsize_path, dtype=str)

        # Ensure all lists have the same length
        assert self.data_paths.size == self.data_gridsizes.size, \
            "Data paths and grid sizes must have the same length"
        assert self.data_paths.size == self.label_paths.size, \
            "Data and label paths must have the same length"
        assert self.data_paths.size == self.label_gridsizes.size, \
            "Data paths and label grid sizes must have the same length"

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.data_paths.size

    def _rotate_x(self, data: torch.Tensor) -> torch.Tensor:
        """Rotate 90 degrees around x-axis."""
        return data.transpose(-1, -2).flip(-1)

    def _rotate_y(self, data: torch.Tensor) -> torch.Tensor:
        """Rotate 90 degrees around y-axis."""
        return data.transpose(-1, -3).flip(-1)

    def _rotate_z(self, data: torch.Tensor) -> torch.Tensor:
        """Rotate 90 degrees around z-axis."""
        return data.transpose(-2, -3).flip(-2)

    def _random_rotation(self, data_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Apply random rotation to all tensors in the list.
        
        Args:
            data_list: List of tensors to rotate together
            
        Returns:
            List of rotated tensors
        """
        # Choose random axis
        axis = np.random.randint(3)
        if axis == 0:
            rotate_func = self._rotate_x
        elif axis == 1:
            rotate_func = self._rotate_y
        else:
            rotate_func = self._rotate_z
        
        # Choose number of rotations (0, 1, 2, or 3 times 90 degrees)
        r = np.random.rand()
        if r < 0.1:
            return data_list  # No rotation
        elif r < 0.4:
            return [rotate_func(d) for d in data_list]  # 90 degrees
        elif r < 0.7:
            return [rotate_func(rotate_func(d)) for d in data_list]  # 180 degrees
        else:
            return [rotate_func(rotate_func(rotate_func(d))) for d in data_list]  # 270 degrees

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (input_density, target_density)
        """
        # Load input density
        rho_input = torch.tensor(
            np.load(self.data_paths[idx]), dtype=torch.float32
        )
        input_size = np.loadtxt(self.data_gridsizes[idx], dtype=int)
        rho_input = rho_input.reshape(1, *input_size)

        # Load target density
        rho_target = torch.tensor(
            np.load(self.label_paths[idx]), dtype=torch.float32
        )
        target_size = np.loadtxt(self.label_gridsizes[idx], dtype=int)
        rho_target = rho_target.reshape(1, *target_size)

        # Apply data augmentation
        if self.data_augmentation:
            rho_input, rho_target = self._random_rotation([rho_input, rho_target])

        # Apply downsampling
        rho_input = self._downsample(rho_input, self.ds_data)
        rho_target = self._downsample(rho_target, self.ds_label)

        return rho_input, rho_target

    def _downsample(self, data: torch.Tensor, factor: int) -> torch.Tensor:
        """
        Downsample 3D data by taking every nth point.
        
        Args:
            data: Input tensor with shape (C, X, Y, Z)
            factor: Downsampling factor
            
        Returns:
            Downsampled tensor
        """
        if factor == 1:
            return data
        
        nx, ny, nz = data.size()[-3:]
        nx = (nx // factor) * factor
        ny = (ny // factor) * factor
        nz = (nz // factor) * factor
        
        return data[..., :nx:factor, :ny:factor, :nz:factor]


class RhoDatasetCharge(Dataset):
    """
    Dataset for loading charged electron density data.
    
    This dataset extracts charge level information from filenames and returns it
    along with the density data.
    
    Args:
        data_list_path: Path to file containing paths to input density files
        label_list_path: Path to file containing paths to target density files
        data_gridsize_path: Path to file containing grid sizes for input densities
        label_gridsize_path: Path to file containing grid sizes for target densities
        data_augmentation: Whether to apply random rotations for data augmentation
        downsample_data: Downsampling factor for input data (default: 1, no downsampling)
        downsample_label: Downsampling factor for target labels (default: 1, no downsampling)
    """
    
    def __init__(
        self,
        data_list_path: Union[str, Path],
        label_list_path: Union[str, Path],
        data_gridsize_path: Union[str, Path],
        label_gridsize_path: Union[str, Path],
        data_augmentation: bool = True,
        downsample_data: int = 1,
        downsample_label: int = 1,
    ):
        self.ds_data = downsample_data
        self.ds_label = downsample_label
        self.data_augmentation = data_augmentation

        # Load file lists
        self.data_paths = np.genfromtxt(data_list_path, dtype=str)
        self.data_gridsizes = np.genfromtxt(data_gridsize_path, dtype=str)
        self.label_paths = np.genfromtxt(label_list_path, dtype=str)
        self.label_gridsizes = np.genfromtxt(label_gridsize_path, dtype=str)

        # Ensure all lists have the same length
        assert self.data_paths.size == self.data_gridsizes.size, \
            "Data paths and grid sizes must have the same length"
        assert self.data_paths.size == self.label_paths.size, \
            "Data and label paths must have the same length"
        assert self.data_paths.size == self.label_gridsizes.size, \
            "Data paths and label grid sizes must have the same length"

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.data_paths.size

    def _rotate_x(self, data: torch.Tensor) -> torch.Tensor:
        """Rotate 90 degrees around x-axis."""
        return data.transpose(-1, -2).flip(-1)

    def _rotate_y(self, data: torch.Tensor) -> torch.Tensor:
        """Rotate 90 degrees around y-axis."""
        return data.transpose(-1, -3).flip(-1)

    def _rotate_z(self, data: torch.Tensor) -> torch.Tensor:
        """Rotate 90 degrees around z-axis."""
        return data.transpose(-2, -3).flip(-2)

    def _random_rotation(self, data_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Apply random rotation to all tensors in the list.
        
        Args:
            data_list: List of tensors to rotate together
            
        Returns:
            List of rotated tensors
        """
        # Choose random axis
        axis = np.random.randint(3)
        if axis == 0:
            rotate_func = self._rotate_x
        elif axis == 1:
            rotate_func = self._rotate_y
        else:
            rotate_func = self._rotate_z
        
        # Choose number of rotations (0, 1, 2, or 3 times 90 degrees)
        r = np.random.rand()
        if r < 0.1:
            return data_list  # No rotation
        elif r < 0.4:
            return [rotate_func(d) for d in data_list]  # 90 degrees
        elif r < 0.7:
            return [rotate_func(rotate_func(d)) for d in data_list]  # 180 degrees
        else:
            return [rotate_func(rotate_func(rotate_func(d))) for d in data_list]  # 270 degrees

    def _extract_charge_level(self, filename: str) -> int:
        """
        Extract charge level from filename.
        
        Expected format: ..._charge_<value>...
        
        Args:
            filename: Path to the file
            
        Returns:
            Charge level as integer, offset by +3 to make non-negative
        """
        match = re.search(r'_charge_([-\d\.]+)', filename)
        if match:
            charge_level = int(match.group(1))
        else:
            charge_level = 0
        # Offset by 3 to make charge levels non-negative
        return charge_level + 3

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (input_density, target_density, charge_level)
        """
        # Load input density
        rho_input = torch.tensor(
            np.load(self.data_paths[idx]), dtype=torch.float32
        )
        input_size = np.loadtxt(self.data_gridsizes[idx], dtype=int)
        rho_input = rho_input.reshape(1, *input_size)

        # Load target density and extract charge level
        rho_target = torch.tensor(
            np.load(self.label_paths[idx]), dtype=torch.float32
        )
        charge_level = self._extract_charge_level(self.label_paths[idx])
        target_size = np.loadtxt(self.label_gridsizes[idx], dtype=int)
        rho_target = rho_target.reshape(1, *target_size)

        # Apply data augmentation
        if self.data_augmentation:
            rho_input, rho_target = self._random_rotation([rho_input, rho_target])

        # Apply downsampling
        rho_input = self._downsample(rho_input, self.ds_data)
        rho_target = self._downsample(rho_target, self.ds_label)

        return rho_input, rho_target, charge_level

    def _downsample(self, data: torch.Tensor, factor: int) -> torch.Tensor:
        """
        Downsample 3D data by taking every nth point.
        
        Args:
            data: Input tensor with shape (C, X, Y, Z)
            factor: Downsampling factor
            
        Returns:
            Downsampled tensor
        """
        if factor == 1:
            return data
        
        nx, ny, nz = data.size()[-3:]
        nx = (nx // factor) * factor
        ny = (ny // factor) * factor
        nz = (nz // factor) * factor
        
        return data[..., :nx:factor, :ny:factor, :nz:factor]
