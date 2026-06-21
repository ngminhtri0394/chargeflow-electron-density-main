#!/usr/bin/env python3
"""
Inference script for Charged Electron Density Prediction.

This script provides a production-ready inference pipeline for generating
electron density predictions from trained models.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import RhoDataset, RhoDatasetCharge
from src.models.model_configs import instantiate_model
from src.training import distributed_mode
from src.training.load_and_save import load_model
from src.utils import setup_logger, load_config, save_config, calculate_density_metrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference for electron density prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to inference configuration file (YAML or JSON)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory from config"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size from config"
    )
    
    return parser.parse_args()


def setup_inference_environment(config: dict, logger):
    """Set up the inference environment."""
    
    # Set device
    device_name = config.get('device', 'cuda')
    if device_name == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device_name = 'cpu'
    device = torch.device(device_name)
    
    # Set seed for reproducibility
    seed = config.get('seed', 0)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if device_name == 'cuda':
        cudnn.benchmark = True
    
    logger.info(f"Device: {device}")
    logger.info(f"Seed: {seed}")
    
    return device


def create_dataloader(config: dict, logger):
    """Create inference dataloader."""
    
    data_config = config['data']
    
    # Determine if we have labels for metric computation
    has_labels = (
        data_config.get('label_list') is not None and
        data_config.get('label_gridsize') is not None
    )
    
    # For inference, typically use charge-aware dataset if labels exist
    # and we want to track charge levels
    use_charge = data_config.get('use_charge_dataset', False) and has_labels
    DatasetClass = RhoDatasetCharge if use_charge else RhoDataset
    
    logger.info(f"Creating inference dataset (charge-aware: {use_charge})")
    logger.info(f"Has labels for metrics: {has_labels}")
    
    # Create dataset
    if has_labels:
        dataset = DatasetClass(
            data_list_path=data_config['data_list'],
            label_list_path=data_config['label_list'],
            data_gridsize_path=data_config['data_gridsize'],
            label_gridsize_path=data_config['label_gridsize'],
            data_augmentation=data_config.get('data_augmentation', False),
            downsample_data=data_config.get('downsample_data', 1),
            downsample_label=data_config.get('downsample_label', 1),
        )
    else:
        # Create a minimal dataset class for input-only inference
        logger.warning("No labels provided - metrics computation will be skipped")
        # For simplicity, still use the full dataset class but labels will be ignored
        dataset = RhoDataset(
            data_list_path=data_config['data_list'],
            label_list_path=data_config['data_list'],  # Dummy
            data_gridsize_path=data_config['data_gridsize'],
            label_gridsize_path=data_config['data_gridsize'],  # Dummy
            data_augmentation=False,
            downsample_data=data_config.get('downsample_data', 1),
            downsample_label=data_config.get('downsample_label', 1),
        )
    
    logger.info(f"Dataset size: {len(dataset)}")
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=data_config.get('batch_size', 1),
        num_workers=data_config.get('num_workers', 4),
        pin_memory=data_config.get('pin_memory', True),
        shuffle=False,  # Don't shuffle for inference
    )
    
    return dataloader, has_labels


def load_inference_model(checkpoint_path: str, config: dict, device, logger):
    """Load model for inference."""
    
    model_config = config['model']
    
    logger.info("Loading model...")
    model = instantiate_model(
        architechture=model_config.get('architecture', 'cifar10'),
        is_discrete=model_config.get('discrete_flow_matching', False),
        use_ema=model_config.get('use_ema', False),
    )
    
    model.to(device)
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present (from DDP)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    logger.info("Model loaded successfully")
    
    return model


@torch.no_grad()
def run_inference(model, dataloader, device, config: dict, logger, has_labels: bool):
    """Run inference on the dataset."""
    
    output_config = config['output']
    output_dir = Path(output_config.get('output_dir', './predictions'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_predictions = output_config.get('save_predictions', True)
    compute_metrics = output_config.get('compute_metrics', True) and has_labels
    save_format = output_config.get('save_format', 'npy')
    
    logger.info("Starting inference...")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Save predictions: {save_predictions}")
    logger.info(f"Compute metrics: {compute_metrics}")
    
    all_metrics = []
    
    # Run inference
    for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Inference")):
        # Unpack batch
        if has_labels:
            if len(batch_data) == 3:  # Charge-aware dataset
                inputs, targets, charge_levels = batch_data
            else:
                inputs, targets = batch_data
                charge_levels = None
        else:
            inputs = batch_data[0]
            targets = None
            charge_levels = None
        
        inputs = inputs.to(device)
        
        # Generate prediction
        # This is a simplified version - actual inference depends on the model type
        # For flow matching, you'd need to solve the ODE
        predictions = model(inputs)  # Placeholder - needs proper ODE solving
        
        # Compute metrics if labels are available
        if compute_metrics and targets is not None:
            targets = targets.to(device)
            batch_metrics = calculate_density_metrics(predictions, targets)
            all_metrics.append(batch_metrics)
        
        # Save predictions
        if save_predictions:
            for i in range(predictions.shape[0]):
                pred = predictions[i].cpu().numpy()
                
                # Create filename
                sample_idx = batch_idx * dataloader.batch_size + i
                if charge_levels is not None:
                    charge = charge_levels[i].item()
                    filename = f"prediction_{sample_idx:06d}_charge_{charge}.{save_format}"
                else:
                    filename = f"prediction_{sample_idx:06d}.{save_format}"
                
                # Save file
                output_path = output_dir / filename
                if save_format == 'npy':
                    np.save(output_path, pred)
                elif save_format == 'npz':
                    np.savez_compressed(output_path, density=pred)
                else:
                    logger.warning(f"Unsupported save format: {save_format}")
    
    # Aggregate and save metrics
    if compute_metrics and all_metrics:
        # Average metrics across all samples
        avg_metrics = {
            key: np.mean([m[key] for m in all_metrics])
            for key in all_metrics[0].keys()
        }
        
        logger.info("=" * 80)
        logger.info("Inference Metrics:")
        logger.info("=" * 80)
        for key, value in avg_metrics.items():
            logger.info(f"{key}: {value:.6f}")
        
        # Save metrics to file
        if output_config.get('save_metrics', True):
            metrics_path = output_dir / "metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(avg_metrics, f, indent=2)
            logger.info(f"Saved metrics to {metrics_path}")
    
    logger.info(f"Inference completed. Results saved to {output_dir}")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line args
    if args.output_dir:
        config.setdefault('output', {})['output_dir'] = args.output_dir
    if args.batch_size:
        config.setdefault('data', {})['batch_size'] = args.batch_size
    
    # Set up output directory
    output_dir = Path(config['output'].get('output_dir', './predictions'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    log_file = output_dir / "inference.log"
    logger = setup_logger('inference', log_file=log_file)
    
    logger.info("=" * 80)
    logger.info("Starting Inference")
    logger.info("=" * 80)
    logger.info(f"Config: {args.config}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Output directory: {output_dir}")
    
    # Set up environment
    device = setup_inference_environment(config, logger)
    
    # Create dataloader
    dataloader, has_labels = create_dataloader(config, logger)
    
    # Load model
    model = load_inference_model(args.checkpoint, config, device, logger)
    
    # Run inference
    run_inference(model, dataloader, device, config, logger, has_labels)
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
