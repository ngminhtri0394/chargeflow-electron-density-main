#!/usr/bin/env python3
"""
Training script for Charged Electron Density Prediction.

This script provides a production-ready training pipeline with configuration-based
setup, comprehensive logging, and checkpoint management.
"""

import argparse
import datetime
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.backends.cudnn as cudnn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import RhoDataset, RhoDatasetCharge
from src.models.model_configs import instantiate_model
from src.training import distributed_mode
from src.training.grad_scaler import NativeScalerWithGradNormCount as NativeScaler
from src.training.load_and_save import load_model, save_model, save_best_model
from src.training.train_loop import train_one_epoch
from src.training.eval_loop import eval_model
from src.utils import setup_logger, load_config, save_config, log_metrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train electron density prediction model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file (YAML or JSON)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run evaluation, no training"
    )
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Run one epoch for testing"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training"
    )
    
    return parser.parse_args()


def setup_training_environment(config: dict, logger):
    """Set up the training environment including device, seed, and distributed mode."""
    
    # Set up distributed mode
    class Args:
        """Temporary args object for distributed_mode compatibility."""
        def __init__(self, config):
            self.distributed = config.get('distributed', {}).get('enabled', False)
            self.dist_backend = config.get('distributed', {}).get('backend', 'nccl')
            self.dist_url = config.get('distributed', {}).get('init_method', 'env://')
            self.world_size = 1
            self.rank = 0
            self.gpu = 0
            self.dist_on_itp = False
            self.local_rank = -1
    
    args = Args(config)
    distributed_mode.init_distributed_mode(args)
    
    # Set device
    device_name = config.get('device', 'cuda')
    if device_name == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device_name = 'cpu'
    device = torch.device(device_name)
    
    # Set seed for reproducibility
    seed = config.get('seed', 0) + distributed_mode.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if device_name == 'cuda':
        cudnn.benchmark = True
    
    logger.info(f"Device: {device}")
    logger.info(f"Seed: {seed}")
    logger.info(f"Distributed: {args.distributed}")
    
    return device, args


def create_dataloaders(config: dict, logger):
    """Create training and validation dataloaders."""
    
    data_config = config['data']
    
    # Determine dataset class
    use_charge = data_config.get('use_charge_dataset', False)
    DatasetClass = RhoDatasetCharge if use_charge else RhoDataset
    
    logger.info(f"Creating dataset (charge-aware: {use_charge})")
    
    # Create training dataset
    train_dataset = DatasetClass(
        data_list_path=data_config['train_data_list'],
        label_list_path=data_config['train_label_list'],
        data_gridsize_path=data_config['train_data_gridsize'],
        label_gridsize_path=data_config['train_label_gridsize'],
        data_augmentation=data_config.get('data_augmentation', True),
        downsample_data=data_config.get('downsample_data', 1),
        downsample_label=data_config.get('downsample_label', 1),
    )
    
    logger.info(f"Training dataset size: {len(train_dataset)}")
    
    # Set up distributed sampler
    num_tasks = distributed_mode.get_world_size()
    global_rank = distributed_mode.get_rank()
    
    sampler_train = torch.utils.data.DistributedSampler(
        train_dataset, 
        num_replicas=num_tasks, 
        rank=global_rank, 
        shuffle=True
    )
    
    # Create dataloader. Densities have per-structure grid sizes, so batching
    # requires batch_size=1 (the original pipeline's approach); use accum_iter
    # for an effective larger batch. No padding — zero-padding 3D density is
    # physically wrong (injects fake vacuum, breaks periodic boundaries, and
    # distorts the normalized-MAE denominator).
    dataloader_train = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler_train,
        batch_size=data_config.get('batch_size', 1),
        num_workers=data_config.get('num_workers', 10),
        pin_memory=data_config.get('pin_memory', True),
        drop_last=True,
    )
    
    return dataloader_train, sampler_train


def create_model(config: dict, logger):
    """Create and configure the model."""
    
    model_config = config['model']
    
    logger.info("Creating model...")
    model = instantiate_model(
        architechture=model_config.get('architecture', 'cifar10'),
        is_discrete=model_config.get('discrete_flow_matching', False),
        use_ema=model_config.get('use_ema', False),
    )
    
    logger.info(f"Model architecture: {model_config.get('architecture')}")
    
    return model


def create_optimizer_and_scheduler(model, config: dict, logger):
    """Create optimizer and learning rate scheduler."""
    
    train_config = config['training']
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.get('learning_rate', 0.0001),
        betas=train_config.get('optimizer_betas', [0.9, 0.95])
    )
    
    # Create learning rate scheduler
    epochs = train_config.get('epochs', 10000)
    if train_config.get('decay_lr', False):
        lr_schedule = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            total_iters=epochs,
            start_factor=1.0,
            end_factor=1e-8 / train_config.get('learning_rate', 0.0001),
        )
    else:
        lr_schedule = torch.optim.lr_scheduler.ConstantLR(
            optimizer, 
            total_iters=epochs, 
            factor=1.0
        )
    
    logger.info(f"Optimizer: {optimizer.__class__.__name__}")
    logger.info(f"Learning rate: {train_config.get('learning_rate')}")
    logger.info(f"LR decay: {train_config.get('decay_lr', False)}")
    
    return optimizer, lr_schedule


def train(config: dict, args):
    """Main training function."""
    
    # Set up output directory
    output_dir = Path(config['output'].get('output_dir', './output_dir'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    log_file = output_dir / config['output'].get('log_file', 'training.log')
    logger = setup_logger('train', log_file=log_file)
    
    logger.info("=" * 80)
    logger.info("Starting Training")
    logger.info("=" * 80)
    logger.info(f"Output directory: {output_dir}")
    
    # Save configuration
    if distributed_mode.is_main_process():
        config_save_path = output_dir / "config.yaml"
        save_config(config, config_save_path)
        logger.info(f"Saved configuration to {config_save_path}")
    
    # Set up environment
    device, dist_args = setup_training_environment(config, logger)
    
    # Create dataloaders
    dataloader_train, sampler_train = create_dataloaders(config, logger)
    
    # Create model
    model = create_model(config, logger)
    model.to(device)
    
    # Set up distributed training
    model_without_ddp = model
    if dist_args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[dist_args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module
    
    # Create optimizer and scheduler
    optimizer, lr_schedule = create_optimizer_and_scheduler(
        model_without_ddp, config, logger
    )
    
    # Create gradient scaler
    loss_scaler = NativeScaler()
    
    # Load checkpoint if resuming
    start_epoch = config['training'].get('start_epoch', 0)
    if args.resume:
        # Create a simple args object for load_model compatibility
        class LoadArgs:
            def __init__(self):
                self.resume = args.resume
        
        load_args = LoadArgs()
        load_model(
            args=load_args,
            model_without_ddp=model_without_ddp,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            lr_schedule=lr_schedule,
        )
        logger.info(f"Resumed from checkpoint: {args.resume}")
    
    # Training loop
    epochs = config['training'].get('epochs', 10000)
    logger.info(f"Training from epoch {start_epoch} to {epochs}")
    
    start_time = time.time()
    best_loss = float('inf')
    
    for epoch in range(start_epoch, epochs):
        if dist_args.distributed:
            sampler_train.set_epoch(epoch)
        
        if not args.eval_only:
            # Training epoch
            train_stats = train_one_epoch(
                model=model,
                data_loader=dataloader_train,
                optimizer=optimizer,
                lr_schedule=lr_schedule,
                device=device,
                epoch=epoch,
                loss_scaler=loss_scaler,
                args=create_train_args(config),
            )
            
            # Log training stats
            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                "epoch": epoch,
            }
            
            if distributed_mode.is_main_process():
                log_metrics(logger, train_stats, prefix="train")
            
            # Save best model
            if 'loss' in train_stats and train_stats['loss'] < best_loss:
                best_loss = train_stats['loss']
                if distributed_mode.is_main_process() and config['training'].get('save_best', True):
                    save_best_model(
                        args=create_train_args(config),
                        epoch=epoch,
                        model=model,
                        model_without_ddp=model_without_ddp,
                        optimizer=optimizer,
                        lr_schedule=lr_schedule,
                        loss_scaler=loss_scaler,
                        bestloss=best_loss,
                    )
                    logger.info(f"Saved best model (loss: {best_loss:.6f})")
        else:
            log_stats = {"epoch": epoch}
        
        # Save checkpoint
        save_freq = config['training'].get('save_frequency', 100)
        if distributed_mode.is_main_process() and ((epoch + 1) % save_freq == 0 or args.test_run):
            save_model(
                args=create_train_args(config),
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                lr_schedule=lr_schedule,
                loss_scaler=loss_scaler,
                epoch=epoch,
                bestloss=best_loss,
            )
            logger.info(f"Saved checkpoint at epoch {epoch}")
        
        # Write log file
        if distributed_mode.is_main_process():
            log_path = output_dir / "log.txt"
            with open(log_path, mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
        
        if args.test_run or args.eval_only:
            break
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Training completed in {total_time_str}")


class TrainArgs:
    """Module-level args object (picklable, so it can be stored in checkpoints)."""
    def __init__(self, config):
        self.output_dir = config['output'].get('output_dir', './output_dir')
        self.save_postfix = config['output'].get('save_postfix', 'FlowEDP')
        self.accum_iter = config['training'].get('accum_iter', 1)
        self.lr = config['training'].get('learning_rate', 0.0001)
        self.loss_type = config['training'].get('loss_type', 'l2')
        self.alpha = config['training'].get('alpha', 2.0)
        self.norm_rho = config['data'].get('normalize_density', False)
        self.discrete_flow_matching = config['model'].get('discrete_flow_matching', False)
        self.class_drop_prob = config['training'].get('class_drop_prob', 0.0)
        self.skewed_timesteps = config['ode'].get('skewed_timesteps', False)
        self.start_sad = config['training'].get('start_sad', False)
        self.test_run = config.get('test_run', False)


def create_train_args(config: dict):
    """Create an args object compatible with training functions."""
    return TrainArgs(config)


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Update config with command line args
    if args.resume:
        config.setdefault('training', {})['resume'] = args.resume
    if args.eval_only:
        config['eval_only'] = True
    if args.test_run:
        config['test_run'] = True
    
    # Run training
    train(config, args)


if __name__ == "__main__":
    main()
