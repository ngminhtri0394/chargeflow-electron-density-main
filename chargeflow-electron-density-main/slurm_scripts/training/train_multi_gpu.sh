#!/bin/bash
#SBATCH --job-name=edp_train_multi_gpu
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --mem=256G

# ============================================================================
# Multi-GPU Single Node Training Template
# 
# Usage:
#   sbatch slurm_scripts/training/train_multi_gpu.sh
#
# This script uses PyTorch DistributedDataParallel for multi-GPU training
# on a single node.
#
# Customize:
#   - Update CONFIG_FILE path
#   - Adjust number of GPUs (--gres=gpu:N)
#   - Adjust SBATCH parameters for your cluster
# ============================================================================

# Exit on error
set -e

# Configuration
CONFIG_FILE="config/train_config.yaml"
CONDA_ENV="edp"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Set up distributed training environment
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=12355

# Print job information
echo "================================================================"
echo "Job Information"
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Number of Nodes: $SLURM_NNODES"
echo "GPUs per Node: $SLURM_GPUS_ON_NODE"
echo "Tasks per Node: $SLURM_NTASKS_PER_NODE"
echo "CPUs per Task: $SLURM_CPUS_PER_TASK"
echo "Start Time: $(date)"
echo "================================================================"

# Print distributed training setup
echo "Distributed Training Setup"
echo "================================================================"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "SLURM_PROCID: $SLURM_PROCID"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "================================================================"

# Setup environment
echo "Setting up environment..."
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV

# Verify environment
echo "Python: $(which python)"
echo "Python version: $(python --version)"

# Check GPU availability
echo "================================================================"
echo "GPU Information"
echo "================================================================"
nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "================================================================"

# Change to project directory
cd $PROJECT_ROOT

# Create logs directory
mkdir -p logs

# Run distributed training
echo "Starting multi-GPU training..."
echo "Config: $CONFIG_FILE"
echo "================================================================"

srun python scripts/train.py \
    --config $CONFIG_FILE

# Print completion info
echo "================================================================"
echo "Training completed successfully!"
echo "End Time: $(date)"
echo "================================================================"
