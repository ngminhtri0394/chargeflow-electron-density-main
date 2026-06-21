#!/bin/bash
#SBATCH --job-name=edp_train_charged
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
# Charged Systems Training Template
# 
# Usage:
#   sbatch slurm_scripts/training/train_charged_systems.sh
#
# Specialized script for training on charged molecular systems.
# Uses the charged-specific configuration.
#
# Customize:
#   - Update paths in config/train_charged_config.yaml
#   - Adjust SBATCH parameters for your cluster
# ============================================================================

# Exit on error
set -e

# Configuration
CONFIG_FILE="config/train_charged_config.yaml"
CONDA_ENV="edp"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Set up distributed training environment
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=12355

# Print job information
echo "================================================================"
echo "Charged Systems Training"
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Start Time: $(date)"
echo "================================================================"

# Print distributed training setup
echo "Distributed Training Setup"
echo "================================================================"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "================================================================"

# Setup environment
echo "Setting up environment..."
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV

echo "Python: $(which python)"
echo "Python version: $(python --version)"

# Check GPU availability
echo "================================================================"
echo "GPU Information"
echo "================================================================"
nvidia-smi
echo "================================================================"

# Change to project directory
cd $PROJECT_ROOT

# Create logs directory
mkdir -p logs

# Run training
echo "Starting charged systems training..."
echo "Config: $CONFIG_FILE"
echo "================================================================"

srun python scripts/train.py \
    --config $CONFIG_FILE

# Print completion info
echo "================================================================"
echo "Training completed successfully!"
echo "End Time: $(date)"
echo "================================================================"
