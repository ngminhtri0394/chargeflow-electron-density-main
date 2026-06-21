#!/bin/bash
#SBATCH --job-name=edp_train_single_gpu
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=64G

# ============================================================================
# Single GPU Training Template
# 
# Usage:
#   sbatch slurm_scripts/training/train_single_gpu.sh
#
# Customize:
#   - Update CONFIG_FILE path
#   - Adjust SBATCH parameters above for your cluster
#   - Modify conda environment name
# ============================================================================

# Exit on error
set -e

# Configuration
CONFIG_FILE="config/train_config.yaml"
CONDA_ENV="edp"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Print job information
echo "================================================================"
echo "Job Information"
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "================================================================"

# Setup environment
echo "Setting up environment..."
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV

# Verify conda environment
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

# Create logs directory if it doesn't exist
mkdir -p logs

# Run training
echo "Starting training..."
echo "Config: $CONFIG_FILE"
echo "================================================================"

python scripts/train.py \
    --config $CONFIG_FILE

# Print completion info
echo "================================================================"
echo "Training completed successfully!"
echo "End Time: $(date)"
echo "================================================================"
