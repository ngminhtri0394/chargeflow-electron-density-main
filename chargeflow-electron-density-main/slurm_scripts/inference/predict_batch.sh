#!/bin/bash
#SBATCH --job-name=edp_predict_batch
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --mem=128G

# ============================================================================
# Multi-GPU Batch Inference Template
# 
# Usage:
#   sbatch slurm_scripts/inference/predict_batch.sh
#
# Run inference on large datasets using multiple GPUs for faster processing.
#
# Customize:
#   - Update CONFIG_FILE and CHECKPOINT_PATH
#   - Adjust batch_size in config file
#   - Adjust number of GPUs (--gres=gpu:N)
# ============================================================================

# Exit on error
set -e

# Configuration
CONFIG_FILE="config/inference_config.yaml"
CHECKPOINT_PATH="output_dir/checkpoint-best.pth"
CONDA_ENV="edp"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Set up distributed inference
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=12355

# Print job information
echo "================================================================"
echo "Batch Inference Job Information"
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Start Time: $(date)"
echo "================================================================"

# Setup environment
echo "Setting up environment..."
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV

echo "Python: $(which python)"

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

# Verify checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT_PATH"
    exit 1
fi

# Run distributed inference
echo "Starting batch inference..."
echo "Config: $CONFIG_FILE"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "================================================================"

srun python scripts/predict.py \
    --config $CONFIG_FILE \
    --checkpoint $CHECKPOINT_PATH

# Print completion info
echo "================================================================"
echo "Batch inference completed successfully!"
echo "End Time: $(date)"
echo "================================================================"
