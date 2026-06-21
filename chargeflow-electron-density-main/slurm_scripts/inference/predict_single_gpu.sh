#!/bin/bash
#SBATCH --job-name=edp_predict
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --mem=32G

# ============================================================================
# Single GPU Inference Template
# 
# Usage:
#   sbatch slurm_scripts/inference/predict_single_gpu.sh
#
# Run inference on test data using a trained model.
#
# Customize:
#   - Update CONFIG_FILE and CHECKPOINT_PATH
#   - Adjust SBATCH parameters for your cluster
# ============================================================================

# Exit on error
set -e

# Configuration
CONFIG_FILE="config/inference_config.yaml"
CHECKPOINT_PATH="output_dir/checkpoint-best.pth"
CONDA_ENV="edp"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Print job information
echo "================================================================"
echo "Inference Job Information"
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
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
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
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

# Run inference
echo "Starting inference..."
echo "Config: $CONFIG_FILE"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "================================================================"

python scripts/predict.py \
    --config $CONFIG_FILE \
    --checkpoint $CHECKPOINT_PATH

# Print completion info
echo "================================================================"
echo "Inference completed successfully!"
echo "End Time: $(date)"
echo "================================================================"
