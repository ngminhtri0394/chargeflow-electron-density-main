#!/bin/bash
#SBATCH --job-name=edp_predict_charged
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
# Charged Systems Inference Template
# 
# Usage:
#   sbatch slurm_scripts/inference/predict_charged_systems.sh
#
# Run inference on charged molecular systems.
# Extracts charge levels from filenames for conditioning.
#
# Customize:
#   - Update CONFIG_FILE and CHECKPOINT_PATH
#   - Ensure data filenames contain charge information
# ============================================================================

# Exit on error
set -e

# Configuration
CONFIG_FILE="config/inference_config.yaml"
CHECKPOINT_PATH="output_dir_charged/checkpoint-best.pth"
OUTPUT_DIR="predictions_charged"
CONDA_ENV="edp"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Print job information
echo "================================================================"
echo "Charged Systems Inference"
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

# Check GPU availability
echo "================================================================"
echo "GPU Information"
echo "================================================================"
nvidia-smi
echo "================================================================"

# Change to project directory
cd $PROJECT_ROOT

# Create logs and output directories
mkdir -p logs
mkdir -p $OUTPUT_DIR

# Verify checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT_PATH"
    exit 1
fi

# Run inference
echo "Starting charged systems inference..."
echo "Config: $CONFIG_FILE"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Output: $OUTPUT_DIR"
echo "================================================================"

python scripts/predict.py \
    --config $CONFIG_FILE \
    --checkpoint $CHECKPOINT_PATH \
    --output-dir $OUTPUT_DIR

# Print completion info
echo "================================================================"
echo "Charged systems inference completed successfully!"
echo "End Time: $(date)"
echo "Results saved to: $OUTPUT_DIR"
echo "================================================================"
