#!/bin/bash
#SBATCH --job-name=edp_train_multi_node
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --mem=256G

# ============================================================================
# Multi-Node Multi-GPU Training Template
# 
# Usage:
#   sbatch slurm_scripts/training/train_multi_node.sh
#
# This script uses PyTorch DistributedDataParallel for multi-node training.
# Total GPUs = nodes * gpus_per_node
#
# Customize:
#   - Update CONFIG_FILE path
#   - Adjust number of nodes (--nodes=N)
#   - Adjust GPUs per node (--gres=gpu:N)
#   - Update partition and resource limits for your cluster
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
echo "Nodes: $SLURM_NODELIST"
echo "Number of Nodes: $SLURM_NNODES"
echo "GPUs per Node: $SLURM_GPUS_ON_NODE"
echo "Total GPUs: $((SLURM_NNODES * SLURM_GPUS_ON_NODE))"
echo "Tasks per Node: $SLURM_NTASKS_PER_NODE"
echo "Total Tasks: $SLURM_NTASKS"
echo "CPUs per Task: $SLURM_CPUS_PER_TASK"
echo "Start Time: $(date)"
echo "================================================================"

# Print distributed training setup
echo "Distributed Training Setup"
echo "================================================================"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "World Size: $SLURM_NTASKS"
echo "================================================================"

# Setup environment
echo "Setting up environment..."
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV

# Verify environment
echo "Python: $(which python)"
echo "Python version: $(python --version)"

# Check GPU availability on master node
if [ $SLURM_PROCID -eq 0 ]; then
    echo "================================================================"
    echo "GPU Information (Master Node)"
    echo "================================================================"
    nvidia-smi
    echo "================================================================"
fi

# Change to project directory
cd $PROJECT_ROOT

# Create logs directory
mkdir -p logs

# Run distributed training
echo "Starting multi-node training..."
echo "Config: $CONFIG_FILE"
echo "================================================================"

srun python scripts/train.py \
    --config $CONFIG_FILE

# Print completion info (only on master)
if [ $SLURM_PROCID -eq 0 ]; then
    echo "================================================================"
    echo "Training completed successfully!"
    echo "End Time: $(date)"
    echo "================================================================"
fi
