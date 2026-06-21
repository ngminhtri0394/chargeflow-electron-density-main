#!/bin/bash
# Run inference on all test datasets
# Usage: ./run_all_inference.sh <checkpoint_path>

if [ $# -eq 0 ]; then
    echo "Usage: $0 <checkpoint_path>"
    echo "Example: $0 output_dir/checkpoint-best.pth"
    exit 1
fi

CHECKPOINT=$1
CONFIG_DIR="config"

if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

echo "Running inference on all test datasets..."
echo "Checkpoint: $CHECKPOINT"
echo ""

# Extreme test sets
echo "=== Extreme Test Sets ==="
for testset in extreme_mofs extreme_electrene extreme_organic; do
    echo "Running inference on: $testset"
    python scripts/predict.py \
        --config ${CONFIG_DIR}/inference_${testset}.yaml \
        --checkpoint $CHECKPOINT
    echo ""
done

# Standard test sets
echo "=== Standard Test Sets ==="
for testset in mofs electrenes organic organic_molecules perovskites; do
    echo "Running inference on: $testset"
    python scripts/predict.py \
        --config ${CONFIG_DIR}/inference_${testset}.yaml \
        --checkpoint $CHECKPOINT
    echo ""
done

# Defect test sets
echo "=== Defect Test Sets ==="
for testset in special_defects_diamond multisite_defects_diamond; do
    echo "Running inference on: $testset"
    python scripts/predict.py \
        --config ${CONFIG_DIR}/inference_${testset}.yaml \
        --checkpoint $CHECKPOINT
    echo ""
done

echo "All inference runs complete!"
echo "Results saved in predictions/ directory"
