#!/bin/bash
# Complete Example: From Setup to Submission
# This script shows the complete workflow for submitting SLURM jobs using configurations

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════╗"
echo "║   SLURM Configuration System - Complete Example            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# ==============================================================================
# STEP 1: One-time setup (only needed once)
# ==============================================================================

echo "📋 STEP 1: Initial Setup (one time only)"
echo "─────────────────────────────────────────"
echo ""
echo "Edit your default configuration:"
echo "  vim slurm_scripts/config/slurm_defaults.yaml"
echo ""
echo "Update these key settings:"
echo "  • notifications.email: your.email@example.com"
echo "  • paths.data_root: /path/to/your/data"
echo "  • environment.conda_env: flowmatch"
echo "  • resources.*.partition: gpu  (check your cluster)"
echo ""
echo "Press Enter to continue to examples..."
read

# ==============================================================================
# EXAMPLE 1: Quick Test
# ==============================================================================

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║   EXAMPLE 1: Quick Test (Test your code changes)          ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

echo "Use case: You made code changes and want to test quickly"
echo ""
echo "Command:"
echo "────────"
cat <<'EOF'
python slurm_scripts/submit_job.py \
    --script training/train_single_gpu.sh \
    --preset quick_test \
    --dry-run
EOF

echo ""
echo "What this does:"
echo "  ✓ Uses minimal resources (1 GPU, 30 min)"
echo "  ✓ Small batch size (4)"
echo "  ✓ Only 5 epochs"
echo "  ✓ --dry-run: Shows what would run without submitting"
echo ""
echo "Run it? (y/n): "
read -r response
if [[ "$response" =~ ^[Yy]$ ]]; then
    python slurm_scripts/submit_job.py \
        --script training/train_single_gpu.sh \
        --preset quick_test \
        --dry-run
fi

echo ""
echo "Press Enter for next example..."
read

# ==============================================================================
# EXAMPLE 2: Medium Training Run
# ==============================================================================

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║   EXAMPLE 2: Medium Training (Single node, multiple GPUs) ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

echo "Use case: Regular training run on single node"
echo ""
echo "Command:"
echo "────────"
cat <<'EOF'
python slurm_scripts/submit_job.py \
    --script training/train_multi_gpu.sh \
    --batch-size 16 \
    --time 48:00:00
EOF

echo ""
echo "What this does:"
echo "  ✓ Uses default multi_gpu resources (4 GPUs)"
echo "  ✓ Overrides batch size to 16"
echo "  ✓ Sets time limit to 48 hours"
echo "  ✓ All other settings from slurm_defaults.yaml"
echo ""

echo "Press Enter for next example..."
read

# ==============================================================================
# EXAMPLE 3: Production Multi-Node Training
# ==============================================================================

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║   EXAMPLE 3: Production (Multi-node training)             ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

echo "Use case: Large-scale production training"
echo ""
echo "Command:"
echo "────────"
cat <<'EOF'
python slurm_scripts/submit_job.py \
    --script training/train_multi_node.sh \
    --preset production
EOF

echo ""
echo "What this does:"
echo "  ✓ Uses production preset (8 nodes, 96 hours)"
echo "  ✓ Large batch size (32)"
echo "  ✓ 200 epochs"
echo "  ✓ Email notifications enabled"
echo ""

echo "Press Enter for next example..."
read

# ==============================================================================
# EXAMPLE 4: Custom Configuration
# ==============================================================================

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║   EXAMPLE 4: Custom Config (For specific experiments)     ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

echo "Use case: You have a specific experiment configuration"
echo ""
echo "Step 1: Create custom config"
echo "────────"
cat <<'EOF'
cp slurm_scripts/config/my_config_template.yaml \
   slurm_scripts/config/experiment_A.yaml

vim slurm_scripts/config/experiment_A.yaml
# Edit: set learning_rate: 0.0002, batch_size: 24, etc.
EOF

echo ""
echo "Step 2: Use custom config"
echo "────────"
cat <<'EOF'
python slurm_scripts/submit_job.py \
    --config slurm_scripts/config/experiment_A.yaml \
    --script training/train_multi_gpu.sh
EOF

echo ""
echo "Benefits:"
echo "  ✓ Reproducible experiments"
echo "  ✓ Version control your configs"
echo "  ✓ Easy to rerun with same settings"
echo ""

echo "Press Enter for next example..."
read

# ==============================================================================
# EXAMPLE 5: Inference with Best Checkpoint
# ==============================================================================

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║   EXAMPLE 5: Inference (Generate predictions)             ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

echo "Use case: Run inference on test set after training"
echo ""
echo "Command:"
echo "────────"
cat <<'EOF'
python slurm_scripts/submit_job.py \
    --script inference/predict_single_gpu.sh \
    --checkpoint output_dir/checkpoint-best.pth
EOF

echo ""
echo "What this does:"
echo "  ✓ Uses inference resources (1 GPU, 12 hours)"
echo "  ✓ Loads your best checkpoint"
echo "  ✓ Generates predictions"
echo "  ✓ Saves metrics"
echo ""

echo "Press Enter for next example..."
read

# ==============================================================================
# EXAMPLE 6: Hyperparameter Sweep
# ==============================================================================

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║   EXAMPLE 6: Hyperparameter Sweep (Multiple jobs)         ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

echo "Use case: Try different hyperparameters"
echo ""
echo "Command:"
echo "────────"
cat <<'EOF'
# Loop over different learning rates
for lr in 0.0001 0.0002 0.0005; do
    python slurm_scripts/submit_job.py \
        --script training/train_multi_gpu.sh \
        --learning-rate $lr \
        --output-dir "output_lr_${lr}"
done
EOF

echo ""
echo "What this does:"
echo "  ✓ Submits 3 jobs with different learning rates"
echo "  ✓ Each uses separate output directory"
echo "  ✓ Easy to compare results"
echo ""

echo "Press Enter for next example..."
read

# ==============================================================================
# EXAMPLE 7: Charged Systems
# ==============================================================================

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║   EXAMPLE 7: Charged Systems (Special preset)             ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

echo "Use case: Training on charged electron density data"
echo ""
echo "Command:"
echo "────────"
cat <<'EOF'
python slurm_scripts/submit_job.py \
    --script training/train_charged_systems.sh \
    --preset charged_systems
EOF

echo ""
echo "What this does:"
echo "  ✓ Uses charged_systems preset"
echo "  ✓ Optimized for charged data"
echo "  ✓ Appropriate resources (4 GPUs, 48 hours)"
echo "  ✓ Points to charged data paths"
echo ""

echo "Press Enter to continue..."
read

# ==============================================================================
# SUMMARY
# ==============================================================================

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║   SUMMARY: Key Takeaways                                   ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

echo "💡 Key Commands:"
echo ""
echo "1. Test first (dry run):"
echo "   python slurm_scripts/submit_job.py --script SCRIPT --preset PRESET --dry-run"
echo ""
echo "2. Quick test:"
echo "   python slurm_scripts/submit_job.py --script training/train_single_gpu.sh --preset quick_test"
echo ""
echo "3. Production run:"
echo "   python slurm_scripts/submit_job.py --script training/train_multi_node.sh --preset production"
echo ""
echo "4. Override values:"
echo "   python slurm_scripts/submit_job.py --script SCRIPT --nodes 8 --time 48:00:00"
echo ""
echo "5. Custom config:"
echo "   python slurm_scripts/submit_job.py --config my_config.yaml --script SCRIPT"
echo ""

echo "📚 Documentation:"
echo ""
echo "  • Configuration guide:  slurm_scripts/config/README.md"
echo "  • SLURM scripts guide:  slurm_scripts/README.md"
echo "  • Main project docs:    README.md"
echo ""

echo "✅ Benefits of Using Configs:"
echo ""
echo "  ✓ No more editing scripts manually"
echo "  ✓ Reproducible experiments"
echo "  ✓ Easy to share settings"
echo "  ✓ Version control friendly"
echo "  ✓ Quick to test and iterate"
echo ""

echo "🚀 Ready to submit your first job!"
echo ""
echo "Next steps:"
echo "  1. Edit slurm_scripts/config/slurm_defaults.yaml"
echo "  2. Test with: python slurm_scripts/submit_job.py --script training/train_single_gpu.sh --preset quick_test --dry-run"
echo "  3. Run for real (remove --dry-run)"
echo ""
