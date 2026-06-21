#!/bin/bash
# Example: How to set up your SLURM configuration for the first time

echo "🔧 SLURM Configuration Setup"
echo "============================="
echo ""
echo "This script helps you set up your SLURM configuration."
echo "Edit slurm_scripts/config/slurm_defaults.yaml to customize these settings:"
echo ""

# Get current user
USER=$(whoami)
CURRENT_DIR=$(pwd)

echo "📝 Configuration Items to Update:"
echo ""
echo "1. Email Notifications"
echo "   notifications:"
echo "     email: \"$USER@example.com\"  # <- Change this!"
echo ""

echo "2. Data Paths"
echo "   paths:"
echo "     data_root: \"/path/to/your/data\"  # <- Update this!"
echo "     # Current directory: $CURRENT_DIR"
echo ""

echo "3. Conda Environment"
echo "   environment:"
echo "     conda_env: \"flowmatch\"  # <- Change if different"
echo ""

echo "4. Cluster Resources (check your cluster defaults)"
echo "   resources:"
echo "     multi_gpu:"
echo "       partition: \"gpu\"      # <- Check your cluster partition names"
echo "       nodes: 1"
echo "       gpus_per_node: 4        # <- Adjust based on available GPUs"
echo ""

echo "📂 Quick Edit Command:"
echo "   vim slurm_scripts/config/slurm_defaults.yaml"
echo ""

echo "✅ After editing, test your configuration:"
echo "   python slurm_scripts/submit_job.py --script training/train_single_gpu.sh --dry-run"
echo ""

echo "📖 For more help:"
echo "   cat slurm_scripts/config/README.md"
echo ""
