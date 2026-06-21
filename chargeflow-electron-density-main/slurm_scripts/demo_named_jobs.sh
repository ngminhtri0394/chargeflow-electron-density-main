#!/bin/bash
# Demo: Named Job Configuration System
# This shows how to use your exact job configurations

echo "╔════════════════════════════════════════════════════════════╗"
echo "║   Named Job Configuration Demo                             ║"
echo "║   Store & Reuse Your Exact Job Settings                    ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# ==============================================================================
# PART 1: Show Available Jobs
# ==============================================================================

echo "📋 PART 1: See All Your Configured Jobs"
echo "══════════════════════════════════════════════════════════════"
echo ""
echo "Command:"
echo "  python slurm_scripts/submit_named_job.py --list"
echo ""
echo "This shows all jobs defined in config/jobs.yaml including:"
echo "  • testset names (special_defects_diamond, extreme_mofs, etc.)"
echo "  • checkpoint paths"
echo "  • resource allocations"
echo "  • all parameters"
echo ""
echo "Try it? (y/n): "
read -r response
if [[ "$response" =~ ^[Yy]$ ]]; then
    python slurm_scripts/submit_named_job.py --list
fi

echo ""
echo "Press Enter to continue..."
read

# ==============================================================================
# PART 2: Submit Named Job
# ==============================================================================

echo ""
echo "🚀 PART 2: Submit a Specific Job by Name"
echo "══════════════════════════════════════════════════════════════"
echo ""
echo "Your original script:"
echo "  sbatch script/predict_neutral_sad_extreme_mofs.sh"
echo ""
echo "Which contained:"
echo "  --testset extreme_mofs"
echo "  --checkpoint /vast/.../best_model-FlowEDP_sad_neutral_xlarge_hybrid_loss_eval_test_set_normal_timestep_lr_00005_3.pth"
echo "  --dataset unet_uncond_neutral_xlarge"
echo "  --ode_method heun2"
echo "  --ode_options '{\"nfe\": 50}'"
echo "  ... and more settings ..."
echo ""
echo "New way (all settings loaded from config!):"
echo "  python slurm_scripts/submit_named_job.py --job neutral_extreme_mofs"
echo ""
echo "Let's do a dry run to see what it would execute:"
echo ""
echo "Try it? (y/n): "
read -r response
if [[ "$response" =~ ^[Yy]$ ]]; then
    python slurm_scripts/submit_named_job.py --job neutral_extreme_mofs --dry-run
fi

echo ""
echo "Press Enter to continue..."
read

# ==============================================================================
# PART 3: Override Specific Values
# ==============================================================================

echo ""
echo "🎯 PART 3: Keep Config but Change One Thing"
echo "══════════════════════════════════════════════════════════════"
echo ""
echo "Use case: You want to run the same job but on a different testset"
echo ""
echo "Original config has: testset=extreme_mofs"
echo "But you want to run on: testset=special_defects_diamond"
echo ""
echo "Command:"
echo "  python slurm_scripts/submit_named_job.py \\"
echo "      --job neutral_extreme_mofs \\"
echo "      --testset special_defects_diamond"
echo ""
echo "This keeps ALL other settings (checkpoint, ODE method, batch size, etc.)"
echo "but changes ONLY the testset!"
echo ""
echo "Try dry run? (y/n): "
read -r response
if [[ "$response" =~ ^[Yy]$ ]]; then
    python slurm_scripts/submit_named_job.py \
        --job neutral_extreme_mofs \
        --testset special_defects_diamond \
        --dry-run
fi

echo ""
echo "Press Enter to continue..."
read

# ==============================================================================
# PART 4: Add Your Own Job Config
# ==============================================================================

echo ""
echo "➕ PART 4: Add Your Own Job Configuration"
echo "══════════════════════════════════════════════════════════════"
echo ""
echo "To add your own job, edit: config/jobs.yaml"
echo ""
echo "Example: Add a new prediction job"
echo "────────────────────────────────────────"
cat <<'EOF'

prediction_jobs:
  # Add your new job here
  my_perovskites_prediction:
    job_name: "FlowEDP_perovskites_prediction"
    checkpoint: "${checkpoints.sad_neutral_xlarge_hybrid_loss}"
    dataset: "unet_uncond_neutral_xlarge"
    testset: "perovskites"  # Your specific testset name!
    
    resources:
      nodes: 1
      gpus: 1
      cpus_per_task: 5
      time: "5:00:00"
    
    script_args:
      start_sad: true
      eval_only: true
      batch_size: 1
      use_ema: true
      ode_method: "heun2"
      ode_options: '{"nfe": 50}'

EOF

echo ""
echo "Then use it:"
echo "  python slurm_scripts/submit_named_job.py --job my_perovskites_prediction"
echo ""

echo "Press Enter to continue..."
read

# ==============================================================================
# PART 5: Add New Testsets
# ==============================================================================

echo ""
echo "📊 PART 5: Add New Test Sets"
echo "══════════════════════════════════════════════════════════════"
echo ""
echo "To add new testsets, edit config/jobs.yaml:"
echo ""
cat <<'EOF'

testsets:
  special_defects_diamond:
    name: "special_defects_diamond"
    description: "Special defects in diamond structures"
  
  extreme_mofs:
    name: "extreme_mofs"
    description: "Extreme metal-organic frameworks"
  
  # Add your new testsets here
  perovskites:
    name: "perovskites"
    description: "Perovskite materials"
  
  my_custom_testset:
    name: "my_custom_testset"
    description: "My custom test set description"

EOF

echo ""
echo "Then you can use them in any job:"
echo "  python slurm_scripts/submit_named_job.py --job ANY_JOB --testset perovskites"
echo ""

echo "Press Enter to continue..."
read

# ==============================================================================
# PART 6: Multiple Testsets Workflow
# ==============================================================================

echo ""
echo "🔄 PART 6: Run Same Job on Multiple Testsets"
echo "══════════════════════════════════════════════════════════════"
echo ""
echo "Use case: Test your model on all testsets"
echo ""
echo "Command:"
cat <<'EOF'

for testset in special_defects_diamond extreme_mofs perovskites; do
    echo "Submitting job for testset: $testset"
    python slurm_scripts/submit_named_job.py \
        --job neutral_extreme_mofs \
        --testset $testset
done

EOF

echo ""
echo "This submits 3 jobs:"
echo "  • Same checkpoint"
echo "  • Same ODE method"
echo "  • Same batch size"
echo "  • Different testset each time"
echo ""

echo "Press Enter to continue..."
read

# ==============================================================================
# SUMMARY
# ==============================================================================

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║   SUMMARY                                                   ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

echo "✅ What You Can Store in config/jobs.yaml:"
echo ""
echo "  • Exact testset names (special_defects_diamond, extreme_mofs, etc.)"
echo "  • Checkpoint paths"
echo "  • ODE methods and parameters"
echo "  • Batch sizes"
echo "  • Resource allocations"
echo "  • Any other job-specific settings"
echo ""

echo "✅ How to Use:"
echo ""
echo "  1. List jobs:"
echo "     python slurm_scripts/submit_named_job.py --list"
echo ""
echo "  2. Submit a job:"
echo "     python slurm_scripts/submit_named_job.py --job JOB_NAME"
echo ""
echo "  3. Override values:"
echo "     python slurm_scripts/submit_named_job.py --job JOB_NAME --testset OTHER_TESTSET"
echo ""
echo "  4. Test first:"
echo "     python slurm_scripts/submit_named_job.py --job JOB_NAME --dry-run"
echo ""

echo "✅ Benefits:"
echo ""
echo "  ✓ No more typing testset names every time"
echo "  ✓ No more looking up checkpoint paths"
echo "  ✓ No more remembering ODE parameters"
echo "  ✓ Just reference jobs by name"
echo "  ✓ Easy to run same job on different testsets"
echo "  ✓ Configuration is version controlled"
echo ""

echo "📚 Documentation:"
echo ""
echo "  • Complete guide:     slurm_scripts/NAMED_JOBS_GUIDE.md"
echo "  • Which system to use: slurm_scripts/WHICH_SYSTEM.md"
echo "  • Config reference:   slurm_scripts/CONFIG_QUICK_REF.md"
echo ""

echo "🎯 Quick Examples:"
echo ""
echo "  # Submit exact job (testset, checkpoint, all params loaded)"
echo "  python slurm_scripts/submit_named_job.py --job neutral_special_defects_diamond"
echo ""
echo "  # Same job, different testset"
echo "  python slurm_scripts/submit_named_job.py --job neutral_extreme_mofs --testset special_defects_diamond"
echo ""
echo "  # All testsets in a loop"
echo "  for t in special_defects_diamond extreme_mofs; do"
echo "    python slurm_scripts/submit_named_job.py --job neutral_extreme_mofs --testset \$t"
echo "  done"
echo ""

echo "🚀 Ready to submit jobs without typing everything!"
echo ""
