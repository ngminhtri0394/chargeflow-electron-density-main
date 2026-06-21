# HPC/SLURM Job Management

Complete guide for running electron density prediction on HPC clusters using SLURM.

**Quick Navigation:**
- [Overview](#overview)
- [Setup](#setup-one-time)
- [Method 1: Template-Based](#method-1-template-based-submission)
- [Method 2: Named Jobs](#method-2-named-job-submission)
- [Examples](#examples)
- [Monitoring](#monitoring-jobs)
- [Troubleshooting](#troubleshooting)
- [Reference](#reference)

---

## Overview

### Two Job Submission Methods

| Feature | Template-Based | Named Jobs |
|---------|---------------|------------|
| **Script** | `submit_job.py` | `submit_named_job.py` |
| **Best For** | Flexible resources, testing | Reproducible runs, saved configs |
| **Resources** | CLI args or presets | Stored in `jobs.yaml` |
| **Testsets** | CLI override | Stored with exact names |
| **Use Case** | Quick experiments | Production, standard tests |

**Recommendation:** Use **Template** for exploration, **Named** for production.

### Directory Structure

```
slurm_scripts/
├── config/
│   ├── slurm_defaults.yaml      # Default SLURM settings
│   ├── jobs.yaml                 # Named job configs
│   └── presets/                  # Resource presets (3)
│       ├── quick_test.yaml
│       ├── production.yaml
│       └── charged_systems.yaml
│
├── training/                     # Training scripts (4)
│   ├── train_single_gpu.sh
│   ├── train_multi_gpu.sh
│   ├── train_multi_node.sh
│   └── train_charged_systems.sh
│
├── inference/                    # Inference scripts (3)
│   ├── predict_single_gpu.sh
│   ├── predict_batch.sh
│   └── predict_charged_systems.sh
│
├── submit_job.py                 # Template submission
└── submit_named_job.py           # Named submission
```

---

## Setup (One-Time)

### 1. Configure Defaults

Edit `config/slurm_defaults.yaml`:

```yaml
# REQUIRED: Update these for your cluster
notifications:
  email: "your.email@example.com"  # ⚠️ Change!

paths:
  data_root: "/path/to/your/data"  # ⚠️ Change!

environment:
  conda_env: "flowmatch"  # ⚠️ Verify name

# VERIFY: These match your cluster
resources:
  multi_gpu:
    partition: "gpu"  # Check cluster docs
    gpus_per_node: 4  # Typical value
    mem_per_cpu: "4G"
```

### 2. Test Configuration

```bash
# Dry run to verify settings
python submit_job.py --script training/train_single_gpu.sh --preset quick_test --dry-run
```

### 3. Configure Named Jobs (Optional)

Edit `config/jobs.yaml` to add your recurring experiments. See examples inside.

---

## Method 1: Template-Based Submission

**Use when:** You need flexible resource allocation or testing different configurations.

### Basic Usage

```bash
# Using preset
python submit_job.py --script training/train_single_gpu.sh --preset quick_test

# Production run
python submit_job.py --script training/train_multi_gpu.sh --preset production
```

### Available Presets

| Preset | Nodes | GPUs | Time | Use Case |
|--------|-------|------|------|----------|
| `quick_test` | 1 | 1 | 30 min | Fast testing |
| `production` | 8 | 4/node | 96 hrs | Full training |
| `charged_systems` | 1 | 4 | 48 hrs | Charged training |

### Override Resources

```bash
# Custom resources
python submit_job.py \
    --script training/train_multi_node.sh \
    --nodes 16 \
    --time 72:00:00 \
    --gpus 8 \
    --batch-size 64
```

### All Options

```bash
python submit_job.py \
    --script SCRIPT                # Required: Script path
    --preset NAME                  # Resource preset
    --nodes N                      # Number of nodes
    --gpus N                       # GPUs per node
    --time HH:MM:SS                # Wall time
    --partition NAME               # SLURM partition
    --job-name NAME                # Job name
    --config PATH                  # Training config
    --checkpoint PATH              # Model checkpoint
    --testset NAME                 # Test dataset
    --output-dir PATH              # Output directory
    --batch-size N                 # Batch size
    --learning-rate LR             # Learning rate
    --num-epochs N                 # Training epochs
    --ode-method NAME              # ODE solver
    --dry-run                      # Show without submitting
```

---

## Method 2: Named Job Submission

**Use when:** You have standard configurations you run repeatedly.

### List Available Jobs

```bash
python submit_named_job.py --list
```

Example output:
```
Training Jobs:
  - neutral_standard: Standard neutral system training
  - charged_standard: Standard charged system training
  
Prediction Jobs:
  - neutral_extreme_mofs: Predict on extreme MOFs (testset: extreme_mofs)
  - neutral_extreme_organic: Predict on extreme organic (testset: extreme_organic)
  - charged_perovskites: Predict on charged perovskites (testset: perovskites)
```

### Submit Named Job

```bash
# Basic submission
python submit_named_job.py --job neutral_extreme_mofs

# Override testset
python submit_named_job.py --job neutral_extreme_mofs --testset extreme_electrene

# Override time
python submit_named_job.py --job neutral_extreme_mofs --time 12:00:00

# Dry run
python submit_named_job.py --job neutral_extreme_mofs --dry-run
```

### Define Your Named Jobs

Edit `config/jobs.yaml`:

```yaml
# Variables for reuse
variables:
  configs:
    train_config: "../config/train_config.yaml"
  checkpoints:
    my_model: "/path/to/checkpoint-best.pth"

# Training jobs
training_jobs:
  my_experiment:
    config: "${configs.train_config}"
    resources:
      nodes: 2
      gpus_per_node: 4
      time: "24:00:00"
    script_args:
      batch_size: 32
      learning_rate: 0.0002
      output_dir: "output_my_experiment"

# Prediction jobs
prediction_jobs:
  test_my_model:
    testset: "extreme_mofs"  # ⚠️ Exact testset name
    checkpoint: "${checkpoints.my_model}"
    resources:
      gpus: 1
      time: "10:00:00"
    script_args:
      batch_size: 1
      ode_method: "heun2"
      output_dir: "predictions_mofs"
```

**Key Features:**
- **Exact testset names** stored (e.g., `extreme_mofs`, `special_defects_diamond`)
- **Checkpoint paths** saved for reuse
- **Variables** for common paths
- **Full reproducibility**

---

## Examples

### Training

#### Quick Test (30 min)
```bash
# Template
python submit_job.py --script training/train_single_gpu.sh --preset quick_test

# Named
python submit_named_job.py --job quick_test_neutral
```

#### Single GPU (24 hrs)
```bash
python submit_job.py --script training/train_single_gpu.sh \
    --time 24:00:00 \
    --config ../config/train_config.yaml \
    --output-dir output_single_gpu
```

#### Multi-GPU Single Node (4 GPUs, 48 hrs)
```bash
python submit_job.py --script training/train_multi_gpu.sh \
    --gpus 4 \
    --time 48:00:00 \
    --batch-size 32
```

#### Multi-Node (8 nodes, 96 hrs)
```bash
# Using preset
python submit_job.py --script training/train_multi_node.sh --preset production

# Custom
python submit_job.py --script training/train_multi_node.sh \
    --nodes 8 \
    --gpus 4 \
    --time 96:00:00 \
    --batch-size 64
```

#### Charged Systems
```bash
# Using preset
python submit_job.py --script training/train_charged_systems.sh --preset charged_systems

# Using named job
python submit_named_job.py --job charged_standard
```

### Inference

#### Single Testset
```bash
python submit_job.py --script inference/predict_single_gpu.sh \
    --checkpoint output_dir/checkpoint-best.pth \
    --testset extreme_mofs \
    --time 10:00:00
```

#### Multiple Testsets (using named jobs)
```bash
# Submit all
python submit_named_job.py --job neutral_extreme_mofs
python submit_named_job.py --job neutral_extreme_organic
python submit_named_job.py --job neutral_extreme_electrene

# Or with override
python submit_named_job.py --job my_test --testset special_defects_diamond
```

#### Batch Prediction
```bash
python submit_job.py --script inference/predict_batch.sh \
    --checkpoint output_dir/checkpoint-best.pth \
    --testset mofs \
    --batch-size 4 \
    --time 24:00:00
```

---

## Available Testsets

For `--testset` option or `testset:` in `jobs.yaml`:

**Extreme (3):** `extreme_mofs`, `extreme_electrene`, `extreme_organic`

**Standard (5):** `mofs`, `electrenes`, `organic`, `organic_molecules`, `perovskites`

**Defects (2):** `special_defects_diamond`, `multisite_defects_diamond`

**See `../data_lists/README.md` for complete dataset documentation.**

---

## Monitoring Jobs

### Check Status
```bash
squeue -u $USER
```

### View Logs
```bash
# SLURM output
tail -f logs/edp_train_*.out

# SLURM errors
tail -f logs/edp_train_*.err

# Application logs
tail -f output_dir/training.log
tail -f output_dir/log.txt
```

### Cancel Job
```bash
scancel JOB_ID
```

### Job Info
```bash
scontrol show job JOB_ID
```

---

## Troubleshooting

### Job Fails Immediately

**1. Check error log:**
```bash
cat logs/edp_train_*.err
```

**2. Common issues:**
- Wrong email/paths in `slurm_defaults.yaml`
- Wrong conda environment name
- Wrong partition name
- Data files not found

**3. Test with dry-run:**
```bash
python submit_job.py --script SCRIPT --dry-run
```

### Out of Memory

**Reduce batch size:**
```bash
python submit_job.py --script SCRIPT --batch-size 8
```

**Or in named job:**
```yaml
script_args:
  batch_size: 8
```

**Or add gradient accumulation in config:**
```yaml
training:
  accum_iter: 4
```

### Wrong Partition

**Override:**
```bash
python submit_job.py --script SCRIPT --partition your_gpu_partition
```

**Or update default:**
```yaml
# In slurm_defaults.yaml
resources:
  multi_gpu:
    partition: "correct_partition"
```

### Checkpoint Not Found

**Check path:**
```bash
ls -lh output_dir/checkpoint-best.pth
```

**Use absolute path:**
```bash
python submit_job.py --script SCRIPT \
    --checkpoint /absolute/path/to/checkpoint-best.pth
```

### Data Not Found

**Verify data root:**
```yaml
# In slurm_defaults.yaml
paths:
  data_root: "/correct/path/to/data"
```

**Check list files:**
```bash
head ../data_lists/training/list_d_charged_train
```

---

## Reference

### Resource Presets

**`config/presets/quick_test.yaml`:**
```yaml
nodes: 1
gpus_per_node: 1
time: "00:30:00"
```

**`config/presets/production.yaml`:**
```yaml
nodes: 8
gpus_per_node: 4
time: "96:00:00"
```

**`config/presets/charged_systems.yaml`:**
```yaml
nodes: 1
gpus_per_node: 4
time: "48:00:00"
```

### Job Config Schema

In `config/jobs.yaml`:

```yaml
job_name:
  # For training
  config: "path/to/config.yaml"
  
  # For prediction
  testset: "testset_name"
  checkpoint: "path/to/checkpoint"
  
  # Resources
  resources:
    nodes: 1
    gpus_per_node: 4
    time: "24:00:00"
    partition: "gpu"  # Optional
  
  # Script arguments
  script_args:
    batch_size: 32
    learning_rate: 0.0001
    output_dir: "output"
    ode_method: "heun2"
    # Any other script arg
```

### Training Scripts

| Script | Use Case | Nodes | GPUs/Node |
|--------|----------|-------|-----------|
| `train_single_gpu.sh` | Testing, small datasets | 1 | 1 |
| `train_multi_gpu.sh` | Single node training | 1 | 4 |
| `train_multi_node.sh` | Large-scale training | 8+ | 4 |
| `train_charged_systems.sh` | Charged systems | 1 | 4 |

### Inference Scripts

| Script | Use Case | GPUs |
|--------|----------|------|
| `predict_single_gpu.sh` | Single testset | 1 |
| `predict_batch.sh` | Large testsets | 1-4 |
| `predict_charged_systems.sh` | Charged systems | 1 |

### Best Practices

1. ✅ **Always test locally first** with `--test-run`
2. ✅ **Use dry-run** before submitting: `--dry-run`
3. ✅ **Start with quick_test** to verify setup
4. ✅ **Use named jobs** for reproducibility
5. ✅ **Monitor logs** regularly
6. ✅ **Keep checkpoints** for resuming
7. ✅ **Use exact testset names** in named jobs
8. ✅ **Store checkpoint paths** in jobs.yaml

---

## Quick Reference Card

```bash
# Template submission
python submit_job.py --script SCRIPT --preset PRESET [--dry-run]

# Named submission
python submit_named_job.py --job JOB_NAME [--testset NAME] [--dry-run]

# List jobs
python submit_named_job.py --list

# Monitor
squeue -u $USER
tail -f logs/*.out

# Cancel
scancel JOB_ID
```

---

**See main `../README.md` for complete project documentation.**
**See `../data_lists/README.md` for dataset details.**
