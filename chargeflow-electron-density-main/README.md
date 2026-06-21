# ChargeFlow Code

Standalone training and inference code for ChargeFlow, a flow-matching model for charge-conditioned electron density prediction.

**Table of Contents:**
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration-system)
- [Dataset Management](#-dataset-management)
- [HPC/SLURM Jobs](#-hpcslurm-job-management)
- [Training](#-training-details)
- [Monitoring](#-monitoring-and-evaluation)
- [Workflows](#-common-workflows)
- [Troubleshooting](#-troubleshooting)
- [API Reference](#-api-reference)

---

## 🚀 Quick Start

### Installation
```bash
pip install -e .
```

### Test Locally
```bash
# Training test
python scripts/train.py --config config/train_config.yaml --test-run

# Inference test  
python scripts/predict.py --config config/inference_config.yaml \
    --checkpoint output_dir/checkpoint-best.pth --test-run
```

### Full Training
```bash
# Single GPU
python scripts/train.py --config config/train_config.yaml

# Multi-GPU (4 GPUs)
torchrun --nproc_per_node=4 scripts/train.py --config config/train_config.yaml
```

### HPC/SLURM
```bash
# Quick test
python slurm_scripts/submit_job.py --script training/train_single_gpu.sh --preset quick_test

# Production
python slurm_scripts/submit_job.py --script training/train_multi_node.sh --preset production

# Named job
python slurm_scripts/submit_named_job.py --job neutral_extreme_mofs
```

---

## 📁 Project Structure

```
chargeflow-electron-density/
├── src/                          # Source modules (24 files)
│   ├── data/dataset.py          # Dataset classes
│   ├── models/                  # 6 U-Net architectures
│   ├── training/                # 11 training modules
│   └── utils/                   # 4 utility modules
│
├── scripts/                      # Executable scripts
│   ├── train.py                 # Training
│   └── predict.py               # Inference
│
├── config/                       # YAML configs (3 files)
│   ├── train_config.yaml
│   ├── train_charged_config.yaml
│   └── inference_config.yaml
│
├── data_lists/                   # Dataset lists (93 files)
│   ├── training/                # Train/val/test (46 files)
│   ├── test_sets/               # Test datasets (40 files)
│   ├── stats/                   # Statistics (8 files)
│   ├── datasets.yaml            # Dataset mapping
│   ├── list_helper.py           # Helper script
│   └── README.md                # Documentation
│
├── slurm_scripts/                # HPC job management
│   ├── config/                  # SLURM configs
│   │   ├── slurm_defaults.yaml # Settings
│   │   ├── jobs.yaml           # Named jobs
│   │   └── presets/            # Presets (3)
│   ├── training/                # Training scripts (4)
│   ├── inference/               # Inference scripts (3)
│   ├── submit_job.py            # Template submission
│   ├── submit_named_job.py      # Named submission
│   └── README.md                # Documentation
│
└── tests/                        # Unit tests (2)
```

The manuscript sources used for the paper submission are intentionally excluded from this repository snapshot.

---

## 📋 Configuration System

### Training Config Example

`config/train_config.yaml`:
```yaml
model:
  name: "unet3d"
  in_channels: 1
  out_channels: 1
  base_channels: 32

data:
  train_data_list: "data_lists/training/list_d_charged_train"
  train_label_list: "data_lists/training/list_l_charged_train"
  train_data_gridsize: "data_lists/training/list_dgs_charged_train"
  train_label_gridsize: "data_lists/training/list_lgs_charged_train"
  val_data_list: "data_lists/training/list_d_charged_val"
  val_label_list: "data_lists/training/list_l_charged_val"
  val_data_gridsize: "data_lists/training/list_dgs_charged_val"
  val_label_gridsize: "data_lists/training/list_lgs_charged_val"
  train_stats: "data_lists/stats/train_stats.json"
  batch_size: 16
  num_workers: 8

training:
  num_epochs: 100
  learning_rate: 0.0001
  weight_decay: 0.01
  gradient_clip: 1.0
  use_ema: true

evaluation:
  eval_frequency: 5
  num_inference_steps: 50
  ode_method: "heun2"

output:
  output_dir: "output_dir"
  log_dir: "logs"
```

### Inference Config Example

`config/inference_config.yaml`:
```yaml
model:
  name: "unet3d"

data:
  test_data_list: "data_lists/test_sets/list_d_extreme_mofs"
  test_label_list: "data_lists/test_sets/list_l_extreme_mofs"
  test_data_gridsize: "data_lists/test_sets/list_dgs_extreme_mofs"
  test_label_gridsize: "data_lists/test_sets/list_lgs_extreme_mofs"
  train_stats: "data_lists/stats/train_stats_mofs.json"
  batch_size: 8

inference:
  num_steps: 50
  ode_method: "heun2"
  use_ema: true
  save_predictions: true
  calculate_metrics: true

output:
  output_dir: "predictions"
```

---

## 📊 Dataset Management

### Available Datasets (14 total)

**Training (4):**
- `charged` - Main charged systems
- `subMP_12k_charged` - Materials Project (12k)
- `subMP_12k_charged_start_charge` - With start charge
- `defects_charged` - Charged defects

**Test - Extreme (3):** `extreme_mofs`, `extreme_electrene`, `extreme_organic`

**Test - Materials (5):** `mofs`, `electrenes`, `organic`, `organic_molecules`, `perovskites`

**Test - Defects (2):** `special_defects_diamond`, `multisite_defects_diamond`

### Using Dataset Helper

```bash
# List all
python data_lists/list_helper.py --list-all

# Get info
python data_lists/list_helper.py --info extreme_mofs

# Generate config
python data_lists/list_helper.py --config charged --split train
```

### Dataset File Structure

**Each dataset has 4 files (always use together):**
1. `list_d_*` - Data (input) paths
2. `list_l_*` - Label (target) paths
3. `list_dgs_*` - Data grid sizes
4. `list_lgs_*` - Label grid sizes

**Plus stats file:** `train_stats*.json` for normalization

### Config Example

```yaml
data:
  # All 4 list files
  train_data_list: "data_lists/training/list_d_charged_train"
  train_label_list: "data_lists/training/list_l_charged_train"
  train_data_gridsize: "data_lists/training/list_dgs_charged_train"
  train_label_gridsize: "data_lists/training/list_lgs_charged_train"
  # Stats file (match to dataset type)
  train_stats: "data_lists/stats/train_stats.json"
```

**See `data_lists/README.md` for complete documentation.**

---

## 🖥️ HPC/SLURM Job Management

### Method 1: Template-Based (Flexible)

Best for: General training, resource scaling

```bash
# Use preset
python slurm_scripts/submit_job.py \
    --script training/train_multi_gpu.sh \
    --preset production

# Override resources
python slurm_scripts/submit_job.py \
    --script training/train_multi_gpu.sh \
    --nodes 8 --time 48:00:00 --batch-size 32

# Dry run
python slurm_scripts/submit_job.py \
    --script training/train_multi_gpu.sh --preset production --dry-run
```

**Presets:**
- `quick_test` - 1 GPU, 30 min
- `production` - 8 nodes, 96 hours  
- `charged_systems` - 4 GPUs, 48 hours

### Method 2: Named Jobs (Saved Configs)

Best for: Repeated experiments, specific testsets

```bash
# List jobs
python slurm_scripts/submit_named_job.py --list

# Submit
python slurm_scripts/submit_named_job.py --job neutral_extreme_mofs

# Override testset
python slurm_scripts/submit_named_job.py --job neutral_extreme_mofs \
    --testset special_defects_diamond
```

### Setup (One-Time)

Edit `slurm_scripts/config/slurm_defaults.yaml`:

```yaml
notifications:
  email: "your.email@example.com"  # Update!

paths:
  data_root: "/path/to/data"  # Update!

environment:
  conda_env: "flowmatch"  # Update if different

resources:
  multi_gpu:
    partition: "gpu"  # Check your cluster
    gpus_per_node: 4
```

### Add Named Jobs

Edit `slurm_scripts/config/jobs.yaml`:

```yaml
prediction_jobs:
  my_test:
    testset: "extreme_mofs"  # Exact testset name
    checkpoint: "${checkpoints.my_model}"
    resources:
      gpus: 1
      time: "10:00:00"
    script_args:
      batch_size: 1
      ode_method: "heun2"
```

**See `slurm_scripts/README.md` for complete documentation.**

---

## 🔧 Training Details

### Distributed Training

```bash
# Local multi-GPU
torchrun --nproc_per_node=4 scripts/train.py --config config/train_config.yaml

# SLURM single node
sbatch slurm_scripts/training/train_multi_gpu.sh

# SLURM multi-node
python slurm_scripts/submit_job.py --script training/train_multi_node.sh --nodes 8
```

### Available Models

In `src/models/`:
- `unet_3d.py` - Standard 3D U-Net
- `unet_simple_3d.py` - Simplified
- `unet_big.py` - Larger capacity
- `unet_xlarge.py` - Extra large
- `unet_transformer_3d.py` - Transformer-based

### Loss Functions

- `l2` - MSE (default)
- `l1` - MAE
- `hybrid` - Combined

### ODE Solvers

- `heun2` - Recommended
- `midpoint`, `rk45`, `euler`

---

## 📈 Monitoring and Evaluation

### Logs

```bash
# Training progress
tail -f output_dir/training.log

# Metrics
tail -f output_dir/log.txt

# SLURM output
tail -f logs/edp_train_*.out
```

### Checkpoints

In `output_dir/`:
- `checkpoint-best.pth` - Best validation
- `checkpoint-latest.pth` - Most recent
- `checkpoint-epoch-XXX.pth` - Per epoch

### Inference Outputs

In `predictions/`:
- `prediction_*.npy` - Arrays
- `metrics.json` - Metrics
- `inference.log` - Log

### Metrics

MAE, RMSE, R², Relative Error

---

## 🎯 Common Workflows

### 1. Training from Scratch

```bash
# Choose dataset
python data_lists/list_helper.py --list-training

# Update config
vim config/train_config.yaml

# Test locally
python scripts/train.py --config config/train_config.yaml --test-run

# Submit to HPC
python slurm_scripts/submit_job.py --script training/train_multi_gpu.sh --preset production
```

### 2. Test on Multiple Datasets

```bash
# Using named jobs
python slurm_scripts/submit_named_job.py --job test_mofs
python slurm_scripts/submit_named_job.py --job test_electrene
python slurm_scripts/submit_named_job.py --job test_organic

# Or override testset
python slurm_scripts/submit_named_job.py --job my_test --testset extreme_mofs
```

### 3. Hyperparameter Sweep

```bash
for lr in 0.0001 0.0002 0.0005; do
    python slurm_scripts/submit_job.py \
        --script training/train_multi_gpu.sh \
        --learning-rate $lr --output-dir "output_lr_${lr}"
done
```

### 4. Resume Training

```yaml
# In config:
training:
  resume_checkpoint: "output_dir/checkpoint-latest.pth"
```

```bash
python scripts/train.py --config config/train_config.yaml
```

---

## 🐛 Troubleshooting

### Out of Memory
```yaml
data:
  batch_size: 8  # Reduce
training:
  accum_iter: 4  # Accumulate gradients
```

### Slow Loading
```yaml
data:
  num_workers: 16  # Increase
  pin_memory: true
```

### Data Not Found
```bash
# Verify files
ls data_lists/training/list_d_charged_train
head data_lists/training/list_d_charged_train
```

### SLURM Fails
```bash
# Dry run to check
python slurm_scripts/submit_job.py --script SCRIPT --dry-run

# Check logs
tail logs/*.err

# Verify settings
vim slurm_scripts/config/slurm_defaults.yaml
```

---

## 📚 API Reference

### Training Script
```bash
python scripts/train.py \
    --config <path>      # Required
    --test-run           # Quick test
    --resume <path>      # Resume checkpoint
    --output-dir <path>  # Override output
```

### Inference Script
```bash
python scripts/predict.py \
    --config <path>         # Required
    --checkpoint <path>     # Required
    --test-run              # Quick test
    --output-dir <path>     # Override output
    --save-predictions      # Save arrays
    --calculate-metrics     # Calc metrics
```

### Submit Job
```bash
python slurm_scripts/submit_job.py \
    --script <path>         # SLURM script
    --preset <name>         # Preset
    --nodes <int>           # Override nodes
    --time <HH:MM:SS>       # Override time
    --batch-size <int>      # Override batch size
    --dry-run               # Show only
```

### Submit Named Job
```bash
python slurm_scripts/submit_named_job.py \
    --list                  # List jobs
    --job <name>            # Job name
    --testset <name>        # Override testset
    --time <HH:MM:SS>       # Override time
    --dry-run               # Show only
```

---

## ✅ Pre-Deployment Checklist

- [ ] Install: `pip install -e .`
- [ ] Update SLURM config: `slurm_scripts/config/slurm_defaults.yaml`
  - [ ] Email
  - [ ] Data paths
  - [ ] Conda env
  - [ ] Partition
- [ ] Choose dataset
- [ ] Update training config
- [ ] Test locally: `--test-run`
- [ ] SLURM dry run: `--dry-run`
- [ ] Submit training
- [ ] Monitor logs
- [ ] Run inference
- [ ] Evaluate results

---

## 📖 Additional Documentation

- **Data Lists**: `data_lists/README.md` - Complete dataset guide
- **SLURM**: `slurm_scripts/README.md` - HPC job management
- **Source Code**: Check docstrings in `src/` modules

---

## 🎉 Features Summary

✅ **Modular** - Clean architecture
✅ **Config-based** - YAML not CLI
✅ **93 datasets** - Ready to use
✅ **Dual SLURM** - Template + named
✅ **Complete docs** - This + 2 detailed guides
✅ **Production-ready** - Logging, metrics, error handling

**Ready to train!** 🚀
