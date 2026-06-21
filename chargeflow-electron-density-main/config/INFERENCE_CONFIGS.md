# Inference Configuration Files Reference

This directory contains ready-to-use inference configuration files for all test datasets.

## Available Inference Configs

### Extreme Test Sets (Generalization Testing)

| Config File | Dataset | Description |
|------------|---------|-------------|
| `inference_extreme_mofs.yaml` | extreme_mofs | Challenging MOF structures |
| `inference_extreme_electrene.yaml` | extreme_electrene | Challenging electrene materials |
| `inference_extreme_organic.yaml` | extreme_organic | Challenging organic materials |

### Standard Test Sets (Domain-Specific)

| Config File | Dataset | Description |
|------------|---------|-------------|
| `inference_mofs.yaml` | mofs | Standard MOF test set |
| `inference_electrenes.yaml` | electrenes | Standard electrene test set |
| `inference_organic.yaml` | organic | Standard organic materials |
| `inference_organic_molecules.yaml` | organic_molecules | Organic molecules |
| `inference_perovskites.yaml` | perovskites | Perovskite structures |

### Defect Test Sets

| Config File | Dataset | Description |
|------------|---------|-------------|
| `inference_special_defects_diamond.yaml` | special_defects_diamond | Special diamond defects |
| `inference_multisite_defects_diamond.yaml` | multisite_defects_diamond | Multi-site diamond defects |

## Usage

### Basic Inference

```bash
# Run inference on extreme MOFs
python scripts/predict.py \
    --config config/inference_extreme_mofs.yaml \
    --checkpoint output_dir/checkpoint-best.pth

# Run inference on perovskites
python scripts/predict.py \
    --config config/inference_perovskites.yaml \
    --checkpoint output_dir/checkpoint-best.pth
```

### Batch Inference on All Test Sets

```bash
# Extreme test sets
for testset in extreme_mofs extreme_electrene extreme_organic; do
    python scripts/predict.py \
        --config config/inference_${testset}.yaml \
        --checkpoint output_dir/checkpoint-best.pth
done

# Standard test sets
for testset in mofs electrenes organic organic_molecules perovskites; do
    python scripts/predict.py \
        --config config/inference_${testset}.yaml \
        --checkpoint output_dir/checkpoint-best.pth
done

# Defect test sets
for testset in special_defects_diamond multisite_defects_diamond; do
    python scripts/predict.py \
        --config config/inference_${testset}.yaml \
        --checkpoint output_dir/checkpoint-best.pth
done
```

### Using SLURM

```bash
# Submit single test
python slurm_scripts/submit_job.py \
    --script inference/predict_single_gpu.sh \
    --config ../config/inference_extreme_mofs.yaml \
    --checkpoint output_dir/checkpoint-best.pth

# Or use named jobs (see slurm_scripts/config/jobs.yaml)
python slurm_scripts/submit_named_job.py --job test_extreme_mofs
```

## Configuration Structure

Each inference config contains:

```yaml
model:
  name: "unet3d"              # Model architecture
  in_channels: 1
  out_channels: 1

data:
  test_data_list: "..."       # Test data paths (4 files)
  test_label_list: "..."
  test_data_gridsize: "..."
  test_label_gridsize: "..."
  train_stats: "..."          # Stats for normalization
  batch_size: 1               # Batch size
  num_workers: 4

inference:
  num_steps: 50               # ODE solver steps
  ode_method: "heun2"         # Solver method
  use_ema: true               # Use EMA weights
  save_predictions: true      # Save predictions
  calculate_metrics: true     # Calculate metrics

output:
  output_dir: "predictions/..." # Output directory
  log_file: "..."             # Log file path
```

## Customization

### Adjust Batch Size

For faster inference (if you have enough memory):

```yaml
data:
  batch_size: 4  # Instead of 1
```

### Change ODE Solver

For different speed/accuracy tradeoffs:

```yaml
inference:
  num_steps: 100      # More steps = higher quality
  ode_method: "rk45"  # Different solver
```

### Disable Metrics

If you only want predictions without metrics:

```yaml
inference:
  calculate_metrics: false
```

## Output Locations

Each config saves outputs to a dedicated directory:

```
predictions/
├── extreme_mofs/
├── extreme_electrene/
├── extreme_organic/
├── mofs/
├── electrenes/
├── organic/
├── organic_molecules/
├── perovskites/
├── special_defects_diamond/
└── multisite_defects_diamond/
```

Each directory contains:
- `prediction_*.npy` - Predicted density arrays
- `metrics.json` - Performance metrics (MAE, RMSE, R²)
- `inference.log` - Inference log

## See Also

- `train_config.yaml` - Training configuration
- `train_charged_config.yaml` - Charged systems training
- `../data_lists/README.md` - Dataset documentation
- `../README.md` - Main project documentation
