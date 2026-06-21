# Data Lists Directory

This directory contains organized data list files for training and testing electron density prediction models.

## 📁 Directory Structure

```
data_lists/
├── training/              # Training, validation, and test split lists
│   ├── list_d_*          # Data (input) file paths
│   ├── list_l_*          # Label (target) file paths
│   ├── list_dgs_*        # Data grid size information
│   └── list_lgs_*        # Label grid size information
│
├── test_sets/            # Specialized test set lists
│   ├── Extreme datasets  # extreme_mofs, extreme_electrene, extreme_organic
│   ├── Defect datasets   # special_defects_diamond, multisite_defects_diamond
│   ├── Material types    # mofs, electrenes, organic, organic_molecules, perovskites
│   └── All file types    # list_d_*, list_l_*, list_dgs_*, list_lgs_*
│
├── stats/                # Training statistics for normalization
│   ├── train_stats.json                          # Default training stats
│   ├── train_stats_defects_charged.json          # Charged defects stats
│   ├── train_stats_electrenes.json               # Electrenes stats
│   ├── train_stats_mofs.json                     # MOFs stats
│   ├── train_stats_multisite_defects_diamond.json
│   ├── train_stats_organic.json
│   ├── train_stats_organic_molecules.json
│   └── train_stats_perovskites.json
│
└── args.json             # Original training arguments/configuration
```

## 📋 List File Naming Convention

All list files follow this naming pattern:

```
list_<type>_<dataset>_<split>
```

### File Type Prefixes

| Prefix | Description | Example Content |
|--------|-------------|-----------------|
| `d` | Data (input) file paths | `/path/to/rho_22.npy` |
| `l` | Label (target) file paths | `/path/to/label.npy` |
| `dgs` | Data grid sizes | Grid size for input data |
| `lgs` | Label grid sizes | Grid size for labels |

### Dataset Names

| Dataset | Description |
|---------|-------------|
| `charged` | Charged electron density systems |
| `subMP_12k_charged` | Subset of Materials Project (12k charged systems) |
| `defects_charged` | Defect structures with charged states |
| `extreme_mofs` | Extreme metal-organic frameworks |
| `extreme_electrene` | Extreme electrene materials |
| `extreme_organic` | Extreme organic materials |
| `mofs` | Metal-organic frameworks |
| `electrenes` | Electrene materials |
| `organic` | Organic materials |
| `organic_molecules` | Organic molecules |
| `perovskites` | Perovskite structures |
| `special_defects_diamond` | Special diamond defects |
| `multisite_defects_diamond` | Multi-site diamond defects |

### Data Splits

| Split | Description |
|-------|-------------|
| `_train` | Training set |
| `_val` | Validation set |
| `_test` | Test set |
| (no suffix) | Full dataset (no split) |

## 🔧 Usage in Training

### Example 1: Train on Charged Systems

```python
# In your training script or config
data_config = {
    'train_data_list': 'data_lists/training/list_d_charged_train',
    'train_label_list': 'data_lists/training/list_l_charged_train',
    'train_data_gridsize': 'data_lists/training/list_dgs_charged_train',
    'train_label_gridsize': 'data_lists/training/list_lgs_charged_train',
    
    'val_data_list': 'data_lists/training/list_d_charged_val',
    'val_label_list': 'data_lists/training/list_l_charged_val',
    'val_data_gridsize': 'data_lists/training/list_dgs_charged_val',
    'val_label_gridsize': 'data_lists/training/list_lgs_charged_val',
    
    'train_stats': 'data_lists/stats/train_stats.json'
}
```

### Example 2: Train on SubMP 12k Charged Dataset

```python
data_config = {
    'train_data_list': 'data_lists/training/list_d_subMP_12k_charged_train',
    'train_label_list': 'data_lists/training/list_l_subMP_12k_charged_train',
    'train_data_gridsize': 'data_lists/training/list_dgs_subMP_12k_charged_train',
    'train_label_gridsize': 'data_lists/training/list_lgs_subMP_12k_charged_train',
    
    'train_stats': 'data_lists/stats/train_stats.json'
}
```

## 🎯 Usage in Testing/Inference

### Example 1: Test on Extreme MOFs

```python
test_config = {
    'test_data_list': 'data_lists/test_sets/list_d_extreme_mofs',
    'test_label_list': 'data_lists/test_sets/list_l_extreme_mofs',
    'test_data_gridsize': 'data_lists/test_sets/list_dgs_extreme_mofs',
    'test_label_gridsize': 'data_lists/test_sets/list_lgs_extreme_mofs',
    
    'train_stats': 'data_lists/stats/train_stats_mofs.json'  # Use matching stats
}
```

### Example 2: Test on Special Diamond Defects

```python
test_config = {
    'test_data_list': 'data_lists/test_sets/list_d_special_defects_diamond',
    'test_label_list': 'data_lists/test_sets/list_l_special_defects_diamond',
    'test_data_gridsize': 'data_lists/test_sets/list_dgs_special_defects_diamond',
    'test_label_gridsize': 'data_lists/test_sets/list_lgs_special_defects_diamond',
    
    'train_stats': 'data_lists/stats/train_stats.json'
}
```

## 📊 Training Statistics Files

Training statistics are used for data normalization. Match the stats file to your dataset type:

| Stats File | Use With |
|------------|----------|
| `train_stats.json` | General/default, charged systems |
| `train_stats_defects_charged.json` | Charged defect structures |
| `train_stats_mofs.json` | Metal-organic frameworks |
| `train_stats_electrenes.json` | Electrene materials |
| `train_stats_organic.json` | Organic materials |
| `train_stats_organic_molecules.json` | Organic molecules |
| `train_stats_perovskites.json` | Perovskite structures |
| `train_stats_multisite_defects_diamond.json` | Diamond defects |

## 🗂️ File Format

### List Files

Each list file contains one file path per line:

```
/path/to/data/rho_22.npy
/path/to/data/rho_23.npy
/path/to/data/rho_24.npy
...
```

### Statistics Files

JSON format with mean/std for normalization:

```json
{
    "mean": 0.12345,
    "std": 0.67890,
    "min": 0.0,
    "max": 10.5
}
```

## 🔍 Finding the Right Lists

### For Training

**Standard Training:**
- Use `list_*_charged_train` + `list_*_charged_val`

**Large Scale Training:**
- Use `list_*_subMP_12k_charged_train`

**Defect-Specific Training:**
- Use `list_*_defects_charged_train`

### For Testing

**Generalization Testing:**
- `extreme_mofs` - Challenging MOF structures
- `extreme_electrene` - Challenging electrene structures  
- `extreme_organic` - Challenging organic materials

**Domain-Specific Testing:**
- `mofs` - Standard MOF test set
- `electrenes` - Standard electrene test set
- `organic` / `organic_molecules` - Organic materials
- `perovskites` - Perovskite structures

**Defect Testing:**
- `special_defects_diamond` - Special diamond defect structures
- `multisite_defects_diamond` - Multi-site defects in diamond

## 💡 Quick Reference Commands

### Count samples in a list file
```bash
wc -l data_lists/training/list_d_charged_train
```

### View first few paths
```bash
head data_lists/training/list_d_charged_train
```

### Check all available datasets
```bash
ls -1 data_lists/training/list_d_* | sed 's/^.*list_d_//' | sort -u
```

### Check all available test sets
```bash
ls -1 data_lists/test_sets/list_d_* | sed 's/^.*list_d_//' | sort -u
```

### Verify file pairs exist
```bash
# For training
ls data_lists/training/list_d_charged_train
ls data_lists/training/list_l_charged_train
ls data_lists/training/list_dgs_charged_train
ls data_lists/training/list_lgs_charged_train
```

## 📖 Integration with Scripts

### Update config/train_config.yaml

```yaml
data:
  # Training data
  train_data_list: "data_lists/training/list_d_charged_train"
  train_label_list: "data_lists/training/list_l_charged_train"
  train_data_gridsize: "data_lists/training/list_dgs_charged_train"
  train_label_gridsize: "data_lists/training/list_lgs_charged_train"
  
  # Validation data
  val_data_list: "data_lists/training/list_d_charged_val"
  val_label_list: "data_lists/training/list_l_charged_val"
  val_data_gridsize: "data_lists/training/list_dgs_charged_val"
  val_label_gridsize: "data_lists/training/list_lgs_charged_val"
  
  # Statistics
  train_stats: "data_lists/stats/train_stats.json"
```

### Update config/inference_config.yaml

```yaml
data:
  test_data_list: "data_lists/test_sets/list_d_extreme_mofs"
  test_label_list: "data_lists/test_sets/list_l_extreme_mofs"
  test_data_gridsize: "data_lists/test_sets/list_dgs_extreme_mofs"
  test_label_gridsize: "data_lists/test_sets/list_lgs_extreme_mofs"
  train_stats: "data_lists/stats/train_stats_mofs.json"
```

## ⚠️ Important Notes

1. **Always use matching prefixes**: If you use `list_d_*` for data, use `list_l_*` for labels with the same suffix
2. **Grid sizes matter**: Include `list_dgs_*` and `list_lgs_*` for proper data handling
3. **Stats files**: Use the appropriate stats file for your dataset type for proper normalization
4. **File paths**: All paths in list files are absolute - no need to modify them
5. **Consistency**: Keep train/val/test splits consistent across all four file types (d, l, dgs, lgs)

## 🚀 Complete Training Example

```yaml
# config/train_config.yaml
data:
  # Training data
  train_data_list: "data_lists/training/list_d_charged_train"
  train_label_list: "data_lists/training/list_l_charged_train"
  train_data_gridsize: "data_lists/training/list_dgs_charged_train"
  train_label_gridsize: "data_lists/training/list_lgs_charged_train"
  
  # Validation data
  val_data_list: "data_lists/training/list_d_charged_val"
  val_label_list: "data_lists/training/list_l_charged_val"
  val_data_gridsize: "data_lists/training/list_dgs_charged_val"
  val_label_gridsize: "data_lists/training/list_lgs_charged_val"
  
  # Normalization
  train_stats: "data_lists/stats/train_stats.json"
  
  # Data loading
  batch_size: 16
  num_workers: 8
  pin_memory: true
```

Then train:
```bash
python scripts/train.py --config config/train_config.yaml
```

## 📝 Summary

- **89 total files** organized into logical categories
- **Training lists**: Dedicated train/val/test splits for training
- **Test sets**: Specialized evaluation datasets for different material types
- **Statistics**: Pre-computed normalization statistics
- **Well-documented**: Clear naming convention and usage examples

All lists are ready to use in your training and prediction scripts!
