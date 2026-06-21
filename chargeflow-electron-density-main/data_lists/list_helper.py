#!/usr/bin/env python3
"""
Dataset List Helper

This script helps you explore and use the organized data lists.
"""

import yaml
from pathlib import Path
import argparse


def load_datasets_config():
    """Load datasets configuration."""
    config_file = Path(__file__).parent / "datasets.yaml"
    with open(config_file) as f:
        return yaml.safe_load(f)


def list_training_datasets():
    """List all available training datasets."""
    config = load_datasets_config()
    
    print("=" * 70)
    print("TRAINING DATASETS")
    print("=" * 70)
    print()
    
    for key, dataset in config['training_datasets'].items():
        print(f"📦 {key}")
        print(f"   Name: {dataset['name']}")
        print(f"   Train: {dataset['data_list']}")
        if 'val_data_list' in dataset:
            print(f"   Val:   {dataset['val_data_list']}")
        if 'test_data_list' in dataset:
            print(f"   Test:  {dataset['test_data_list']}")
        print(f"   Stats: {dataset['train_stats']}")
        print()


def list_test_sets():
    """List all available test sets."""
    config = load_datasets_config()
    
    print("=" * 70)
    print("TEST SETS - EXTREME (Generalization Testing)")
    print("=" * 70)
    print()
    
    for key, dataset in config['test_sets_extreme'].items():
        print(f"🔬 {key}")
        print(f"   Name: {dataset['name']}")
        print(f"   Description: {dataset['description']}")
        print(f"   Data: {dataset['data_list']}")
        print(f"   Stats: {dataset['train_stats']}")
        print()
    
    print("=" * 70)
    print("TEST SETS - MATERIALS (Domain-Specific)")
    print("=" * 70)
    print()
    
    for key, dataset in config['test_sets_materials'].items():
        print(f"🧪 {key}")
        print(f"   Name: {dataset['name']}")
        print(f"   Description: {dataset['description']}")
        print(f"   Data: {dataset['data_list']}")
        print(f"   Stats: {dataset['train_stats']}")
        print()
    
    print("=" * 70)
    print("TEST SETS - DEFECTS (Defect-Specific)")
    print("=" * 70)
    print()
    
    for key, dataset in config['test_sets_defects'].items():
        print(f"💎 {key}")
        print(f"   Name: {dataset['name']}")
        print(f"   Description: {dataset['description']}")
        print(f"   Data: {dataset['data_list']}")
        print(f"   Stats: {dataset['train_stats']}")
        print()


def get_dataset_paths(dataset_name, split='train'):
    """Get file paths for a specific dataset."""
    config = load_datasets_config()
    
    # Search in training datasets
    if dataset_name in config['training_datasets']:
        dataset = config['training_datasets'][dataset_name]
        
        if split == 'train':
            return {
                'data_list': dataset['data_list'],
                'label_list': dataset['label_list'],
                'data_gridsize': dataset['data_gridsize'],
                'label_gridsize': dataset['label_gridsize'],
                'train_stats': dataset['train_stats']
            }
        elif split == 'val' and 'val_data_list' in dataset:
            return {
                'data_list': dataset['val_data_list'],
                'label_list': dataset['val_label_list'],
                'data_gridsize': dataset['val_data_gridsize'],
                'label_gridsize': dataset['val_label_gridsize'],
                'train_stats': dataset['train_stats']
            }
        elif split == 'test' and 'test_data_list' in dataset:
            return {
                'data_list': dataset['test_data_list'],
                'label_list': dataset['test_label_list'],
                'data_gridsize': dataset['test_data_gridsize'],
                'label_gridsize': dataset['test_label_gridsize'],
                'train_stats': dataset['train_stats']
            }
    
    # Search in test sets
    for category in ['test_sets_extreme', 'test_sets_materials', 'test_sets_defects']:
        if dataset_name in config[category]:
            dataset = config[category][dataset_name]
            return {
                'data_list': dataset['data_list'],
                'label_list': dataset['label_list'],
                'data_gridsize': dataset['data_gridsize'],
                'label_gridsize': dataset['label_gridsize'],
                'train_stats': dataset['train_stats']
            }
    
    raise ValueError(f"Dataset not found: {dataset_name}")


def print_dataset_info(dataset_name):
    """Print detailed information about a dataset."""
    config = load_datasets_config()
    
    # Search in all categories
    all_datasets = {
        **config['training_datasets'],
        **config['test_sets_extreme'],
        **config['test_sets_materials'],
        **config['test_sets_defects']
    }
    
    if dataset_name not in all_datasets:
        print(f"❌ Dataset not found: {dataset_name}")
        print("\nAvailable datasets:")
        for key in all_datasets.keys():
            print(f"  - {key}")
        return
    
    dataset = all_datasets[dataset_name]
    
    print("=" * 70)
    print(f"Dataset: {dataset_name}")
    print("=" * 70)
    print()
    print(f"Name: {dataset['name']}")
    if 'description' in dataset:
        print(f"Description: {dataset['description']}")
    print()
    print("File Paths:")
    print(f"  Data:         {dataset['data_list']}")
    print(f"  Labels:       {dataset['label_list']}")
    print(f"  Data Grid:    {dataset['data_gridsize']}")
    print(f"  Label Grid:   {dataset['label_gridsize']}")
    print(f"  Train Stats:  {dataset['train_stats']}")
    
    if 'val_data_list' in dataset:
        print()
        print("Validation Split:")
        print(f"  Data:         {dataset['val_data_list']}")
        print(f"  Labels:       {dataset['val_label_list']}")
        print(f"  Data Grid:    {dataset['val_data_gridsize']}")
        print(f"  Label Grid:   {dataset['val_label_gridsize']}")
    
    if 'test_data_list' in dataset:
        print()
        print("Test Split:")
        print(f"  Data:         {dataset['test_data_list']}")
        print(f"  Labels:       {dataset['test_label_list']}")
        print(f"  Data Grid:    {dataset['test_data_gridsize']}")
        print(f"  Label Grid:   {dataset['test_label_gridsize']}")
    
    print()
    print("=" * 70)


def generate_config_snippet(dataset_name, split='train'):
    """Generate config file snippet for a dataset."""
    try:
        paths = get_dataset_paths(dataset_name, split)
        
        print("=" * 70)
        print(f"Config Snippet for: {dataset_name} ({split})")
        print("=" * 70)
        print()
        print("# Add this to your config YAML file:")
        print()
        print("data:")
        print(f"  data_list: \"{paths['data_list']}\"")
        print(f"  label_list: \"{paths['label_list']}\"")
        print(f"  data_gridsize: \"{paths['data_gridsize']}\"")
        print(f"  label_gridsize: \"{paths['label_gridsize']}\"")
        print(f"  train_stats: \"{paths['train_stats']}\"")
        print()
        print("=" * 70)
        
    except ValueError as e:
        print(f"❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Dataset List Helper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all training datasets
  python list_helper.py --list-training
  
  # List all test sets
  python list_helper.py --list-test
  
  # Get info about a specific dataset
  python list_helper.py --info extreme_mofs
  
  # Generate config snippet
  python list_helper.py --config charged --split train
        """
    )
    
    parser.add_argument('--list-training', action='store_true',
                       help='List all training datasets')
    parser.add_argument('--list-test', action='store_true',
                       help='List all test sets')
    parser.add_argument('--list-all', action='store_true',
                       help='List everything')
    parser.add_argument('--info', metavar='DATASET',
                       help='Show detailed info about a dataset')
    parser.add_argument('--config', metavar='DATASET',
                       help='Generate config snippet for a dataset')
    parser.add_argument('--split', default='train',
                       choices=['train', 'val', 'test'],
                       help='Split to use (default: train)')
    
    args = parser.parse_args()
    
    if args.list_all or args.list_training:
        list_training_datasets()
    
    if args.list_all or args.list_test:
        list_test_sets()
    
    if args.info:
        print_dataset_info(args.info)
    
    if args.config:
        generate_config_snippet(args.config, args.split)
    
    if not any([args.list_training, args.list_test, args.list_all, 
                args.info, args.config]):
        parser.print_help()


if __name__ == '__main__':
    main()
