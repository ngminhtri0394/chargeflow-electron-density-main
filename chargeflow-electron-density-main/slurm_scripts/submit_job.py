#!/usr/bin/env python3
"""
SLURM Job Submission Helper

This script reads configuration files and submits SLURM jobs with the specified settings,
eliminating the need to type all parameters manually.

Usage:
    # Use defaults
    python submit_job.py --script training/train_multi_gpu.sh
    
    # Use a preset
    python submit_job.py --script training/train_multi_gpu.sh --preset quick_test
    
    # Override specific values
    python submit_job.py --script training/train_multi_gpu.sh --preset production --nodes 16
    
    # Use custom config
    python submit_job.py --script training/train_multi_gpu.sh --config my_config.yaml
"""

import argparse
import os
import subprocess
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import re


class SlurmJobSubmitter:
    """Helper class to submit SLURM jobs with configuration files."""
    
    def __init__(self, script_dir: Path):
        self.script_dir = script_dir
        self.config_dir = script_dir / "config"
        self.defaults = self.load_defaults()
    
    def load_defaults(self) -> Dict[str, Any]:
        """Load default configuration."""
        defaults_file = self.config_dir / "slurm_defaults.yaml"
        if defaults_file.exists():
            with open(defaults_file) as f:
                return yaml.safe_load(f)
        return {}
    
    def load_preset(self, preset_name: str) -> Dict[str, Any]:
        """Load a preset configuration."""
        preset_file = self.config_dir / "presets" / f"{preset_name}.yaml"
        if not preset_file.exists():
            raise ValueError(f"Preset not found: {preset_name}")
        
        with open(preset_file) as f:
            return yaml.safe_load(f)
    
    def load_custom_config(self, config_path: str) -> Dict[str, Any]:
        """Load a custom configuration file."""
        config_file = Path(config_path)
        if not config_file.exists():
            raise ValueError(f"Config file not found: {config_path}")
        
        with open(config_file) as f:
            return yaml.safe_load(f)
    
    def merge_configs(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge multiple configuration dictionaries."""
        result = {}
        for config in configs:
            self._deep_merge(result, config)
        return result
    
    def _deep_merge(self, base: Dict, update: Dict) -> None:
        """Recursively merge update into base."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def expand_variables(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Expand ${variable} references in config."""
        # First pass: expand code_dir and project_root
        if 'paths' in config:
            if 'code_dir' in config['paths']:
                code_dir = config['paths']['code_dir']
            if 'project_root' in config['paths']:
                project_root = config['paths']['project_root']
        
        # Second pass: expand all other variables
        return self._expand_dict(config, config)
    
    def _expand_dict(self, obj: Any, context: Dict) -> Any:
        """Recursively expand variables in dictionary."""
        if isinstance(obj, dict):
            return {k: self._expand_dict(v, context) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._expand_dict(item, context) for item in obj]
        elif isinstance(obj, str):
            return self._expand_string(obj, context)
        return obj
    
    def _expand_string(self, s: str, context: Dict) -> str:
        """Expand ${variable} references in a string."""
        pattern = r'\$\{([^}]+)\}'
        
        def replace(match):
            var_path = match.group(1).split('.')
            value = context
            for key in var_path:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return match.group(0)  # Keep original if not found
            return str(value)
        
        return re.sub(pattern, replace, s)
    
    def config_to_env_vars(self, config: Dict[str, Any], overrides: Dict[str, str]) -> Dict[str, str]:
        """Convert config to environment variables for SLURM script."""
        env_vars = {}
        
        # Resources
        if 'resources' in config:
            for job_type, resources in config['resources'].items():
                for key, value in resources.items():
                    env_key = f"SLURM_{key.upper()}"
                    env_vars[env_key] = str(value)
        
        # Environment
        if 'environment' in config:
            if 'conda_env' in config['environment']:
                env_vars['CONDA_ENV'] = config['environment']['conda_env']
            if 'module_loads' in config['environment']:
                env_vars['MODULE_LOADS'] = ' '.join(config['environment']['module_loads'])
        
        # Paths
        if 'paths' in config:
            for key, value in config['paths'].items():
                env_key = key.upper()
                env_vars[env_key] = str(value)
        
        # Training
        if 'training' in config:
            for key, value in config['training'].items():
                env_key = f"TRAIN_{key.upper()}"
                env_vars[env_key] = str(value)
        
        # Inference
        if 'inference' in config:
            for key, value in config['inference'].items():
                env_key = f"INFER_{key.upper()}"
                env_vars[env_key] = str(value)
        
        # Notifications
        if 'notifications' in config:
            if 'email' in config['notifications']:
                env_vars['SLURM_EMAIL'] = config['notifications']['email']
        
        # Apply overrides
        env_vars.update(overrides)
        
        return env_vars
    
    def submit_job(self, script_path: Path, env_vars: Dict[str, str], dry_run: bool = False) -> None:
        """Submit the SLURM job."""
        if not script_path.exists():
            raise ValueError(f"Script not found: {script_path}")
        
        # Build sbatch command
        cmd = ['sbatch']
        
        # Add environment variables as sbatch export
        if env_vars:
            export_str = ','.join(f"{k}={v}" for k, v in env_vars.items())
            cmd.extend(['--export', f"ALL,{export_str}"])
        
        cmd.append(str(script_path))
        
        if dry_run:
            print("Dry run - would execute:")
            print(f"Command: {' '.join(cmd)}")
            print("\nEnvironment variables:")
            for key, value in sorted(env_vars.items()):
                print(f"  {key}={value}")
            return
        
        # Submit job
        print(f"Submitting job: {script_path.name}")
        print(f"With {len(env_vars)} environment variables")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✓ Job submitted successfully")
            print(result.stdout)
        else:
            print(f"✗ Job submission failed")
            print(result.stderr)
            raise RuntimeError(f"sbatch failed with return code {result.returncode}")


def main():
    parser = argparse.ArgumentParser(
        description="Submit SLURM jobs with configuration files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use defaults
  python submit_job.py --script training/train_multi_gpu.sh
  
  # Use a preset
  python submit_job.py --script training/train_multi_gpu.sh --preset quick_test
  
  # Override specific values
  python submit_job.py --script training/train_multi_gpu.sh --nodes 8 --time 48:00:00
  
  # Use custom config
  python submit_job.py --script training/train_multi_gpu.sh --config my_config.yaml
  
  # Dry run to see what would be executed
  python submit_job.py --script training/train_multi_gpu.sh --preset production --dry-run
        """
    )
    
    parser.add_argument('--script', required=True, help='SLURM script to submit (relative to slurm_scripts/)')
    parser.add_argument('--preset', help='Preset configuration to use')
    parser.add_argument('--config', help='Custom configuration file')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be executed without submitting')
    
    # Resource overrides
    parser.add_argument('--partition', help='Override partition')
    parser.add_argument('--nodes', type=int, help='Override number of nodes')
    parser.add_argument('--gpus-per-node', type=int, help='Override GPUs per node')
    parser.add_argument('--cpus-per-task', type=int, help='Override CPUs per task')
    parser.add_argument('--memory', help='Override memory (e.g., 64G)')
    parser.add_argument('--time', help='Override time limit (e.g., 24:00:00)')
    
    # Training overrides
    parser.add_argument('--batch-size', type=int, help='Override batch size')
    parser.add_argument('--learning-rate', type=float, help='Override learning rate')
    parser.add_argument('--num-epochs', type=int, help='Override number of epochs')
    
    # Path overrides
    parser.add_argument('--output-dir', help='Override output directory')
    parser.add_argument('--checkpoint', help='Checkpoint path for inference')
    
    args = parser.parse_args()
    
    # Initialize submitter
    script_dir = Path(__file__).parent
    submitter = SlurmJobSubmitter(script_dir)
    
    # Load configurations
    configs_to_merge = [submitter.defaults]
    
    if args.preset:
        print(f"Loading preset: {args.preset}")
        configs_to_merge.append(submitter.load_preset(args.preset))
    
    if args.config:
        print(f"Loading custom config: {args.config}")
        configs_to_merge.append(submitter.load_custom_config(args.config))
    
    # Merge all configs
    merged_config = submitter.merge_configs(*configs_to_merge)
    merged_config = submitter.expand_variables(merged_config)
    
    # Build overrides from command line
    overrides = {}
    if args.partition:
        overrides['SLURM_PARTITION'] = args.partition
    if args.nodes:
        overrides['SLURM_NODES'] = str(args.nodes)
    if args.gpus_per_node:
        overrides['SLURM_GPUS_PER_NODE'] = str(args.gpus_per_node)
    if args.cpus_per_task:
        overrides['SLURM_CPUS_PER_TASK'] = str(args.cpus_per_task)
    if args.memory:
        overrides['SLURM_MEMORY'] = args.memory
    if args.time:
        overrides['SLURM_TIME'] = args.time
    if args.batch_size:
        overrides['TRAIN_BATCH_SIZE'] = str(args.batch_size)
    if args.learning_rate:
        overrides['TRAIN_LEARNING_RATE'] = str(args.learning_rate)
    if args.num_epochs:
        overrides['TRAIN_NUM_EPOCHS'] = str(args.num_epochs)
    if args.output_dir:
        overrides['OUTPUT_DIR'] = args.output_dir
    if args.checkpoint:
        overrides['CHECKPOINT_PATH'] = args.checkpoint
    
    # Convert config to environment variables
    env_vars = submitter.config_to_env_vars(merged_config, overrides)
    
    # Submit job
    script_path = script_dir / args.script
    submitter.submit_job(script_path, env_vars, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
