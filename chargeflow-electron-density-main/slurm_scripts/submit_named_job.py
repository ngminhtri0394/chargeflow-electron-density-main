#!/usr/bin/env python3
"""
SLURM Job Submission with Named Configurations

This script lets you submit jobs using predefined configurations from jobs.yaml,
so you don't have to remember all the specific values like checkpoint paths,
test set names, and parameters.

Usage:
    # List available job configurations
    python submit_named_job.py --list
    
    # Submit a specific job configuration
    python submit_named_job.py --job neutral_special_defects_diamond
    
    # Submit with dry run to see what would execute
    python submit_named_job.py --job neutral_extreme_mofs --dry-run
    
    # Override specific values
    python submit_named_job.py --job neutral_extreme_mofs --time 5:00:00 --batch-size 2
"""

import argparse
import os
import subprocess
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import re


class NamedJobSubmitter:
    """Submit SLURM jobs using named configurations from jobs.yaml."""
    
    def __init__(self, script_dir: Path):
        self.script_dir = script_dir
        self.config_dir = script_dir / "config"
        self.jobs_config = self.load_jobs_config()
    
    def load_jobs_config(self) -> Dict[str, Any]:
        """Load jobs configuration."""
        jobs_file = self.config_dir / "jobs.yaml"
        if not jobs_file.exists():
            raise ValueError(f"Jobs config not found: {jobs_file}")
        
        with open(jobs_file) as f:
            config = yaml.safe_load(f)
        
        # Expand variables
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
    
    def list_jobs(self) -> None:
        """List all available job configurations."""
        print("Available Job Configurations:")
        print("=" * 70)
        
        if 'prediction_jobs' in self.jobs_config:
            print("\n📊 Prediction Jobs:")
            print("-" * 70)
            for job_name, config in self.jobs_config['prediction_jobs'].items():
                print(f"\n  {job_name}")
                print(f"    Job Name: {config.get('job_name', 'N/A')}")
                print(f"    Test Set: {config.get('testset', 'N/A')}")
                print(f"    Dataset:  {config.get('dataset', 'N/A')}")
                if 'resources' in config:
                    res = config['resources']
                    print(f"    Resources: {res.get('gpus', 1)} GPU(s), {res.get('time', 'N/A')}")
        
        if 'training_jobs' in self.jobs_config:
            print("\n🔬 Training Jobs:")
            print("-" * 70)
            for job_name, config in self.jobs_config['training_jobs'].items():
                print(f"\n  {job_name}")
                print(f"    Job Name: {config.get('job_name', 'N/A')}")
                print(f"    Dataset:  {config.get('dataset', 'N/A')}")
                if 'resources' in config:
                    res = config['resources']
                    print(f"    Resources: {res.get('nodes', 1)} node(s), {res.get('time', 'N/A')}")
        
        print("\n" + "=" * 70)
        print("\nAvailable Test Sets:")
        if 'testsets' in self.jobs_config:
            for testset_name, testset_config in self.jobs_config['testsets'].items():
                print(f"  - {testset_name}: {testset_config.get('description', '')}")
        
        print("\n" + "=" * 70)
        print("\nAvailable Checkpoints:")
        if 'checkpoints' in self.jobs_config:
            for ckpt_name, ckpt_path in self.jobs_config['checkpoints'].items():
                # Truncate long paths for display
                display_path = ckpt_path if len(ckpt_path) < 60 else "..." + ckpt_path[-57:]
                print(f"  - {ckpt_name}:")
                print(f"      {display_path}")
    
    def get_job_config(self, job_name: str, job_type: str = None) -> Dict[str, Any]:
        """Get configuration for a specific job."""
        # Try prediction jobs first
        if job_type is None or job_type == 'prediction':
            if 'prediction_jobs' in self.jobs_config:
                if job_name in self.jobs_config['prediction_jobs']:
                    return self.jobs_config['prediction_jobs'][job_name]
        
        # Try training jobs
        if job_type is None or job_type == 'training':
            if 'training_jobs' in self.jobs_config:
                if job_name in self.jobs_config['training_jobs']:
                    return self.jobs_config['training_jobs'][job_name]
        
        raise ValueError(f"Job configuration not found: {job_name}")
    
    def build_sbatch_script(self, job_config: Dict[str, Any], overrides: Dict[str, Any]) -> str:
        """Build SBATCH script content from job configuration."""
        common = self.jobs_config.get('common', {})
        
        # Get resources
        resources = job_config.get('resources', {})
        partition = overrides.get('partition', common.get('partition', 'gpu'))
        nodes = overrides.get('nodes', resources.get('nodes', 1))
        gpus = overrides.get('gpus', resources.get('gpus', 1))
        cpus = overrides.get('cpus_per_task', resources.get('cpus_per_task', 5))
        time_limit = overrides.get('time', resources.get('time', '1:00:00'))
        qos = overrides.get('qos', common.get('qos', 'batch-short'))
        gpu_type = overrides.get('gpu_type', common.get('gpu_type', 'v100'))
        
        # Build script
        job_name = job_config.get('job_name', 'FlowEDP_job')
        log_dir = common.get('log_dir', 'logs')
        
        script_lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --output={log_dir}/{job_name}.out",
            f"#SBATCH --error={log_dir}/{job_name}.err",
            f"#SBATCH --partition={partition}",
            f"#SBATCH --nodes={nodes}",
            f"#SBATCH --gres=gpu:{gpu_type}:{gpus}",
            f"#SBATCH --cpus-per-task={cpus}",
            f"#SBATCH --ntasks-per-node=1",
            f"#SBATCH --time={time_limit}",
            f"#SBATCH --qos={qos}",
            "",
            "# Set up environment",
            'eval "$(conda shell.bash hook)"',
        ]
        
        # Add module loads
        for module in common.get('modules', []):
            script_lines.append(f"module load {module}")
        
        # Activate conda
        conda_env = common.get('conda_env', 'flow_matching')
        script_lines.extend([
            "source activate",
            f"conda activate {conda_env}",
            "",
            "export PYTHONNOUSERSITE=True",
            "",
        ])
        
        # Build Python command
        project_root = common.get('project_root', '/home/minhtrin/Charged_electron_density/flow_matching')
        
        # Determine script type and path
        if 'script_args' in job_config:
            # Prediction job
            script_path = f"{project_root}/examples/image/predict_charge.py"
            script_args = job_config['script_args']
            
            cmd = f"python {script_path}"
            
            # Add checkpoint
            checkpoint = overrides.get('checkpoint', job_config.get('checkpoint'))
            if checkpoint:
                cmd += f" --resume {checkpoint}"
            
            # Add standard args
            if script_args.get('start_sad'):
                cmd += " --start_sad"
            if script_args.get('eval_only'):
                cmd += " --eval_only"
            if script_args.get('use_ema'):
                cmd += " --use_ema"
            
            # Add configurable args
            batch_size = overrides.get('batch_size', script_args.get('batch_size', 1))
            cmd += f" --batch_size {batch_size}"
            
            dataset = overrides.get('dataset', job_config.get('dataset'))
            if dataset:
                cmd += f" --dataset {dataset}"
            
            testset = overrides.get('testset', job_config.get('testset'))
            if testset:
                cmd += f" --testset {testset}"
            
            ode_method = overrides.get('ode_method', script_args.get('ode_method'))
            if ode_method:
                cmd += f" --ode_method {ode_method}"
            
            ode_options = overrides.get('ode_options', script_args.get('ode_options'))
            if ode_options:
                cmd += f" --ode_options '{ode_options}'"
        
        elif 'training_args' in job_config:
            # Training job - similar pattern
            script_path = f"{project_root}/final_code/scripts/train.py"
            cmd = f"python {script_path}"
            # Add training args...
        else:
            cmd = "echo 'No script configured'"
        
        script_lines.append(cmd)
        
        return "\n".join(script_lines)
    
    def submit_job(self, job_name: str, overrides: Dict[str, Any], dry_run: bool = False) -> None:
        """Submit a job by name."""
        # Get job configuration
        job_config = self.get_job_config(job_name)
        
        # Build script
        script_content = self.build_sbatch_script(job_config, overrides)
        
        if dry_run:
            print("=" * 70)
            print(f"DRY RUN - Job: {job_name}")
            print("=" * 70)
            print("\nSBATCH script that would be submitted:")
            print("-" * 70)
            print(script_content)
            print("-" * 70)
            return
        
        # Write to temporary file
        temp_script = self.script_dir / f"_temp_{job_name}.sh"
        with open(temp_script, 'w') as f:
            f.write(script_content)
        temp_script.chmod(0o755)
        
        # Submit
        print(f"Submitting job: {job_name}")
        result = subprocess.run(['sbatch', str(temp_script)], capture_output=True, text=True)
        
        # Clean up
        temp_script.unlink()
        
        if result.returncode == 0:
            print(f"✓ Job submitted successfully")
            print(result.stdout)
        else:
            print(f"✗ Job submission failed")
            print(result.stderr)
            raise RuntimeError(f"sbatch failed with return code {result.returncode}")


def main():
    parser = argparse.ArgumentParser(
        description="Submit SLURM jobs using named configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available job configurations
  python submit_named_job.py --list
  
  # Submit a specific job
  python submit_named_job.py --job neutral_special_defects_diamond
  
  # Dry run to see what would be executed
  python submit_named_job.py --job neutral_extreme_mofs --dry-run
  
  # Override specific values
  python submit_named_job.py --job neutral_extreme_mofs --time 5:00:00 --batch-size 2
  
  # Use different test set
  python submit_named_job.py --job neutral_extreme_mofs --testset special_defects_diamond
        """
    )
    
    parser.add_argument('--list', action='store_true', help='List all available job configurations')
    parser.add_argument('--job', help='Name of the job configuration to submit')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be executed without submitting')
    
    # Override options
    parser.add_argument('--partition', help='Override SLURM partition')
    parser.add_argument('--nodes', type=int, help='Override number of nodes')
    parser.add_argument('--gpus', type=int, help='Override number of GPUs')
    parser.add_argument('--cpus-per-task', type=int, help='Override CPUs per task')
    parser.add_argument('--time', help='Override time limit')
    parser.add_argument('--qos', help='Override QOS')
    
    # Job-specific overrides
    parser.add_argument('--checkpoint', help='Override checkpoint path')
    parser.add_argument('--testset', help='Override test set')
    parser.add_argument('--dataset', help='Override dataset')
    parser.add_argument('--batch-size', type=int, help='Override batch size')
    parser.add_argument('--ode-method', help='Override ODE method')
    parser.add_argument('--ode-options', help='Override ODE options')
    
    args = parser.parse_args()
    
    # Initialize submitter
    script_dir = Path(__file__).parent
    submitter = NamedJobSubmitter(script_dir)
    
    # List jobs if requested
    if args.list:
        submitter.list_jobs()
        return
    
    # Submit job
    if not args.job:
        parser.error("--job is required (or use --list to see available jobs)")
    
    # Build overrides
    overrides = {}
    if args.partition:
        overrides['partition'] = args.partition
    if args.nodes:
        overrides['nodes'] = args.nodes
    if args.gpus:
        overrides['gpus'] = args.gpus
    if args.cpus_per_task:
        overrides['cpus_per_task'] = args.cpus_per_task
    if args.time:
        overrides['time'] = args.time
    if args.qos:
        overrides['qos'] = args.qos
    if args.checkpoint:
        overrides['checkpoint'] = args.checkpoint
    if args.testset:
        overrides['testset'] = args.testset
    if args.dataset:
        overrides['dataset'] = args.dataset
    if args.batch_size:
        overrides['batch_size'] = args.batch_size
    if args.ode_method:
        overrides['ode_method'] = args.ode_method
    if args.ode_options:
        overrides['ode_options'] = args.ode_options
    
    # Submit
    submitter.submit_job(args.job, overrides, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
