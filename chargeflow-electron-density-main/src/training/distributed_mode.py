# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import os
from datetime import timedelta
import random
import time
import torch
import torch.distributed as dist


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        args.world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        args.gpu = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        args.dist_url = "tcp://%s:%s" % (
            os.environ["MASTER_ADDR"],
            os.environ["MASTER_PORT"],
        )
        os.environ["LOCAL_RANK"] = str(args.gpu)
        os.environ["RANK"] = str(args.rank)
        os.environ["WORLD_SIZE"] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        print("Using distributed mode with environment variables")
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif (
        "SLURM_PROCID" in os.environ and os.environ["SLURM_JOB_NAME"] != "bash"
    ):  # Exclude interactive shells
        print("Using distributed mode with SLURM")
        args.rank = int(os.environ["SLURM_PROCID"])
        args.world_size = int(os.environ["SLURM_NTASKS"])
        print('args.rank', args.rank)
        print('args.world_size', args.world_size)
        args.gpu = args.rank % torch.cuda.device_count()
        
        # Set up master address and port for SLURM
        if "SLURM_NODELIST" in os.environ:
            import subprocess
            master_node = subprocess.check_output(
                "scontrol show hostname $SLURM_NODELIST | head -n 1", 
                shell=True
            ).decode().strip()
            os.environ["MASTER_ADDR"] = master_node
        else:
            os.environ["MASTER_ADDR"] = "localhost"
        
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = str(random.randint(10000, 20000))
        # random.seed(time.time())
        # os.environ["MASTER_PORT"] = str(random.randint(10000, 20000))
        
        args.dist_url = f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True
    # Remove the random port assignment for SLURM
    # import random
    # os.environ['MASTER_ADDR'] = 'localhost'
    # port = str(random.randint(1000, 9999))
    # os.environ['MASTER_PORT'] = port

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(
        "| distributed init (rank {}): {}, gpu {}".format(
            args.rank, args.dist_url, args.gpu
        ),
        flush=True,
    )
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
        timeout=timedelta(hours=1),
    )
    torch.distributed.barrier()

# Add SLURM debugging and explicit master address/port setup for distributed training
if __name__ == "__main__":
    import sys

    print("SLURM_PROCID=", os.environ.get("SLURM_PROCID"))
    print("SLURM_NTASKS=", os.environ.get("SLURM_NTASKS"))
    print("SLURM_NODELIST=", os.environ.get("SLURM_NODELIST"))
    print("MASTER_ADDR=", os.environ.get("MASTER_ADDR"))
    print("MASTER_PORT=", os.environ.get("MASTER_PORT"))
    print("CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))
