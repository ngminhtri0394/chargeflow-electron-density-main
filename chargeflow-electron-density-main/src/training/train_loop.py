# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import gc
import logging
import math
from typing import Iterable

import torch
from flow_matching.path import CondOTProbPath, MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from ..models.ema import EMA
from torch.nn.parallel import DistributedDataParallel
from torchmetrics.aggregation import MeanMetric
from .grad_scaler import NativeScalerWithGradNormCount
# from training.eval_loop import NormMAE

class NormMAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mae = torch.nn.L1Loss(reduction="none")

    def forward(self, output, target):
        mae = self.mae(output, target)
        nelec = torch.sum(target, axis=(-3, -2, -1))
        mae = mae / nelec[..., None, None, None]
        return torch.sum(mae)

logger = logging.getLogger(__name__)

MASK_TOKEN = 256
PRINT_FREQUENCY = 50


def skewed_timestep_sample(num_samples: int, device: torch.device) -> torch.Tensor:
    P_mean = -1.2
    P_std = 1.2
    rnd_normal = torch.randn((num_samples,), device=device)
    sigma = (rnd_normal * P_std + P_mean).exp()
    time = 1 / (1 + sigma)
    time = torch.clip(time, min=0.0001, max=1.0)
    return time


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    lr_schedule: torch.torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    epoch: int,
    loss_scaler: NativeScalerWithGradNormCount,
    args: argparse.Namespace,
):
    gc.collect()
    model.train(True)
    batch_loss = MeanMetric().to(device, non_blocking=True)
    epoch_loss = MeanMetric().to(device, non_blocking=True)

    accum_iter = args.accum_iter
    if args.discrete_flow_matching:
        scheduler = PolynomialConvexScheduler(n=3.0)
        path = MixtureDiscreteProbPath(scheduler=scheduler)
    else:
        path = CondOTProbPath()

    # Prepare optional loss modules
    normmae_loss = None
    if not args.discrete_flow_matching and args.loss_type in {"normmae", "hybrid"}:
        normmae_loss = NormMAE()

    for data_iter_step, (low_rs, high_rs) in enumerate(data_loader):
        if data_iter_step % accum_iter == 0:
            optimizer.zero_grad()
            batch_loss.reset()
            if data_iter_step > 0 and args.test_run:
                break

        low_rs = low_rs.to(device, non_blocking=True)
        high_rs = high_rs.to(device, non_blocking=True)

        if torch.rand(1) < args.class_drop_prob:
            conditioning = {}
        else:
            # conditioning = {"label": low_rs}
            if args.start_sad:
                conditioning = {"label": low_rs}
            else:
                conditioning = {"label": low_rs, "concat_conditioning": low_rs}

        if args.discrete_flow_matching:
            low_rs = (low_rs * 255.0).to(torch.long)
            t = torch.torch.rand(low_rs.shape[0]).to(device)

            # sample probability path
            x_0 = (
                torch.zeros(low_rs.shape, dtype=torch.long, device=device) + MASK_TOKEN
            )
            path_sample = path.sample(t=t, x_0=x_0, x_1=low_rs)

            # discrete flow matching loss
            logits = model(path_sample.x_t, t=t, extra=conditioning)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape([-1, 257]), low_rs.reshape([-1])
            ).mean()
        else:
            # Scaling to [-1, 1] from [0, 1]
            # samples = samples * 2.0 - 1.0
            # print('Low RS shape:', low_rs.shape, 'High RS shape:', high_rs.shape)
            noise = torch.randn_like(low_rs).to(device)*torch.sqrt(torch.tensor(4e-05))
            # noise = torch.randn_like(low_rs).to(device)
            if args.norm_rho:
                if args.start_sad:
                    scaled_low_rs = low_rs.sign() * (1 + 1.25 * low_rs.abs()).log() / 4.2
                    scaled_high_rs = high_rs.sign() * (1 + 1.25 * high_rs.abs()).log() / 4.2

            if args.skewed_timesteps:
                t = skewed_timestep_sample(low_rs.shape[0], device=device)
            else:
                t = torch.torch.rand(low_rs.shape[0]).to(device)
            
            if args.start_sad:
                if args.norm_rho:
                     path_sample = path.sample(t=t, x_0=scaled_low_rs, x_1=scaled_high_rs)
                else:
                   path_sample = path.sample(t=t, x_0=low_rs, x_1=high_rs)
            else:
                residual = high_rs - low_rs
                if args.norm_rho:
                    residual = residual.sign() * (1 + 10 * residual.abs()).log() / 4.6
                path_sample = path.sample(t=t, x_0=noise, x_1=residual)

            x_t = path_sample.x_t
            u_t = path_sample.dx_t

            with torch.amp.autocast(device_type="cuda"):
                pred = model(x_t, t, extra=conditioning)
                l2_loss = torch.pow(pred - u_t, 2).mean()
                if normmae_loss is not None:
                    if args.start_sad:
                        if args.norm_rho:
                            predict_scaled_rho = (x_t + pred * (1-t))*4.2
                            final_step_prediction = torch.sign(predict_scaled_rho) * (predict_scaled_rho.abs().exp() - 1) / 1.25
                        else:
                            final_step_prediction = x_t + pred * (1-t)
                    else:
                        if args.norm_rho:
                            # Rescaling back to original scale
                            predict_scaled_residual = (x_t + pred * (1-t))*4.6
                            final_step_prediction = low_rs + torch.sign(predict_scaled_residual) * (predict_scaled_residual.abs().exp() - 1) / 10
                        else:
                            final_step_prediction = low_rs + x_t + pred * (1-t)
                    nm_loss = normmae_loss(final_step_prediction, high_rs.to(u_t.dtype))
                    if args.loss_type == "normmae":
                        loss = nm_loss
                    else:  # hybrid
                        # print('L2 Loss:', l2_loss.item(), 'NormMAE Loss:', nm_loss.item())
                        loss = l2_loss + args.alpha * nm_loss
                else:
                    loss = l2_loss

        loss_value = loss.item()
        batch_loss.update(loss)
        epoch_loss.update(loss)

        if not math.isfinite(loss_value):
            raise ValueError(f"Loss is {loss_value}, stopping training")

        loss /= accum_iter

        # Loss scaler applies the optimizer when update_grad is set to true.
        # Otherwise just updates the internal gradient scales
        apply_update = (data_iter_step + 1) % accum_iter == 0
        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=apply_update,
        )
        if apply_update and isinstance(model, EMA):
            model.update_ema()
        elif (
            apply_update
            and isinstance(model, DistributedDataParallel)
            and isinstance(model.module, EMA)
        ):
            model.module.update_ema()

        lr = optimizer.param_groups[0]["lr"]
        if data_iter_step % PRINT_FREQUENCY == 0:
            logger.info(
                f"Epoch {epoch} [{data_iter_step}/{len(data_loader)}]: loss = {batch_loss.compute()}, lr = {lr}"
            )

    lr_schedule.step()
    return {"loss": float(epoch_loss.compute().detach().cpu())}
