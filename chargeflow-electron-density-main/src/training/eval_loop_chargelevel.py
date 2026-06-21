# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import gc
import logging
import os
from argparse import Namespace
from pathlib import Path
from typing import Iterable
import matplotlib.pyplot as plt
import PIL.Image
import numpy as np

import torch
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.solver.ode_solver import ODESolver
from flow_matching.utils import ModelWrapper
from ..models.discrete_unet import DiscreteUNetModel
from ..models.ema import EMA
from torch.nn.modules import Module
from torch.nn.parallel import DistributedDataParallel
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import save_image
from . import distributed_mode
from .edm_time_discretization import get_time_discretization
from .train_loop import MASK_TOKEN

logger = logging.getLogger(__name__)

PRINT_FREQUENCY = 50


class CFGScaledModel(ModelWrapper):
    def __init__(self, model: Module):
        super().__init__(model)
        self.nfe_counter = 0

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, cfg_scale: float, label: torch.Tensor, concat_condition: torch.Tensor
    ):
        module = (
            self.model.module
            if isinstance(self.model, DistributedDataParallel)
            else self.model
        )
        is_discrete = isinstance(module, DiscreteUNetModel) or (
            isinstance(module, EMA) and isinstance(module.model, DiscreteUNetModel)
        )
        assert (
            cfg_scale == 0.0 or not is_discrete
        ), f"Cfg scaling does not work for the logit outputs of discrete models. Got cfg weight={cfg_scale} and model {type(self.model)}."
        t = torch.zeros(x.shape[0], device=x.device) + t
        if cfg_scale != 0.0:
            with torch.cuda.amp.autocast(), torch.no_grad():
                conditional = self.model(x, t, extra={"label": label})
                condition_free = self.model(x, t, extra={})
            result = (1.0 + cfg_scale) * conditional - cfg_scale * condition_free
        else:
            # Model is fully conditional, no cfg weighting needed
            with torch.cuda.amp.autocast(), torch.no_grad():
                # result = self.model(x, t, extra={"label": label})
                result = self.model(x, t, extra={"label": label, "concat_conditioning": concat_condition})

        self.nfe_counter += 1
        if is_discrete:
            return torch.softmax(result.to(dtype=torch.float32), dim=-1)
        else:
            return result.to(dtype=torch.float32)

    def reset_nfe_counter(self) -> None:
        self.nfe_counter = 0

    def get_nfe(self) -> int:
        return self.nfe_counter


def eval_model(
    model: DistributedDataParallel,
    data_loader: Iterable,
    device: torch.device,
    epoch: int,
    fid_samples: int,
    args: Namespace,
):
    gc.collect()
    cfg_scaled_model = CFGScaledModel(model=model)
    cfg_scaled_model.train(False)

    if args.discrete_flow_matching:
        scheduler = PolynomialConvexScheduler(n=3.0)
        path = MixtureDiscreteProbPath(scheduler=scheduler)
        p = torch.zeros(size=[257], dtype=torch.float32, device=device)
        p[256] = 1.0
        solver = MixtureDiscreteEulerSolver(
            model=cfg_scaled_model,
            path=path,
            vocabulary_size=257,
            source_distribution_p=p,
        )
    else:
        solver = ODESolver(velocity_model=cfg_scaled_model)
        ode_opts = args.ode_options

    fid_metric = FrechetInceptionDistance(normalize=True).to(
        device=device, non_blocking=True
    )

    num_synthetic = 0
    snapshots_saved = False
    if args.output_dir:
        (Path(args.output_dir) / "snapshots").mkdir(parents=True, exist_ok=True)

    for data_iter_step, (samples, labels, chargelevel) in enumerate(data_loader):
        samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        chargelevel = chargelevel.to(device, non_blocking=True)
        fid_metric.update(samples, real=True)

        if num_synthetic < fid_samples:
            cfg_scaled_model.reset_nfe_counter()
            if args.discrete_flow_matching:
                # Discrete sampling
                x_0 = (
                    torch.zeros(samples.shape, dtype=torch.long, device=device)
                    + MASK_TOKEN
                )
                if args.sym_func:
                    sym = lambda t: 12.0 * torch.pow(t, 2.0) * torch.pow(1.0 - t, 0.25)
                else:
                    sym = args.sym
                if args.sampling_dtype == "float32":
                    dtype = torch.float32
                elif args.sampling_dtype == "float64":
                    dtype = torch.float64

                synthetic_samples = solver.sample(
                    x_init=x_0,
                    step_size=1.0 / args.discrete_fm_steps,
                    verbose=False,
                    div_free=sym,
                    dtype_categorical=dtype,
                    label=labels,
                    cfg_scale=args.cfg_scale,
                )
            else:
                # Continuous sampling
                x_0 = torch.randn(samples.shape, dtype=torch.float32, device=device)

                if args.edm_schedule:
                    time_grid = get_time_discretization(nfes=ode_opts["nfe"])
                else:
                    time_grid = torch.tensor([0.0, 1.0], device=device)

                synthetic_samples = solver.sample(
                    time_grid=time_grid,
                    x_init=x_0,
                    method=args.ode_method,
                    return_intermediates=False,
                    atol=ode_opts["atol"] if "atol" in ode_opts else 1e-5,
                    rtol=ode_opts["rtol"] if "atol" in ode_opts else 1e-5,
                    step_size=ode_opts["step_size"]
                    if "step_size" in ode_opts
                    else None,
                    label=chargelevel,
                    cfg_scale=args.cfg_scale,
                )

                # Scaling to [0, 1] from [-1, 1]
                synthetic_samples = torch.clamp(
                    synthetic_samples * 0.5 + 0.5, min=0.0, max=1.0
                )
                synthetic_samples = torch.floor(synthetic_samples * 255)
            synthetic_samples = synthetic_samples.to(torch.float32) / 255.0
            logger.info(
                f"{samples.shape[0]} samples generated in {cfg_scaled_model.get_nfe()} evaluations."
            )
            if num_synthetic + synthetic_samples.shape[0] > fid_samples:
                synthetic_samples = synthetic_samples[: fid_samples - num_synthetic]
            fid_metric.update(synthetic_samples, real=False)
            num_synthetic += synthetic_samples.shape[0]
            if not snapshots_saved and args.output_dir:
                save_image(
                    synthetic_samples,
                    fp=Path(args.output_dir)
                    / "snapshots"
                    / f"{epoch}_{data_iter_step}.png",
                )
                snapshots_saved = True

            if args.save_fid_samples and args.output_dir:
                images_np = (
                    (synthetic_samples * 255.0)
                    .clip(0, 255)
                    .to(torch.uint8)
                    .permute(0, 2, 3, 1)
                    .cpu()
                    .numpy()
                )
                for batch_index, image_np in enumerate(images_np):
                    image_dir = Path(args.output_dir) / "fid_samples"
                    os.makedirs(image_dir, exist_ok=True)
                    image_path = (
                        image_dir
                        / f"{distributed_mode.get_rank()}_{data_iter_step}_{batch_index}.png"
                    )
                    PIL.Image.fromarray(image_np, "RGB").save(image_path)

        if not args.compute_fid:
            return {}

        if data_iter_step % PRINT_FREQUENCY == 0:
            # Sync fid metric to ensure that the processes dont deviate much.
            gc.collect()
            running_fid = fid_metric.compute()
            logger.info(
                f"Evaluating [{data_iter_step}/{len(data_loader)}] samples generated [{num_synthetic}/{fid_samples}] running fid {running_fid}"
            )

        if args.test_run:
            break

    return {"fid": float(fid_metric.compute().detach().cpu())}


class NormMAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mae = torch.nn.L1Loss(reduction="none")

    def forward(self, output, target):
        mae = self.mae(output, target)
        nelec = torch.sum(target, axis=(-3, -2, -1))
        mae = mae / nelec[..., None, None, None]
        return torch.sum(mae)


def eval_model_edp(
    model: DistributedDataParallel,
    data_loader: Iterable,
    device: torch.device,
    return_intermediates: bool,
    args: Namespace,
    visualize: bool = False,
):
    gc.collect()
    cfg_scaled_model = CFGScaledModel(model=model)
    cfg_scaled_model.train(False)

    if args.discrete_flow_matching:
        scheduler = PolynomialConvexScheduler(n=3.0)
        path = MixtureDiscreteProbPath(scheduler=scheduler)
        p = torch.zeros(size=[257], dtype=torch.float32, device=device)
        p[256] = 1.0
        solver = MixtureDiscreteEulerSolver(
            model=cfg_scaled_model,
            path=path,
            vocabulary_size=257,
            source_distribution_p=p,
        )
    else:
        solver = ODESolver(velocity_model=cfg_scaled_model)
        ode_opts = args.ode_options

    normmae_loss = NormMAE()
    total_normmae = 0.0
    total_samples = 0
    synthetic_samples_list = []
    PREDICTION_DIR = '/vast/minhtrin/Charged_electron_density/FlowEDP/predictions/'
    VISUALIZATION_DIR = '/vast/minhtrin/Charged_electron_density/FlowEDP/visualizations/'
    from tqdm import tqdm
    for data_iter_step, (samples, labels, chargelevel) in tqdm(enumerate(data_loader), total=len(data_loader)):
        samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        cfg_scaled_model.reset_nfe_counter()
        if args.discrete_flow_matching:
            x_0 = (
                torch.zeros(samples.shape, dtype=torch.long, device=device)
                + MASK_TOKEN
            )
            if args.sym_func:
                sym = lambda t: 12.0 * torch.pow(t, 2.0) * torch.pow(1.0 - t, 0.25)
            else:
                sym = args.sym
            if args.sampling_dtype == "float32":
                dtype = torch.float32
            elif args.sampling_dtype == "float64":
                dtype = torch.float64

            synthetic_samples = solver.sample(
                x_init=x_0,
                step_size=1.0 / args.discrete_fm_steps,
                verbose=False,
                div_free=sym,
                dtype_categorical=dtype,
                label=None,
                cfg_scale=args.cfg_scale,
            )
        else:
            if args.start_sad:
                x_0 = samples
            else:
                x_0 = torch.randn(samples.shape, dtype=torch.float32, device=device)*torch.sqrt(torch.tensor(1e-05))

            if args.edm_schedule:
                time_grid = get_time_discretization(nfes=ode_opts["nfe"])
            else:
                time_grid = torch.tensor([0.0, 1.0], device=device)
            synthetic_samples = solver.sample(
                time_grid=time_grid,
                x_init=x_0,
                method=args.ode_method,
                # return_intermediates=False,
                atol=ode_opts["atol"] if "atol" in ode_opts else 1e-5,
                rtol=ode_opts["rtol"] if "atol" in ode_opts else 1e-5,
                step_size=ode_opts["step_size"]
                if "step_size" in ode_opts
                else None,
                label=chargelevel,
                cfg_scale=args.cfg_scale,
                return_intermediates=return_intermediates,
                concat_condition=samples if not args.start_sad else None,
            )
        
        if return_intermediates:
            output = synthetic_samples[-1].to(torch.float32)
        else:
            output = synthetic_samples.to(torch.float32)
        if not args.start_sad:
            # print('Samples:', samples)
            # print('Synthetic Samples:', synthetic_samples)
            # print('x_0:', x_0)
            synthetic_samples = samples + output
            # print('Synthetic Samples after addition:', synthetic_samples)
            # print('Labels:', labels)
        else:
            synthetic_samples = output

        # # Compute NormMAE between synthetic_samples and labels
        # print('Sum of synthetic samples:', torch.sum(synthetic_samples, axis=(-3,-2,-1)))
        # print('Sum of labels:', torch.sum(labels, axis=(-3,-2,-1)))
        # print('Sum of samples:', torch.sum(samples, axis=(-3,-2,-1)))

        # # synthetic_samples = torch.clamp(synthetic_samples, min=torch.min(labels), max=torch.max(labels))
        # # print('Clamped synthetic samples:', torch.sum(synthetic_samples, axis=(-3,-2,-1)))
        # synthetic_samples = synthetic_samples / torch.sum(synthetic_samples, axis=(-3,-2,-1))[...,None,None,None]
        # synthetic_samples = synthetic_samples * torch.sum(samples, axis=(-3,-2,-1))[...,None,None,None]
        # print('Normalized synthetic samples:', torch.sum(synthetic_samples, axis=(-3,-2,-1)))

        normmae = normmae_loss(synthetic_samples, labels.to(torch.float32))
        print(f"NormMAE for batch {data_iter_step}: {normmae.item()}")
        prediction = synthetic_samples.cpu().detach().numpy()
        synthetic_samples_list.append(prediction)
        total_normmae += normmae.item()
        total_samples += samples.shape[0]

        if visualize:
        # --- Visualization of prediction, ground truth, and low_res_upsampled ---
        # Prediction
            pred_tensor = prediction
            if pred_tensor.ndim == 5:
                pred_tensor = pred_tensor[0]
            if pred_tensor.shape[0] > 1:
                pred_slice = pred_tensor[0]
            else:
                pred_slice = pred_tensor.squeeze(0)
            z_center = pred_slice.shape[0] // 2
            plt.imshow(pred_slice[z_center], cmap='viridis')
            plt.title(f'Central Slice of Prediction Sample {data_iter_step} (z={z_center})')
            plt.colorbar()
            out_path_pred = os.path.join(PREDICTION_DIR, f'prediction_slice_sample_{data_iter_step}.png')
            plt.savefig(out_path_pred)
            plt.close()
            # print(f"Saved central slice visualization to {out_path_pred}")

            if not args.start_sad:
                pred_tensor = output.cpu().detach().numpy()
                if pred_tensor.ndim == 5:
                    pred_tensor = pred_tensor[0]
                if pred_tensor.shape[0] > 1:
                    pred_slice = pred_tensor[0]
                else:
                    pred_slice = pred_tensor.squeeze(0)
                plt.imshow(pred_slice[z_center], cmap='viridis')
                plt.title(f'Central Slice of Residual Prediction Sample {data_iter_step} (z={z_center})')
                plt.colorbar()
                out_path_pred = os.path.join(PREDICTION_DIR, f'prediction_residual_slice_sample_{data_iter_step}.png')
                plt.savefig(out_path_pred)
                plt.close()
                # print(f"Saved central slice visualization to {out_path_pred}")


            # Ground truth
            gt_tensor = labels.cpu()
            if gt_tensor.ndim == 5:
                gt_tensor = gt_tensor[0]
            if gt_tensor.shape[0] > 1:
                gt_slice = gt_tensor[0]
            else:
                gt_slice = gt_tensor.squeeze(0)
            z_center = gt_slice.shape[0] // 2
            plt.imshow(gt_slice[z_center].cpu().numpy(), cmap='viridis')
            plt.title(f'Central Slice of Ground Truth Sample {data_iter_step} (z={z_center})')
            plt.colorbar()
            out_path_gt = os.path.join(PREDICTION_DIR, f'ground_truth_slice_sample_{data_iter_step}.png')
            plt.savefig(out_path_gt)
            plt.close()
            # print(f"Saved central slice visualization to {out_path_gt}")

            # Low-res upsampled
            lr_tensor = samples.cpu()
            if lr_tensor.ndim == 5:
                lr_tensor = lr_tensor[0]
            if lr_tensor.shape[0] > 1:
                lr_slice = lr_tensor[0]
            else:
                lr_slice = lr_tensor.squeeze(0)
            z_center = lr_slice.shape[0] // 2
            plt.imshow(lr_slice[z_center].cpu().numpy(), cmap='viridis')
            plt.title(f'Central Slice of Low-Res Upsampled Sample {data_iter_step} (z={z_center})')
            plt.colorbar()
            out_path_lr = os.path.join(PREDICTION_DIR, f'lowres_slice_sample_{data_iter_step}.png')
            plt.savefig(out_path_lr)
            plt.close()
            # print(f"Saved central slice visualization to {out_path_lr}")

            # Absolute error visualization
            abs_error = pred_slice - gt_slice.cpu().numpy()
            plt.imshow(abs_error[z_center], cmap='hot')
            plt.title(f'Central Slice of Absolute Error Sample {data_iter_step} (z={z_center})')
            plt.colorbar()
            out_path_error = os.path.join(PREDICTION_DIR, f'abs_error_slice_sample_{data_iter_step}.png')
            plt.savefig(out_path_error)
            plt.close()
            # print(f"Saved central slice absolute error visualization to {out_path_error}")

            if not args.start_sad:
                # Absolute residual visualization
                abs_error_residual = np.abs(gt_slice - lr_slice)
                plt.imshow(abs_error_residual[z_center], cmap='viridis')
                plt.title(f'Central Slice of Absolute Residual Sample {data_iter_step} (z={z_center})')
                plt.colorbar()
                out_path_error = os.path.join(PREDICTION_DIR, f'abs_residual_slice_sample_{data_iter_step}.png')
                plt.savefig(out_path_error)
                plt.close()
                # print(f"Saved central slice absolute residual visualization to {out_path_error}")

                #Visualize the residual error
                residual_error = np.abs(pred_slice - abs_error_residual.cpu().numpy())
                plt.imshow(residual_error[z_center], cmap='hot')
                plt.title(f'Central Slice of Residual Error Sample {data_iter_step} (z={z_center})')
                plt.colorbar()
                out_path_error = os.path.join(PREDICTION_DIR, f'residual_error_slice_sample_{data_iter_step}.png')
                plt.savefig(out_path_error)
                plt.close()
                # print(f"Saved central slice residual error visualization to {out_path_error}")

        if args.test_run:
            break

    mean_normmae = total_normmae / total_samples if total_samples > 0 else 0.0
    return {"normmae": mean_normmae, "samples": synthetic_samples_list}
