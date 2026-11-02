#!/usr/bin/env python3
"""
Train DIRECT for CMEMS Mediterranean chlorophyll-a reconstruction.

This trainer is adapted from the DIRECT method described in Grega Rovšček's
thesis, with Chl-a-specific preprocessing supplied by chl_data.py.

Expected project layout
-----------------------
src/
    chl_data.py
    train_direct.py
    models/
        direct.py

The dataset must be the corrected version with:
    input_mode="direct"
    direct_context           [B,3,H,W]
    direct_temporal_masks    [B,8,H,W]
    current_masked           [B,1,H,W]
    current_visible_mask     [B,1,H,W]
    target                   [B,1,H,W]
    target_valid_mask        [B,1,H,W]
    loss_mask                [B,1,H,W]
    ocean_mask               [B,1,H,W]
    doy                      [B,2]

Optional auxiliary tensors when --aux-mode is enabled:
    sst                      [B,1,H,W]
    sst_valid_mask           [B,1,H,W]
    bbp443                   [B,1,H,W]
    bbp443_valid_mask        [B,1,H,W]

Training procedure
------------------
For each sample:

1. Draw a continuous flow/noise coordinate k ~ Uniform(0,1).

2. Draw Gaussian noise epsilon ~ N(0,I).

3. Construct the linear noise-to-data path:
       x_k = (1-k) * x_target + k * epsilon

   k=0 is clean data.
   k=1 is Gaussian noise.

4. Apply DIRECT's FUSE observation-guided rectification:
       x_tilde_k =
           [x_k * (1-M_visible)
            + x_observed * M_visible] * M_ocean

   Thus known centre-frame observations are fixed exactly and land is zero.

5. Rectify the initial Gaussian noise in the same way:
       epsilon_tilde = FUSE(epsilon)

6. The thesis-specific DIRECT target velocity is:
       u = x_target - epsilon_tilde

   This is the reverse/noise-to-data velocity used by the DIRECT training
   section. At visible observations this target becomes zero because FUSE
   fixes those pixels to the observation.

7. If auxiliary conditioning is enabled, normalized SST and/or BBP443
   value/mask pairs are concatenated to the canonical 12 DIRECT channels.
   They are conditioning only and do not participate in FUSE.

8. The DIRECT U-Net predicts v_theta and minimizes masked MSE over valid
   centre-frame ground-truth pixels.

Implementation details from the thesis
--------------------------------------
- AdamW
- learning rate = 1e-4
- beta1 = 0.9
- beta2 = 0.95
- 200 epochs
- 20-epoch linear warm-up
- cosine decay for the remaining epochs
- minimum learning rate = 1e-8

The thesis excerpt available to us does not specify the AdamW weight-decay
coefficient. It is therefore configurable here and defaults to 0.0 instead of
silently inventing a value.

Validation
----------
Validation masks are deterministic in chl_data.py. This trainer additionally
uses deterministic sample-specific k/noise so validation loss is reproducible
across epochs.

The final thesis RMSE/MAE/correlation/SSIM must be computed separately with
evaluate_direct.py using the reverse flow sampler and the same deterministic
hidden test pixels as the U-Net/STIMP baselines.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
SRC_DIR = THIS_FILE.parent
PROJECT_DIR = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chl_data_new import ChlorophyllReconstructionDataset
from models.direct_multivar import DIRECT, fuse_rectify, auxiliary_channels_for_mode


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------

def make_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
):
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(**kwargs)


# ---------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------

def to_device(
    batch: dict,
    key: str,
    device: torch.device,
) -> torch.Tensor:
    return batch[key].to(
        device=device,
        non_blocking=True,
    )


def count_parameters(model: torch.nn.Module) -> int:
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def build_auxiliary_context(
    batch: dict,
    aux_mode: str,
    device: torch.device,
) -> torch.Tensor | None:
    """Assemble DIRECT auxiliary conditioning channels in a stable order."""
    mode = str(aux_mode).lower()
    if mode == "none":
        return None

    channels = []
    if mode in {"sst", "sst_bbp443"}:
        channels.extend([
            to_device(batch, "sst", device),
            to_device(batch, "sst_valid_mask", device),
        ])
    if mode in {"bbp443", "sst_bbp443"}:
        channels.extend([
            to_device(batch, "bbp443", device),
            to_device(batch, "bbp443_valid_mask", device),
        ])
    if not channels:
        raise ValueError(f"Unsupported aux_mode: {aux_mode!r}")

    spatial = channels[0].shape[-2:]
    batch_size = channels[0].shape[0]
    for tensor in channels:
        if tensor.ndim != 4 or tensor.shape[1] != 1:
            raise ValueError(
                "Each auxiliary tensor must be [B,1,H,W], "
                f"got {tuple(tensor.shape)}"
            )
        if tensor.shape[0] != batch_size or tensor.shape[-2:] != spatial:
            raise ValueError("Auxiliary tensors have inconsistent shapes")
    return torch.cat(channels, dim=1)


# ---------------------------------------------------------------------
# Flow-matching construction
# ---------------------------------------------------------------------

def construct_direct_training_state(
    target: torch.Tensor,
    current_masked: torch.Tensor,
    visible_mask: torch.Tensor,
    ocean_mask: torch.Tensor,
    k: torch.Tensor,
    noise: torch.Tensor | None = None,
):
    """
    Construct DIRECT training input and target velocity.

    Parameters
    ----------
    target:
        [B,1,H,W], clean normalized Chl-a where genuinely unavailable
        target pixels are zero and excluded by target_valid_mask later.

    current_masked:
        [B,1,H,W], centre observations available to the model after
        artificial real-cloud masking.

    visible_mask:
        [B,1,H,W], 1 where current_masked is genuinely visible.

    ocean_mask:
        [B,1,H,W], 1 over ocean and 0 over land.

    k:
        [B], continuous flow coordinate in [0,1].

    noise:
        Optional [B,1,H,W] Gaussian noise for deterministic validation.

    Returns
    -------
    noisy_rectified:
        FUSE-rectified point on the interpolation path.

    target_velocity:
        DIRECT reverse/noise-to-data velocity:
            target - rectified_initial_noise

    rectified_noise:
        FUSE-rectified initial Gaussian noise.
    """
    if noise is None:
        noise = torch.randn_like(target)

    if target.ndim != 4 or target.shape[1] != 1:
        raise ValueError(
            f"target must be [B,1,H,W], got {tuple(target.shape)}"
        )

    if k.ndim != 1 or k.shape[0] != target.shape[0]:
        raise ValueError(
            f"k must be [B], got {tuple(k.shape)} for B={target.shape[0]}"
        )

    k_view = k[:, None, None, None]

    # Eq. 4.1 path:
    # x_hat(k) = (1-k) * x + k * epsilon
    noisy_center = (
        (1.0 - k_view) * target
        + k_view * noise
    )

    # FUSE rectification at the sampled k.
    noisy_rectified = fuse_rectify(
        noisy_center=noisy_center,
        observed_center=current_masked,
        visible_mask=visible_mask,
        ocean_mask=ocean_mask,
    )

    # Thesis training section defines the target velocity using the
    # initial Gaussian noise after the same observation-guided rectification.
    rectified_noise = fuse_rectify(
        noisy_center=noise,
        observed_center=current_masked,
        visible_mask=visible_mask,
        ocean_mask=ocean_mask,
    )

    target_velocity = (
        target
        - rectified_noise
    )

    return (
        noisy_rectified,
        target_velocity,
        rectified_noise,
    )


def masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Pixel-weighted masked MSE.

    mask is expected to be [B,1,H,W].
    """
    mask = mask.to(
        dtype=prediction.dtype
    )

    denominator = mask.sum()

    if denominator.item() <= 0:
        raise RuntimeError(
            "masked_mse received a mask with zero valid pixels"
        )

    squared_error = (
        prediction - target
    ) ** 2

    return (
        squared_error * mask
    ).sum() / denominator


# ---------------------------------------------------------------------
# Learning-rate schedule
# ---------------------------------------------------------------------

def make_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup_epochs: int,
    min_lr: float,
    base_lr: float,
):
    """
    Thesis schedule:
      - linear warm-up
      - cosine decay
      - minimum LR = 1e-8 by default

    We use SequentialLR:
      LinearLR for warm-up_epochs
      CosineAnnealingLR for the remaining epochs
    """

    if warmup_epochs < 0:
        raise ValueError(
            "warmup_epochs must be >= 0"
        )

    if warmup_epochs >= epochs:
        raise ValueError(
            "warmup_epochs must be smaller than epochs"
        )

    if not (0.0 <= min_lr < base_lr):
        raise ValueError(
            "Require 0 <= min_lr < base_lr"
        )

    if warmup_epochs == 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs),
            eta_min=min_lr,
        )

    # Start at 1/warmup_epochs of the base LR and increase linearly.
    start_factor = 1.0 / float(
        max(warmup_epochs, 1)
    )

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=start_factor,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )

    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(
            1,
            epochs - warmup_epochs,
        ),
        eta_min=min_lr,
    )

    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            warmup,
            cosine,
        ],
        milestones=[
            warmup_epochs,
        ],
    )


# ---------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------

def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_val_loss: float,
    args,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (
            scheduler.state_dict()
            if scheduler is not None
            else None
        ),
        "best_val_loss": float(best_val_loss),
        "args": vars(args),
        "training_version": "direct_chla_flow_matching_multivar_v2",
    }

    if scaler is not None:
        payload["scaler_state_dict"] = (
            scaler.state_dict()
        )

    torch.save(
        payload,
        path,
    )


# ---------------------------------------------------------------------
# Train epoch
# ---------------------------------------------------------------------

def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    use_amp,
    grad_clip,
    log_every,
    epoch,
    aux_mode,
):
    model.train()

    weighted_loss_sum = 0.0
    valid_pixel_count = 0

    hidden_weighted_loss_sum = 0.0
    hidden_pixel_count = 0

    start = time.time()

    for batch_idx, batch in enumerate(
        loader,
        start=1,
    ):
        target = to_device(
            batch,
            "target",
            device,
        )
        current_masked = to_device(
            batch,
            "current_masked",
            device,
        )
        visible_mask = to_device(
            batch,
            "current_visible_mask",
            device,
        )
        target_valid_mask = to_device(
            batch,
            "target_valid_mask",
            device,
        )
        loss_mask = to_device(
            batch,
            "loss_mask",
            device,
        )
        ocean_mask = to_device(
            batch,
            "ocean_mask",
            device,
        )
        temporal_context = to_device(
            batch,
            "direct_context",
            device,
        )
        temporal_masks = to_device(
            batch,
            "direct_temporal_masks",
            device,
        )
        doy = to_device(
            batch,
            "doy",
            device,
        )

        auxiliary_context = build_auxiliary_context(
            batch=batch,
            aux_mode=aux_mode,
            device=device,
        )

        batch_size = target.shape[0]

        # Continuous flow coordinate k ~ Uniform(0,1).
        k = torch.rand(
            batch_size,
            device=device,
            dtype=torch.float32,
        )

        noise = torch.randn_like(
            target
        )

        (
            noisy_rectified,
            target_velocity,
            _,
        ) = construct_direct_training_state(
            target=target,
            current_masked=current_masked,
            visible_mask=visible_mask,
            ocean_mask=ocean_mask,
            k=k,
            noise=noise,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            predicted_velocity = model(
                noisy_center=noisy_rectified,
                temporal_context=temporal_context,
                temporal_masks=temporal_masks,
                doy=doy,
                k=k,
                auxiliary_context=auxiliary_context,
            )

            # Thesis objective: valid observed centre-frame pixels.
            loss = masked_mse(
                prediction=predicted_velocity,
                target=target_velocity,
                mask=target_valid_mask,
            )

            # Diagnostic only: flow loss on artificially hidden pixels.
            hidden_loss = masked_mse(
                prediction=predicted_velocity,
                target=target_velocity,
                mask=loss_mask,
            )

        if not torch.isfinite(loss):
            print("\nNON-FINITE DIRECT TRAINING STATE")
            print("epoch:", epoch)
            print("batch:", batch_idx)
            print("k min/max:", k.min().item(), k.max().item())

            tensors = {
                "target": target,
                "current_masked": current_masked,
                "visible_mask": visible_mask,
                "ocean_mask": ocean_mask,
                "temporal_context": temporal_context,
                "temporal_masks": temporal_masks,
                "noisy_rectified": noisy_rectified,
                "target_velocity": target_velocity,
                "predicted_velocity": predicted_velocity,
            }

            if auxiliary_context is not None:
                tensors["auxiliary_context"] = auxiliary_context

            for name, tensor in tensors.items():
                finite = torch.isfinite(tensor)

                print(
                    name,
                    "finite=", finite.all().item(),
                    "nan=", torch.isnan(tensor).sum().item(),
                    "inf=", torch.isinf(tensor).sum().item(),
                )

                if finite.any():
                    values = tensor[finite]

                    print(
                        "   min=",
                        values.min().item(),
                        "max=",
                        values.max().item(),
                        "mean=",
                        values.float().mean().item(),
                    )

            raise RuntimeError(
                f"Non-finite training loss at "
                f"epoch={epoch}, batch={batch_idx}: "
                f"{loss.item()}"
            )

        if scaler is not None and use_amp:
            scaler.scale(
                loss
            ).backward()

            if grad_clip > 0:
                scaler.unscale_(
                    optimizer
                )
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    grad_clip,
                )

            scaler.step(
                optimizer
            )
            scaler.update()

        else:
            loss.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    grad_clip,
                )

            optimizer.step()

        n_valid = int(
            target_valid_mask.sum().item()
        )
        n_hidden = int(
            loss_mask.sum().item()
        )

        weighted_loss_sum += (
            float(
                loss.detach().item()
            )
            * max(
                n_valid,
                1,
            )
        )
        valid_pixel_count += (
            n_valid
        )

        hidden_weighted_loss_sum += (
            float(
                hidden_loss.detach().item()
            )
            * max(
                n_hidden,
                1,
            )
        )
        hidden_pixel_count += (
            n_hidden
        )

        if (
            batch_idx % log_every == 0
            or batch_idx == len(loader)
        ):
            elapsed = (
                time.time()
                - start
            )

            avg_sec = (
                elapsed
                / batch_idx
            )

            print(
                f"    Train batch "
                f"{batch_idx:,}/"
                f"{len(loader):,} "
                f"| flow loss: "
                f"{loss.item():.6f} "
                f"| hidden flow: "
                f"{hidden_loss.item():.6f} "
                f"| valid px: "
                f"{n_valid:,} "
                f"| avg: "
                f"{avg_sec:.3f}s",
                flush=True,
            )

    train_loss = (
        weighted_loss_sum
        / valid_pixel_count
        if valid_pixel_count > 0
        else math.nan
    )

    train_hidden_loss = (
        hidden_weighted_loss_sum
        / hidden_pixel_count
        if hidden_pixel_count > 0
        else math.nan
    )

    return (
        train_loss,
        train_hidden_loss,
    )


# ---------------------------------------------------------------------
# Deterministic validation
# ---------------------------------------------------------------------

@torch.no_grad()
def validate(
    model,
    loader,
    device,
    use_amp,
    seed,
    aux_mode,
):
    model.eval()

    weighted_loss_sum = 0.0
    valid_pixel_count = 0

    hidden_weighted_loss_sum = 0.0
    hidden_pixel_count = 0

    for batch in loader:
        target = to_device(
            batch,
            "target",
            device,
        )
        current_masked = to_device(
            batch,
            "current_masked",
            device,
        )
        visible_mask = to_device(
            batch,
            "current_visible_mask",
            device,
        )
        target_valid_mask = to_device(
            batch,
            "target_valid_mask",
            device,
        )
        loss_mask = to_device(
            batch,
            "loss_mask",
            device,
        )
        ocean_mask = to_device(
            batch,
            "ocean_mask",
            device,
        )
        temporal_context = to_device(
            batch,
            "direct_context",
            device,
        )
        temporal_masks = to_device(
            batch,
            "direct_temporal_masks",
            device,
        )
        doy = to_device(
            batch,
            "doy",
            device,
        )

        auxiliary_context = build_auxiliary_context(
            batch=batch,
            aux_mode=aux_mode,
            device=device,
        )

        time_idx = int(
            batch["time_idx"][0].item()
        )
        patch_y = int(
            batch["patch_y"][0].item()
        )
        patch_x = int(
            batch["patch_x"][0].item()
        )

        # Deterministic sample-specific RNG on CPU.
        sample_seed = (
            int(seed)
            + 1_000_003 * time_idx
            + 10_007 * patch_y
            + 101 * patch_x
        ) % (2**63 - 1)

        generator = torch.Generator(
            device="cpu"
        )
        generator.manual_seed(
            int(sample_seed)
        )

        # One deterministic k per sample.
        k = torch.rand(
            target.shape[0],
            generator=generator,
            dtype=torch.float32,
        ).to(
            device,
            non_blocking=True,
        )

        noise = torch.randn(
            target.shape,
            generator=generator,
            dtype=torch.float32,
        ).to(
            device,
            non_blocking=True,
        )

        (
            noisy_rectified,
            target_velocity,
            _,
        ) = construct_direct_training_state(
            target=target,
            current_masked=current_masked,
            visible_mask=visible_mask,
            ocean_mask=ocean_mask,
            k=k,
            noise=noise,
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            predicted_velocity = model(
                noisy_center=noisy_rectified,
                temporal_context=temporal_context,
                temporal_masks=temporal_masks,
                doy=doy,
                k=k,
                auxiliary_context=auxiliary_context,
            )

            loss = masked_mse(
                prediction=predicted_velocity,
                target=target_velocity,
                mask=target_valid_mask,
            )

            hidden_loss = masked_mse(
                prediction=predicted_velocity,
                target=target_velocity,
                mask=loss_mask,
            )

        n_valid = int(
            target_valid_mask.sum().item()
        )
        n_hidden = int(
            loss_mask.sum().item()
        )

        weighted_loss_sum += (
            float(
                loss.item()
            )
            * max(
                n_valid,
                1,
            )
        )
        valid_pixel_count += (
            n_valid
        )

        hidden_weighted_loss_sum += (
            float(
                hidden_loss.item()
            )
            * max(
                n_hidden,
                1,
            )
        )
        hidden_pixel_count += (
            n_hidden
        )

    val_loss = (
        weighted_loss_sum
        / valid_pixel_count
        if valid_pixel_count > 0
        else math.nan
    )

    val_hidden_loss = (
        hidden_weighted_loss_sum
        / hidden_pixel_count
        if hidden_pixel_count > 0
        else math.nan
    )

    return (
        val_loss,
        val_hidden_loss,
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train DIRECT flow-matching model "
            "for Chl-a reconstruction"
        )
    )

    # ---------------------------------------------------------
    # Data
    # ---------------------------------------------------------

    parser.add_argument(
        "--processed-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--model-dir",
        type=Path,
        default=(
            PROJECT_DIR
            / "models"
            / "direct"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "The thesis-sized DIRECT U-Net is large; "
            "batch size 1 is the safe starting point on a 32GB V100."
        ),
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--aux-mode",
        type=str,
        default="none",
        choices=["none", "sst", "bbp443", "sst_bbp443"],
        help=(
            "Auxiliary conditioning fields. 'sst' adds normalized SST + mask; "
            "'bbp443' adds normalized BBP443 + mask; 'sst_bbp443' adds both."
        ),
    )

    # ---------------------------------------------------------
    # Architecture
    # ---------------------------------------------------------

    parser.add_argument(
        "--conditioning-dim",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--k-embedding-dim",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--attention-heads",
        type=int,
        default=8,
        help=(
            "The thesis text does not specify the number of attention heads; "
            "8 is an explicit configurable implementation choice."
        ),
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--group-norm-groups",
        type=int,
        default=32,
    )

    # ---------------------------------------------------------
    # Optimization: thesis defaults
    # ---------------------------------------------------------

    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
    )

    parser.add_argument(
        "--beta2",
        type=float,
        default=0.95,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help=(
            "AdamW weight decay is not specified in the thesis excerpt. "
            "Set explicitly if you obtain the original implementation value."
        ),
    )

    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--min-learning-rate",
        type=float,
        default=1e-8,
    )

    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
    )

    # No early stopping in thesis; optional here.
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help=(
            "Optional early stopping. "
            "0 disables it, matching the thesis' fixed 200-epoch training."
        ),
    )

    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-6,
    )

    # ---------------------------------------------------------
    # Runtime
    # ---------------------------------------------------------

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--no-amp",
        action="store_true",
    )

    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    args = parse_args()

    seed_everything(
        args.seed
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    use_amp = (
        device.type == "cuda"
        and not args.no_amp
    )

    args.model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_path = (
        args.model_dir
        / "best.pt"
    )
    last_path = (
        args.model_dir
        / "last.pt"
    )
    config_path = (
        args.model_dir
        / "training_config.json"
    )

    with open(
        config_path,
        "w",
    ) as f:
        json.dump(
            vars(args),
            f,
            indent=2,
            default=str,
        )

    print("=" * 76)
    print("DIRECT / CHL-A FLOW-MATCHING TRAINING")
    print("=" * 76)
    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )
        print(
            "CUDA memory:",
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB",
        )

    print("AMP:", use_amp)
    print("Processed dir:", args.processed_dir)
    print("Model dir:", args.model_dir)
    print("Batch size:", args.batch_size)
    print("Epochs:", args.epochs)
    print("Learning rate:", args.learning_rate)
    print("AdamW betas:", (args.beta1, args.beta2))
    print("Weight decay:", args.weight_decay)
    print("Warm-up epochs:", args.warmup_epochs)
    print("Min LR:", args.min_learning_rate)
    print("Auxiliary mode:", args.aux_mode)
    print("Auxiliary channels:", auxiliary_channels_for_mode(args.aux_mode))
    print()

    # ---------------------------------------------------------
    # Datasets
    # ---------------------------------------------------------

    train_dataset = (
        ChlorophyllReconstructionDataset(
            processed_dir=args.processed_dir,
            split="train",
            input_mode="direct",
            seed=args.seed,
            aux_mode=args.aux_mode,
        )
    )

    val_dataset = (
        ChlorophyllReconstructionDataset(
            processed_dir=args.processed_dir,
            split="validation",
            input_mode="direct",
            seed=args.seed,
            aux_mode=args.aux_mode,
        )
    )

    print(
        "Train samples:",
        f"{len(train_dataset):,}",
    )
    print(
        "Validation samples:",
        f"{len(val_dataset):,}",
    )
    print(
        "Normalization:",
        f"mean={train_dataset.train_mean:.6f}, "
        f"std={train_dataset.train_std:.6f}",
    )
    print()

    pin_memory = (
        device.type == "cuda"
    )

    train_loader = make_loader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        prefetch_factor=args.prefetch_factor,
    )

    val_loader = make_loader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        prefetch_factor=args.prefetch_factor,
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    aux_channels = auxiliary_channels_for_mode(args.aux_mode)

    model = DIRECT(
        auxiliary_channels=aux_channels,
        output_channels=1,
        channels=(
            128,
            256,
            384,
            512,
        ),
        blocks_per_level=(
            5,
            3,
            3,
            3,
        ),
        conditioning_dim=(
            args.conditioning_dim
        ),
        k_embedding_dim=(
            args.k_embedding_dim
        ),
        attention_heads=(
            args.attention_heads
        ),
        dropout=args.dropout,
        group_norm_groups=(
            args.group_norm_groups
        ),
    ).to(
        device
    )

    print(
        "Trainable parameters:",
        f"{count_parameters(model):,}",
    )
    print("DIRECT input channels:", model.input_channels)
    print()

    # ---------------------------------------------------------
    # Preflight one real validation batch
    # ---------------------------------------------------------
    preflight_batch = next(iter(val_loader))
    with torch.no_grad():
        preflight_aux = build_auxiliary_context(
            preflight_batch, args.aux_mode, device
        )
        preflight_target = to_device(preflight_batch, "target", device)
        preflight_current = to_device(preflight_batch, "current_masked", device)
        preflight_visible = to_device(preflight_batch, "current_visible_mask", device)
        preflight_ocean = to_device(preflight_batch, "ocean_mask", device)
        preflight_context = to_device(preflight_batch, "direct_context", device)
        preflight_masks = to_device(preflight_batch, "direct_temporal_masks", device)
        preflight_doy = to_device(preflight_batch, "doy", device)
        preflight_k = torch.full(
            (preflight_target.shape[0],), 0.5, device=device, dtype=torch.float32
        )
        preflight_center, _, _ = construct_direct_training_state(
            target=preflight_target,
            current_masked=preflight_current,
            visible_mask=preflight_visible,
            ocean_mask=preflight_ocean,
            k=preflight_k,
            noise=torch.zeros_like(preflight_target),
        )
        preflight_output = model(
            noisy_center=preflight_center,
            temporal_context=preflight_context,
            temporal_masks=preflight_masks,
            doy=preflight_doy,
            k=preflight_k,
            auxiliary_context=preflight_aux,
        )
        if preflight_output.shape != preflight_target.shape:
            raise RuntimeError(
                f"Preflight shape mismatch: pred={tuple(preflight_output.shape)} "
                f"target={tuple(preflight_target.shape)}"
            )
        if not torch.isfinite(preflight_output).all():
            raise RuntimeError("DIRECT preflight produced non-finite output")
        if preflight_aux is not None:
            print("Auxiliary context shape:", tuple(preflight_aux.shape))
        print("Preflight output:", tuple(preflight_output.shape), "finite=True")
        print()
    del preflight_batch, preflight_aux, preflight_target, preflight_current
    del preflight_visible, preflight_ocean, preflight_context, preflight_masks
    del preflight_doy, preflight_k, preflight_center, preflight_output
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # Optimizer + LR schedule
    # ---------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(
            args.beta1,
            args.beta2,
        ),
        weight_decay=(
            args.weight_decay
        ),
    )

    scheduler = make_lr_scheduler(
        optimizer=optimizer,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        min_lr=args.min_learning_rate,
        base_lr=args.learning_rate,
    )

    scaler = None

    if use_amp:
        try:
            scaler = torch.amp.GradScaler(
                "cuda"
            )
        except Exception:
            scaler = (
                torch.cuda.amp.GradScaler()
            )

    # ---------------------------------------------------------
    # Resume
    # ---------------------------------------------------------

    start_epoch = 1
    best_val_loss = float("inf")
    patience_counter = 0

    if args.resume is not None:
        print(
            "Resuming from:",
            args.resume,
        )

        checkpoint = torch.load(
            args.resume,
            map_location=device,
            weights_only=False,
        )

        checkpoint_version = checkpoint.get("training_version")
        allowed_versions = {"direct_chla_flow_matching_multivar_v2"}
        if args.aux_mode == "none":
            allowed_versions.add("direct_chla_flow_matching_v1")
        if checkpoint_version not in allowed_versions:
            raise RuntimeError(
                "Checkpoint is incompatible with this DIRECT trainer: "
                f"training_version={checkpoint_version!r}, aux_mode={args.aux_mode!r}"
            )
        checkpoint_args = checkpoint.get("args", {})
        checkpoint_aux_mode = checkpoint_args.get("aux_mode", "none")
        if checkpoint_aux_mode != args.aux_mode:
            raise RuntimeError(
                "Cannot resume with a different auxiliary mode. "
                f"Checkpoint aux_mode={checkpoint_aux_mode!r}, requested={args.aux_mode!r}"
            )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

        scheduler_state = checkpoint.get(
            "scheduler_state_dict"
        )

        if scheduler_state is not None:
            scheduler.load_state_dict(
                scheduler_state
            )

        if (
            scaler is not None
            and "scaler_state_dict"
            in checkpoint
        ):
            scaler.load_state_dict(
                checkpoint[
                    "scaler_state_dict"
                ]
            )

        start_epoch = (
            int(
                checkpoint["epoch"]
            )
            + 1
        )

        best_val_loss = float(
            checkpoint.get(
                "best_val_loss",
                float("inf"),
            )
        )

        print(
            "Resume epoch:",
            start_epoch,
        )
        print(
            "Best validation loss:",
            best_val_loss,
        )
        print()

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    training_start = (
        time.time()
    )

    try:
        for epoch in range(
            start_epoch,
            args.epochs + 1,
        ):
            epoch_start = (
                time.time()
            )

            current_lr = (
                optimizer.param_groups[0][
                    "lr"
                ]
            )

            print()
            print("=" * 76)
            print(
                f"Epoch {epoch:03d}/"
                f"{args.epochs}"
            )
            print("=" * 76)
            print(
                "Learning rate:",
                f"{current_lr:.8g}",
            )

            (
                train_loss,
                train_hidden_loss,
            ) = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                use_amp=use_amp,
                grad_clip=args.grad_clip,
                log_every=args.log_every,
                epoch=epoch,
                aux_mode=args.aux_mode,
            )

            train_time = (
                time.time()
                - epoch_start
            )

            val_start = (
                time.time()
            )

            (
                val_loss,
                val_hidden_loss,
            ) = validate(
                model=model,
                loader=val_loader,
                device=device,
                use_amp=use_amp,
                seed=args.seed,
                aux_mode=args.aux_mode,
            )

            val_time = (
                time.time()
                - val_start
            )

            improved = (
                val_loss
                < (
                    best_val_loss
                    - args.min_delta
                )
            )

            if improved:
                best_val_loss = (
                    val_loss
                )
                patience_counter = 0
            else:
                patience_counter += 1

            # Step once after each completed epoch.
            scheduler.step()

            save_checkpoint(
                path=last_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_val_loss=best_val_loss,
                args=args,
            )

            if improved:
                save_checkpoint(
                    path=best_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    best_val_loss=best_val_loss,
                    args=args,
                )

            total_epoch_time = (
                time.time()
                - epoch_start
            )

            next_lr = (
                optimizer.param_groups[0][
                    "lr"
                ]
            )

            print()
            print(
                f"Epoch {epoch:03d}/"
                f"{args.epochs} "
                f"| Train flow: "
                f"{train_loss:.6f} "
                f"| Train hidden: "
                f"{train_hidden_loss:.6f} "
                f"| Val flow: "
                f"{val_loss:.6f} "
                f"| Val hidden: "
                f"{val_hidden_loss:.6f}"
            )

            print(
                f"    Train: "
                f"{train_time / 60:.1f} min "
                f"| Val: "
                f"{val_time / 60:.1f} min "
                f"| Total: "
                f"{total_epoch_time / 60:.1f} min"
            )

            print(
                f"    Best val flow: "
                f"{best_val_loss:.6f} "
                f"| Next LR: "
                f"{next_lr:.8g} "
                f"| Patience: "
                f"{patience_counter}"
            )

            if improved:
                print(
                    "    -> Saved new best model"
                )

            if (
                args.patience > 0
                and patience_counter
                >= args.patience
            ):
                print()
                print(
                    "Early stopping triggered."
                )
                break

    finally:
        train_dataset.close()
        val_dataset.close()

    total_hours = (
        time.time()
        - training_start
    ) / 3600.0

    print()
    print("=" * 76)
    print("TRAINING FINISHED")
    print("=" * 76)
    print(
        "Total training time:",
        f"{total_hours:.2f} h",
    )
    print(
        "Best validation flow loss:",
        best_val_loss,
    )
    print(
        "Best checkpoint:",
        best_path,
    )
    print(
        "Last checkpoint:",
        last_path,
    )


if __name__ == "__main__":
    main()
