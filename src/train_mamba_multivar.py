#!/usr/bin/env python3
"""
Train the spatiotemporal Mamba-3 chlorophyll reconstruction model.

Expected project layout
-----------------------
src/
    chl_data.py
    train_mamba.py
    models/
        mamba_chl.py

Expected dataset mode
---------------------
ChlorophyllReconstructionDataset(..., input_mode="mamba") must return:

    mamba_sequence          [T,C,H,W]

where C depends on --aux-mode:
    none        -> 3 channels/day
    sst         -> 5 channels/day
    bbp443      -> 5 channels/day
    sst_bbp443  -> 7 channels/day

    mamba_doy               [T,2]
    mamba_time_valid        [T]
    mamba_offsets           [T]
    target                  [1,H,W]
    loss_mask               [1,H,W]
    target_valid_mask       [1,H,W]
    current_visible_mask    [1,H,W]

Training objective
------------------
The model predicts normalized log-CHL for the target day.

By default the loss is MSE ONLY on the artificially hidden, genuinely valid
target pixels (`loss_mask`). This matches the controlled reconstruction task:
the same pixels hidden from the U-Net/STIMP/DIRECT benchmarks are the pixels
the model is trained to reconstruct.

Optionally, a small visible-pixel consistency loss can be enabled with:
    --visible-loss-weight <value>

The default is 0.0 so the first experiment remains a clean hidden-pixel
reconstruction objective.

Important
---------
Mamba-3 is used here as a task-specific spatiotemporal backbone. The optimizer,
learning-rate schedule, and CNN encoder/decoder are implementation choices for
this thesis adaptation; they are not claimed to be hyperparameters prescribed
by the Mamba-3 language-model paper.

Recommended first run on H100
-----------------------------
- sequence length: 15
- batch size: 1
- FP32 (default; do NOT pass --amp initially)
- d_model: 256
- d_state: 64
- 4 bidirectional Mamba-3 layers
- headdim: 64
- chunk_size: 64
- AdamW lr=1e-4
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
from models.mamba_chl_multivar import ChlorophyllMamba


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# Data
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


def to_device(
    batch: dict,
    key: str,
    device: torch.device,
) -> torch.Tensor:
    return batch[key].to(
        device=device,
        non_blocking=True,
    )


# ---------------------------------------------------------------------
# Losses / metrics
# ---------------------------------------------------------------------

def masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    mask = mask.to(
        dtype=prediction.dtype
    )

    denom = mask.sum()

    if denom.item() <= 0:
        raise RuntimeError(
            "masked_mse received a mask with zero valid pixels."
        )

    return (
        ((prediction - target) ** 2)
        * mask
    ).sum() / denom


def masked_mae(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    mask = mask.to(
        dtype=prediction.dtype
    )

    denom = mask.sum()

    if denom.item() <= 0:
        raise RuntimeError(
            "masked_mae received a mask with zero valid pixels."
        )

    return (
        torch.abs(
            prediction - target
        )
        * mask
    ).sum() / denom


def visible_consistency_mask(
    target_valid_mask: torch.Tensor,
    current_visible_mask: torch.Tensor,
) -> torch.Tensor:
    return (
        target_valid_mask.bool()
        & current_visible_mask.bool()
    ).to(
        dtype=target_valid_mask.dtype
    )


# ---------------------------------------------------------------------
# Learning-rate schedule
# ---------------------------------------------------------------------

def make_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup_epochs: int,
    min_lr: float,
    base_lr: float,
):
    """
    Small warm-up followed by cosine decay.

    This is a thesis implementation choice, not a Mamba-3 paper requirement.
    """
    if warmup_epochs < 0:
        raise ValueError(
            "warmup_epochs must be >= 0"
        )

    if warmup_epochs >= epochs:
        raise ValueError(
            "warmup_epochs must be smaller than epochs"
        )

    if not (
        0.0 <= min_lr < base_lr
    ):
        raise ValueError(
            "Require 0 <= min_lr < base_lr"
        )

    if warmup_epochs == 0:
        return (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(
                    1,
                    epochs,
                ),
                eta_min=min_lr,
            )
        )

    start_factor = 1.0 / float(
        max(
            warmup_epochs,
            1,
        )
    )

    warmup = (
        torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=start_factor,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
    )

    cosine = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(
                1,
                epochs - warmup_epochs,
            ),
            eta_min=min_lr,
        )
    )

    return (
        torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                warmup,
                cosine,
            ],
            milestones=[
                warmup_epochs,
            ],
        )
    )


# ---------------------------------------------------------------------
# Model call
# ---------------------------------------------------------------------

def model_forward(
    model: torch.nn.Module,
    batch: dict,
    device: torch.device,
) -> torch.Tensor:
    sequence = to_device(
        batch,
        "mamba_sequence",
        device,
    )

    doy = to_device(
        batch,
        "mamba_doy",
        device,
    )

    time_valid = to_device(
        batch,
        "mamba_time_valid",
        device,
    )

    offsets = to_device(
        batch,
        "mamba_offsets",
        device,
    )

    return model(
        mamba_sequence=sequence,
        mamba_doy=doy,
        mamba_time_valid=time_valid,
        mamba_offsets=offsets,
    )


# ---------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------

@torch.no_grad()
def preflight_check(
    model,
    loader,
    device,
    use_amp,
):
    """
    Run one real dataset batch through the actual Mamba-3 kernels before
    starting a long Slurm job.
    """
    model.eval()

    batch = next(
        iter(loader)
    )

    target = to_device(
        batch,
        "target",
        device,
    )

    loss_mask = to_device(
        batch,
        "loss_mask",
        device,
    )

    sequence = batch[
        "mamba_sequence"
    ]

    print("Preflight batch:")
    print(
        "  mamba_sequence:",
        tuple(
            sequence.shape
        ),
    )
    print(
        "  target:",
        tuple(
            batch[
                "target"
            ].shape
        ),
    )
    print(
        "  hidden pixels:",
        int(
            batch[
                "loss_mask"
            ].sum().item()
        ),
    )

    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=use_amp,
    ):
        prediction = model_forward(
            model=model,
            batch=batch,
            device=device,
        )

        loss = masked_mse(
            prediction=prediction,
            target=target,
            mask=loss_mask,
        )

    print(
        "  prediction:",
        tuple(
            prediction.shape
        ),
    )
    print(
        "  prediction finite:",
        bool(
            torch.isfinite(
                prediction
            ).all()
        ),
    )
    print(
        "  initial hidden MSE:",
        float(
            loss.item()
        ),
    )

    if (
        prediction.shape
        != target.shape
    ):
        raise RuntimeError(
            f"Prediction shape "
            f"{tuple(prediction.shape)} "
            f"!= target shape "
            f"{tuple(target.shape)}"
        )

    if not torch.isfinite(
        prediction
    ).all():
        raise RuntimeError(
            "Preflight produced non-finite predictions."
        )

    if not torch.isfinite(loss):
        raise RuntimeError(
            "Preflight produced a non-finite loss."
        )

    print(
        "Preflight check: PASSED"
    )
    print()


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    use_amp,
    grad_clip,
    visible_loss_weight,
    log_every,
    epoch,
):
    model.train()

    hidden_sse_weighted = 0.0
    hidden_mae_weighted = 0.0
    hidden_pixels = 0

    total_loss_weighted = 0.0
    total_loss_weight = 0

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

        loss_mask = to_device(
            batch,
            "loss_mask",
            device,
        )

        target_valid = to_device(
            batch,
            "target_valid_mask",
            device,
        )

        current_visible = to_device(
            batch,
            "current_visible_mask",
            device,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            prediction = model_forward(
                model=model,
                batch=batch,
                device=device,
            )

            hidden_loss = masked_mse(
                prediction=prediction,
                target=target,
                mask=loss_mask,
            )

            hidden_mae = masked_mae(
                prediction=prediction,
                target=target,
                mask=loss_mask,
            )

            loss = hidden_loss

            visible_loss = None

            if visible_loss_weight > 0:
                vis_mask = (
                    visible_consistency_mask(
                        target_valid_mask=(
                            target_valid
                        ),
                        current_visible_mask=(
                            current_visible
                        ),
                    )
                )

                if vis_mask.sum().item() > 0:
                    visible_loss = (
                        masked_mse(
                            prediction=(
                                prediction
                            ),
                            target=target,
                            mask=vis_mask,
                        )
                    )

                    loss = (
                        loss
                        + visible_loss_weight
                        * visible_loss
                    )

        if not torch.isfinite(loss):
            print()
            print(
                "NON-FINITE MAMBA TRAINING STATE"
            )
            print("epoch:", epoch)
            print("batch:", batch_idx)

            for name, tensor in {
                "target": target,
                "loss_mask": (
                    loss_mask
                ),
                "prediction": (
                    prediction
                ),
            }.items():
                finite = torch.isfinite(
                    tensor
                )

                print(
                    name,
                    "finite=",
                    bool(
                        finite.all()
                    ),
                    "nan=",
                    int(
                        torch.isnan(
                            tensor
                        ).sum().item()
                    ),
                    "inf=",
                    int(
                        torch.isinf(
                            tensor
                        ).sum().item()
                    ),
                )

                if finite.any():
                    vals = tensor[
                        finite
                    ]
                    print(
                        "   min=",
                        float(
                            vals.min().item()
                        ),
                        "max=",
                        float(
                            vals.max().item()
                        ),
                        "mean=",
                        float(
                            vals.float()
                            .mean().item()
                        ),
                    )

            raise RuntimeError(
                f"Non-finite Mamba training loss "
                f"at epoch={epoch}, "
                f"batch={batch_idx}: "
                f"{loss.item()}"
            )

        if (
            scaler is not None
            and use_amp
        ):
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

        n_hidden = int(
            loss_mask.sum().item()
        )

        hidden_sse_weighted += (
            float(
                hidden_loss.detach()
                .item()
            )
            * n_hidden
        )

        hidden_mae_weighted += (
            float(
                hidden_mae.detach()
                .item()
            )
            * n_hidden
        )

        hidden_pixels += n_hidden

        total_loss_weighted += (
            float(
                loss.detach().item()
            )
            * n_hidden
        )

        total_loss_weight += (
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

            msg = (
                f"    Train batch "
                f"{batch_idx:,}/"
                f"{len(loader):,} "
                f"| hidden MSE: "
                f"{hidden_loss.item():.6f} "
                f"| hidden MAE: "
                f"{hidden_mae.item():.6f}"
            )

            if (
                visible_loss
                is not None
            ):
                msg += (
                    f" | visible MSE: "
                    f"{visible_loss.item():.6f}"
                )

            msg += (
                f" | hidden px: "
                f"{n_hidden:,} "
                f"| avg: "
                f"{avg_sec:.3f}s"
            )

            print(
                msg,
                flush=True,
            )

    return {
        "loss": (
            total_loss_weighted
            / max(
                total_loss_weight,
                1,
            )
        ),
        "hidden_mse": (
            hidden_sse_weighted
            / max(
                hidden_pixels,
                1,
            )
        ),
        "hidden_mae": (
            hidden_mae_weighted
            / max(
                hidden_pixels,
                1,
            )
        ),
        "hidden_pixels": (
            hidden_pixels
        ),
    }


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

@torch.no_grad()
def validate(
    model,
    loader,
    device,
    use_amp,
    visible_loss_weight,
):
    model.eval()

    hidden_sse_weighted = 0.0
    hidden_mae_weighted = 0.0
    hidden_pixels = 0

    total_loss_weighted = 0.0
    total_loss_weight = 0

    for batch in loader:
        target = to_device(
            batch,
            "target",
            device,
        )

        loss_mask = to_device(
            batch,
            "loss_mask",
            device,
        )

        target_valid = to_device(
            batch,
            "target_valid_mask",
            device,
        )

        current_visible = to_device(
            batch,
            "current_visible_mask",
            device,
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            prediction = model_forward(
                model=model,
                batch=batch,
                device=device,
            )

            hidden_loss = masked_mse(
                prediction=prediction,
                target=target,
                mask=loss_mask,
            )

            hidden_mae = masked_mae(
                prediction=prediction,
                target=target,
                mask=loss_mask,
            )

            loss = hidden_loss

            if visible_loss_weight > 0:
                vis_mask = (
                    visible_consistency_mask(
                        target_valid_mask=(
                            target_valid
                        ),
                        current_visible_mask=(
                            current_visible
                        ),
                    )
                )

                if (
                    vis_mask.sum().item()
                    > 0
                ):
                    visible_loss = (
                        masked_mse(
                            prediction=(
                                prediction
                            ),
                            target=target,
                            mask=vis_mask,
                        )
                    )

                    loss = (
                        loss
                        + visible_loss_weight
                        * visible_loss
                    )

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Non-finite validation loss."
            )

        n_hidden = int(
            loss_mask.sum().item()
        )

        hidden_sse_weighted += (
            float(
                hidden_loss.item()
            )
            * n_hidden
        )

        hidden_mae_weighted += (
            float(
                hidden_mae.item()
            )
            * n_hidden
        )

        hidden_pixels += (
            n_hidden
        )

        total_loss_weighted += (
            float(
                loss.item()
            )
            * n_hidden
        )

        total_loss_weight += (
            n_hidden
        )

    return {
        "loss": (
            total_loss_weighted
            / max(
                total_loss_weight,
                1,
            )
        ),
        "hidden_mse": (
            hidden_sse_weighted
            / max(
                hidden_pixels,
                1,
            )
        ),
        "hidden_mae": (
            hidden_mae_weighted
            / max(
                hidden_pixels,
                1,
            )
        ),
        "hidden_pixels": (
            hidden_pixels
        ),
    }


# ---------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------

def save_checkpoint(
    path: Path,
    model,
    optimizer,
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
        "epoch": int(
            epoch
        ),
        "model_state_dict": (
            model.state_dict()
        ),
        "optimizer_state_dict": (
            optimizer.state_dict()
        ),
        "scheduler_state_dict": (
            scheduler.state_dict()
            if scheduler is not None
            else None
        ),
        "best_val_loss": float(
            best_val_loss
        ),
        "args": vars(args),
        "training_version": (
            "mamba3_chla_multivar_v2"
        ),
    }

    if scaler is not None:
        payload[
            "scaler_state_dict"
        ] = scaler.state_dict()

    torch.save(
        payload,
        path,
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Train bidirectional Mamba-3 "
            "for Chl-a reconstruction"
        )
    )

    # Data
    p.add_argument(
        "--processed-dir",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--model-dir",
        type=Path,
        default=(
            PROJECT_DIR
            / "models"
            / "mamba"
        ),
    )

    p.add_argument(
        "--sequence-length",
        type=int,
        default=15,
    )

    p.add_argument(
        "--aux-mode",
        type=str,
        default="none",
        choices=[
            "none",
            "sst",
            "bbp443",
            "sst_bbp443",
        ],
        help=(
            "Auxiliary per-day inputs. "
            "none=CHL only; sst=CHL+SST; "
            "bbp443=CHL+BBP443; "
            "sst_bbp443=CHL+SST+BBP443."
        ),
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )

    p.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    p.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
    )

    # Model
    p.add_argument(
        "--encoder-channels",
        type=int,
        nargs=3,
        default=[
            64,
            128,
            256,
        ],
    )

    p.add_argument(
        "--d-model",
        type=int,
        default=256,
    )

    p.add_argument(
        "--d-state",
        type=int,
        default=64,
    )

    p.add_argument(
        "--num-mamba-layers",
        type=int,
        default=4,
    )

    p.add_argument(
        "--headdim",
        type=int,
        default=64,
    )

    p.add_argument(
        "--expand",
        type=int,
        default=2,
    )

    p.add_argument(
        "--chunk-size",
        type=int,
        default=64,
    )

    p.add_argument(
        "--dropout",
        type=float,
        default=0.0,
    )

    p.add_argument(
        "--group-norm-groups",
        type=int,
        default=8,
    )

    p.add_argument(
        "--causal",
        action="store_true",
        help=(
            "Use only forward/causal Mamba. "
            "Default is bidirectional because "
            "the reconstruction sequence contains "
            "both past and future context."
        ),
    )

    p.add_argument(
        "--no-temporal-metadata",
        action="store_true",
    )

    p.add_argument(
        "--predict-residual",
        action="store_true",
    )

    # Objective
    p.add_argument(
        "--visible-loss-weight",
        type=float,
        default=0.0,
    )

    # Optimization
    p.add_argument(
        "--epochs",
        type=int,
        default=100,
    )

    p.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
    )

    p.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    p.add_argument(
        "--warmup-epochs",
        type=int,
        default=5,
    )

    p.add_argument(
        "--min-learning-rate",
        type=float,
        default=1e-6,
    )

    p.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
    )

    p.add_argument(
        "--patience",
        type=int,
        default=20,
        help=(
            "Early stopping patience. "
            "0 disables early stopping."
        ),
    )

    p.add_argument(
        "--min-delta",
        type=float,
        default=1e-6,
    )

    # Runtime
    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    p.add_argument(
        "--log-every",
        type=int,
        default=250,
    )

    p.add_argument(
        "--amp",
        action="store_true",
        help=(
            "Enable FP16 autocast. "
            "Leave OFF for the first V100 run."
        ),
    )

    p.add_argument(
        "--resume",
        type=Path,
        default=None,
    )

    return p.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    args = parse_args()

    if (
        args.sequence_length < 1
        or args.sequence_length
        % 2 == 0
    ):
        raise ValueError(
            "--sequence-length must be "
            "a positive odd integer."
        )

    seed_everything(
        args.seed
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if device.type != "cuda":
        raise RuntimeError(
            "Mamba-3 training requires a CUDA GPU. "
            "Run this script on a GPU compute node."
        )

    use_amp = bool(
        args.amp
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
    print("MAMBA-3 CHL-A TRAINING")
    print("=" * 76)
    print("Device:", device)
    print(
        "GPU:",
        torch.cuda.get_device_name(
            0
        ),
    )
    print(
        "CUDA memory:",
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB",
    )
    print("AMP:", use_amp)
    print(
        "Sequence length:",
        args.sequence_length,
    )
    print(
        "Auxiliary mode:",
        args.aux_mode,
    )
    print(
        "Bidirectional:",
        not args.causal,
    )
    print(
        "Chunk size:",
        args.chunk_size,
    )
    print()

    # ---------------------------------------------------------
    # Datasets
    # ---------------------------------------------------------

    train_dataset = (
        ChlorophyllReconstructionDataset(
            processed_dir=(
                args.processed_dir
            ),
            split="train",
            input_mode="mamba",
            seed=args.seed,
            mamba_sequence_length=(
                args.sequence_length
            ),
            aux_mode=args.aux_mode,
        )
    )

    val_dataset = (
        ChlorophyllReconstructionDataset(
            processed_dir=(
                args.processed_dir
            ),
            split="validation",
            input_mode="mamba",
            seed=args.seed,
            mamba_sequence_length=(
                args.sequence_length
            ),
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

    train_loader = make_loader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=(
            args.prefetch_factor
        ),
    )

    val_loader = make_loader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=(
            args.prefetch_factor
        ),
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = ChlorophyllMamba(
        aux_mode=args.aux_mode,
        encoder_channels=tuple(
            args.encoder_channels
        ),
        d_model=args.d_model,
        d_state=args.d_state,
        num_mamba_layers=(
            args.num_mamba_layers
        ),
        headdim=args.headdim,
        expand=args.expand,
        chunk_size=args.chunk_size,
        bidirectional=(
            not args.causal
        ),
        dropout=args.dropout,
        group_norm_groups=(
            args.group_norm_groups
        ),
        use_temporal_metadata=(
            not args.no_temporal_metadata
        ),
        center_index=(
            args.sequence_length // 2
        ),
        predict_residual=(
            args.predict_residual
        ),
    ).to(
        device
    )

    print(
        "Mamba channels/day:",
        model.in_channels,
    )
    if hasattr(model, "channel_names"):
        print(
            "Channel order:",
            tuple(model.channel_names),
        )
    print()

    parameter_count = sum(
        p.numel()
        for p
        in model.parameters()
        if p.requires_grad
    )

    print(
        "Trainable parameters:",
        f"{parameter_count:,}",
    )
    print()

    # ---------------------------------------------------------
    # Preflight real Mamba kernel test
    # ---------------------------------------------------------

    preflight_check(
        model=model,
        loader=val_loader,
        device=device,
        use_amp=use_amp,
    )

    # ---------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=(
            args.weight_decay
        ),
    )

    scheduler = make_scheduler(
        optimizer=optimizer,
        epochs=args.epochs,
        warmup_epochs=(
            args.warmup_epochs
        ),
        min_lr=(
            args.min_learning_rate
        ),
        base_lr=(
            args.learning_rate
        ),
    )

    scaler = None

    if use_amp:
        try:
            scaler = (
                torch.amp.GradScaler(
                    "cuda"
                )
            )
        except Exception:
            scaler = (
                torch.cuda.amp.GradScaler()
            )

    # ---------------------------------------------------------
    # Resume
    # ---------------------------------------------------------

    start_epoch = 1
    best_val_loss = (
        float("inf")
    )
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

        checkpoint_version = checkpoint.get(
            "training_version"
        )

        allowed_versions = {
            "mamba3_chla_multivar_v2",
        }
        if args.aux_mode == "none":
            allowed_versions.add(
                "mamba3_chla_bidirectional_v1"
            )

        if checkpoint_version not in allowed_versions:
            raise RuntimeError(
                "Checkpoint is incompatible with this Mamba trainer: "
                f"training_version={checkpoint_version!r}, "
                f"aux_mode={args.aux_mode!r}"
            )

        checkpoint_args = checkpoint.get("args", {})
        checkpoint_aux_mode = checkpoint_args.get(
            "aux_mode",
            "none",
        )
        if checkpoint_aux_mode != args.aux_mode:
            raise RuntimeError(
                "Cannot resume with a different auxiliary mode. "
                f"Checkpoint aux_mode={checkpoint_aux_mode!r}, "
                f"requested aux_mode={args.aux_mode!r}"
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

        scheduler_state = (
            checkpoint.get(
                "scheduler_state_dict"
            )
        )

        if (
            scheduler_state
            is not None
        ):
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
            "Starting at epoch:",
            start_epoch,
        )
        print(
            "Best val loss:",
            best_val_loss,
        )
        print()

    # ---------------------------------------------------------
    # Train
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
                optimizer.param_groups[
                    0
                ]["lr"]
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

            train_stats = (
                train_one_epoch(
                    model=model,
                    loader=train_loader,
                    optimizer=optimizer,
                    scaler=scaler,
                    device=device,
                    use_amp=use_amp,
                    grad_clip=(
                        args.grad_clip
                    ),
                    visible_loss_weight=(
                        args.visible_loss_weight
                    ),
                    log_every=(
                        args.log_every
                    ),
                    epoch=epoch,
                )
            )

            val_start = (
                time.time()
            )

            val_stats = validate(
                model=model,
                loader=val_loader,
                device=device,
                use_amp=use_amp,
                visible_loss_weight=(
                    args.visible_loss_weight
                ),
            )

            val_time = (
                time.time()
                - val_start
            )

            improved = (
                val_stats["loss"]
                < (
                    best_val_loss
                    - args.min_delta
                )
            )

            if improved:
                best_val_loss = (
                    val_stats["loss"]
                )
                patience_counter = 0
            else:
                patience_counter += 1

            scheduler.step()

            save_checkpoint(
                path=last_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_val_loss=(
                    best_val_loss
                ),
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
                    best_val_loss=(
                        best_val_loss
                    ),
                    args=args,
                )

            epoch_time = (
                time.time()
                - epoch_start
            )

            next_lr = (
                optimizer.param_groups[
                    0
                ]["lr"]
            )

            print()
            print(
                f"Epoch {epoch:03d}/"
                f"{args.epochs} "
                f"| Train hidden MSE: "
                f"{train_stats['hidden_mse']:.6f} "
                f"| Train hidden MAE: "
                f"{train_stats['hidden_mae']:.6f} "
                f"| Val hidden MSE: "
                f"{val_stats['hidden_mse']:.6f} "
                f"| Val hidden MAE: "
                f"{val_stats['hidden_mae']:.6f}"
            )

            print(
                f"    Val time: "
                f"{val_time / 60:.1f} min "
                f"| Epoch total: "
                f"{epoch_time / 60:.1f} min"
            )

            print(
                f"    Best val loss: "
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
    print("MAMBA TRAINING FINISHED")
    print("=" * 76)
    print(
        "Total time:",
        f"{total_hours:.2f} h",
    )
    print(
        "Best validation loss:",
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
