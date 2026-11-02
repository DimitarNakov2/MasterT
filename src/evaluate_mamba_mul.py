#!/usr/bin/env python3
"""
Evaluate the multivariable Mamba-3 chlorophyll-a reconstruction model.

This evaluator is designed for checkpoints produced by train_mamba_multivar.py.

It:
- reconstructs the model architecture from checkpoint["args"]
- automatically restores aux_mode and sequence length
- builds the test split with the exact same deterministic artificial masks
- evaluates ONLY artificially hidden, genuinely valid CHL pixels
- converts normalized log-CHL back to physical mg/m^3
- reports global RMSE, MAE, Pearson correlation
- reports mean/median patch SSIM using completed reconstructions
- writes metrics.json and per_sample_metrics.csv

Expected project layout
-----------------------
src/
    chl_data.py
    evaluate_mamba_mul.py
    models/
        mamba_chl.py

Important
---------
The multivariable loader should be installed as src/chl_data.py and the
multivariable Mamba model as src/models/mamba_chl.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from skimage.metrics import structural_similarity
except Exception as exc:
    raise ImportError(
        "evaluate_mamba_mul.py requires scikit-image for SSIM. "
        "Install it in the active environment, e.g. `pip install scikit-image` "
        "or `conda install scikit-image`."
    ) from exc


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
# Helpers
# ---------------------------------------------------------------------

def make_loader(
    dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
):
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
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


def inverse_transform(
    normalized: torch.Tensor,
    mean: float,
    std: float,
) -> torch.Tensor:
    """
    normalized log-CHL -> CHL in mg/m^3.
    """
    log_chl = normalized * std + mean
    return torch.exp(log_chl)


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


def calculate_completed_ssim(
    prediction: np.ndarray,
    target: np.ndarray,
    loss_mask: np.ndarray,
    target_valid_mask: np.ndarray,
) -> float:
    """
    SSIM procedure consistent with the existing benchmark logic:

    - start from the real target image
    - replace ONLY artificially hidden pixels with the model prediction
    - set genuinely invalid/non-target pixels to zero in both images
    - compute SSIM over the completed patch

    Arrays must be HxW in physical CHL units.
    """
    pred = np.asarray(prediction, dtype=np.float64)
    true = np.asarray(target, dtype=np.float64)
    hidden = np.asarray(loss_mask, dtype=bool)
    valid = np.asarray(target_valid_mask, dtype=bool)

    if not valid.any():
        return float("nan")

    completed = true.copy()
    completed[hidden] = pred[hidden]

    completed[~valid] = 0.0
    reference = true.copy()
    reference[~valid] = 0.0

    valid_values = reference[valid]
    data_range = float(valid_values.max() - valid_values.min())

    if not np.isfinite(data_range) or data_range <= 0:
        return float("nan")

    return float(
        structural_similarity(
            reference,
            completed,
            data_range=data_range,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
        )
    )


def checkpoint_arg(
    checkpoint_args: dict,
    name: str,
    default,
):
    value = checkpoint_args.get(name, default)
    return value


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Evaluate multivariable Mamba-3 "
            "for CHL-a reconstruction"
        )
    )

    p.add_argument(
        "--processed-dir",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for metrics.json and per_sample_metrics.csv. "
            "Default: <checkpoint parent>/evaluation"
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

    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Override checkpoint seed. Default uses checkpoint seed "
            "(normally 42), preserving deterministic test masks."
        ),
    )

    p.add_argument(
        "--aux-mode",
        type=str,
        default=None,
        choices=[
            "none",
            "sst",
            "bbp443",
            "sst_bbp443",
        ],
        help=(
            "Normally omit this: aux_mode is read from the checkpoint. "
            "If supplied, it must match the checkpoint."
        ),
    )

    p.add_argument(
        "--sequence-length",
        type=int,
        default=None,
        help=(
            "Normally omit this: sequence length is read from the checkpoint. "
            "If supplied, it must match the checkpoint."
        ),
    )

    p.add_argument(
        "--amp",
        action="store_true",
        help=(
            "Use FP16 autocast during evaluation. "
            "Default is FP32 for maximum numerical consistency."
        ),
    )

    p.add_argument(
        "--physical-clamp",
        action="store_true",
        help=(
            "Clamp physical predictions to preprocessing chl_min/chl_max. "
            "Leave OFF for the closest comparison with deterministic U-Net "
            "evaluation unless you intentionally use clamping for all models."
        ),
    )

    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=(
            "Optional debugging limit. Omit for the full test set."
        ),
    )

    p.add_argument(
        "--log-every",
        type=int,
        default=100,
    )

    return p.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

@torch.no_grad()
def main():
    args = parse_args()

    seed_everything(42)

    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if device.type != "cuda":
        raise RuntimeError(
            "Mamba-3 evaluation requires a CUDA GPU with the installed "
            "Mamba/Triton backend. Run this on an H100 GPU node."
        )

    print("=" * 78)
    print("MAMBA-3 MULTIVARIABLE CHL-A EVALUATION")
    print("=" * 78)
    print("Device:", device)
    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )
    print(
        "CUDA memory:",
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB",
    )
    print()

    # -----------------------------------------------------------------
    # Load checkpoint first: architecture/data mode comes from training.
    # -----------------------------------------------------------------

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
	weights_only=False,
    )

    checkpoint_version = checkpoint.get(
        "training_version",
        "unknown",
    )

    allowed_versions = {
        "mamba3_chla_multivar_v2",
        # Backward compatibility for old CHL-only Mamba checkpoint.
        "mamba3_chla_bidirectional_v1",
    }

    if checkpoint_version not in allowed_versions:
        raise RuntimeError(
            "Unsupported checkpoint training_version: "
            f"{checkpoint_version!r}. Expected one of {allowed_versions}."
        )

    ckpt_args = checkpoint.get(
        "args",
        {},
    )

    checkpoint_aux_mode = str(
        checkpoint_arg(
            ckpt_args,
            "aux_mode",
            "none",
        )
    ).lower()

    checkpoint_sequence_length = int(
        checkpoint_arg(
            ckpt_args,
            "sequence_length",
            15,
        )
    )

    checkpoint_seed = int(
        checkpoint_arg(
            ckpt_args,
            "seed",
            42,
        )
    )

    aux_mode = (
        checkpoint_aux_mode
        if args.aux_mode is None
        else args.aux_mode
    )

    sequence_length = (
        checkpoint_sequence_length
        if args.sequence_length is None
        else int(args.sequence_length)
    )

    seed = (
        checkpoint_seed
        if args.seed is None
        else int(args.seed)
    )

    if aux_mode != checkpoint_aux_mode:
        raise RuntimeError(
            "Requested --aux-mode does not match checkpoint. "
            f"checkpoint={checkpoint_aux_mode!r}, requested={aux_mode!r}"
        )

    if sequence_length != checkpoint_sequence_length:
        raise RuntimeError(
            "Requested --sequence-length does not match checkpoint. "
            f"checkpoint={checkpoint_sequence_length}, "
            f"requested={sequence_length}"
        )

    seed_everything(seed)

    # -----------------------------------------------------------------
    # Preprocessing config / inverse transformation
    # -----------------------------------------------------------------

    config_path = (
        args.processed_dir
        / "preprocessing_config.json"
    )

    with open(config_path, "r") as f:
        config = json.load(f)

    train_mean = float(
        config.get(
            "chl_train_mean",
            config["train_mean"],
        )
    )

    train_std = float(
        config.get(
            "chl_train_std",
            config["train_std"],
        )
    )

    chl_min = float(
        config.get(
            "chl_min",
            0.01,
        )
    )

    chl_max = float(
        config.get(
            "chl_max",
            100.0,
        )
    )

    # -----------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------

    test_dataset = ChlorophyllReconstructionDataset(
        processed_dir=args.processed_dir,
        split="test",
        input_mode="mamba",
        seed=seed,
        mamba_sequence_length=sequence_length,
        aux_mode=aux_mode,
    )

    test_loader = make_loader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
    )

    print("Checkpoint:", args.checkpoint)
    print(
        "Checkpoint epoch:",
        checkpoint.get(
            "epoch",
            "unknown",
        ),
    )
    print(
        "Training version:",
        checkpoint_version,
    )
    print("Seed:", seed)
    print("Auxiliary mode:", aux_mode)
    print(
        "Sequence length:",
        sequence_length,
    )
    print(
        "Test samples:",
        f"{len(test_dataset):,}",
    )
    print(
        "CHL normalization:",
        f"mean={train_mean:.6f}, std={train_std:.6f}",
    )
    print(
        "Physical clamp:",
        bool(args.physical_clamp),
    )
    print()

    # -----------------------------------------------------------------
    # Recreate exact architecture
    # -----------------------------------------------------------------

    encoder_channels = tuple(
        int(v)
        for v in checkpoint_arg(
            ckpt_args,
            "encoder_channels",
            [64, 128, 256],
        )
    )

    model = ChlorophyllMamba(
        aux_mode=aux_mode,
        encoder_channels=encoder_channels,
        d_model=int(
            checkpoint_arg(
                ckpt_args,
                "d_model",
                256,
            )
        ),
        d_state=int(
            checkpoint_arg(
                ckpt_args,
                "d_state",
                64,
            )
        ),
        num_mamba_layers=int(
            checkpoint_arg(
                ckpt_args,
                "num_mamba_layers",
                4,
            )
        ),
        headdim=int(
            checkpoint_arg(
                ckpt_args,
                "headdim",
                64,
            )
        ),
        expand=int(
            checkpoint_arg(
                ckpt_args,
                "expand",
                2,
            )
        ),
        chunk_size=int(
            checkpoint_arg(
                ckpt_args,
                "chunk_size",
                64,
            )
        ),
        bidirectional=not bool(
            checkpoint_arg(
                ckpt_args,
                "causal",
                False,
            )
        ),
        dropout=float(
            checkpoint_arg(
                ckpt_args,
                "dropout",
                0.0,
            )
        ),
        group_norm_groups=int(
            checkpoint_arg(
                ckpt_args,
                "group_norm_groups",
                8,
            )
        ),
        use_temporal_metadata=not bool(
            checkpoint_arg(
                ckpt_args,
                "no_temporal_metadata",
                False,
            )
        ),
        center_index=sequence_length // 2,
        predict_residual=bool(
            checkpoint_arg(
                ckpt_args,
                "predict_residual",
                False,
            )
        ),
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model = model.to(device)
    model.eval()

    print(
        "Mamba channels/day:",
        model.in_channels,
    )
    if hasattr(model, "channel_names"):
        print(
            "Channel order:",
            tuple(model.channel_names),
        )

    params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "Model parameters:",
        f"{params:,}",
    )
    print()

    # -----------------------------------------------------------------
    # Preflight
    # -----------------------------------------------------------------

    first_batch = next(iter(test_loader))

    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=args.amp,
    ):
        first_prediction = model_forward(
            model=model,
            batch=first_batch,
            device=device,
        )

    expected_target_shape = tuple(
        first_batch["target"].shape
    )

    if (
        tuple(first_prediction.shape)
        != expected_target_shape
    ):
        raise RuntimeError(
            "Preflight output shape mismatch: "
            f"prediction={tuple(first_prediction.shape)}, "
            f"target={expected_target_shape}"
        )

    if not torch.isfinite(
        first_prediction
    ).all():
        raise RuntimeError(
            "Preflight produced non-finite predictions."
        )

    print(
        "Preflight sequence:",
        tuple(
            first_batch[
                "mamba_sequence"
            ].shape
        ),
    )
    print(
        "Preflight output:",
        tuple(first_prediction.shape),
        "finite=True",
    )
    print()

    del first_batch
    del first_prediction
    torch.cuda.empty_cache()

    # -----------------------------------------------------------------
    # Evaluation accumulators
    # -----------------------------------------------------------------

    squared_error_sum = 0.0
    absolute_error_sum = 0.0
    hidden_pixel_count = 0

    # Streaming Pearson correlation accumulators.
    sum_pred = 0.0
    sum_target = 0.0
    sum_pred_sq = 0.0
    sum_target_sq = 0.0
    sum_cross = 0.0

    ssim_values = []
    per_sample_rows = []

    samples_evaluated = 0
    start_time = time.time()

    # -----------------------------------------------------------------
    # Evaluation loop
    # -----------------------------------------------------------------

    for batch_idx, batch in enumerate(
        test_loader,
        start=1,
    ):
        if (
            args.max_samples is not None
            and samples_evaluated >= args.max_samples
        ):
            break

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=args.amp,
        ):
            prediction_norm = model_forward(
                model=model,
                batch=batch,
                device=device,
            )

        target_norm = to_device(
            batch,
            "target",
            device,
        )

        loss_mask = to_device(
            batch,
            "loss_mask",
            device,
        ).bool()

        target_valid = to_device(
            batch,
            "target_valid_mask",
            device,
        ).bool()

        prediction = inverse_transform(
            prediction_norm.float(),
            train_mean,
            train_std,
        )

        target = inverse_transform(
            target_norm.float(),
            train_mean,
            train_std,
        )

        if args.physical_clamp:
            prediction = prediction.clamp(
                min=chl_min,
                max=chl_max,
            )

        if not torch.isfinite(
            prediction
        ).all():
            raise RuntimeError(
                f"Non-finite physical prediction at batch {batch_idx}."
            )

        B = prediction.shape[0]

        for b in range(B):
            if (
                args.max_samples is not None
                and samples_evaluated >= args.max_samples
            ):
                break

            pred_b = prediction[b, 0]
            target_b = target[b, 0]
            hidden_b = loss_mask[b, 0]
            valid_b = target_valid[b, 0]

            n_hidden = int(
                hidden_b.sum().item()
            )

            if n_hidden <= 0:
                continue

            p = pred_b[hidden_b].double()
            t = target_b[hidden_b].double()

            err = p - t

            sample_sse = float(
                torch.sum(
                    err * err
                ).item()
            )

            sample_ae = float(
                torch.sum(
                    torch.abs(err)
                ).item()
            )

            squared_error_sum += sample_sse
            absolute_error_sum += sample_ae
            hidden_pixel_count += n_hidden

            sum_pred += float(
                p.sum().item()
            )
            sum_target += float(
                t.sum().item()
            )
            sum_pred_sq += float(
                (p * p).sum().item()
            )
            sum_target_sq += float(
                (t * t).sum().item()
            )
            sum_cross += float(
                (p * t).sum().item()
            )

            sample_rmse = math.sqrt(
                sample_sse / n_hidden
            )

            sample_mae = (
                sample_ae / n_hidden
            )

            pred_np = (
                pred_b.detach()
                .cpu()
                .numpy()
            )

            target_np = (
                target_b.detach()
                .cpu()
                .numpy()
            )

            hidden_np = (
                hidden_b.detach()
                .cpu()
                .numpy()
            )

            valid_np = (
                valid_b.detach()
                .cpu()
                .numpy()
            )

            sample_ssim = (
                calculate_completed_ssim(
                    prediction=pred_np,
                    target=target_np,
                    loss_mask=hidden_np,
                    target_valid_mask=valid_np,
                )
            )

            if np.isfinite(sample_ssim):
                ssim_values.append(
                    sample_ssim
                )

            row = {
                "sample_number": (
                    samples_evaluated
                ),
                "hidden_pixels": (
                    n_hidden
                ),
                "rmse_mg_m3": (
                    sample_rmse
                ),
                "mae_mg_m3": (
                    sample_mae
                ),
                "ssim": (
                    sample_ssim
                ),
            }

            # Preserve useful metadata if available.
            for key in (
                "time_idx",
                "target_time_idx",
                "patch_id",
                "y",
                "x",
            ):
                if key in batch:
                    value = batch[key]
                    try:
                        if torch.is_tensor(value):
                            value = value[b].item()
                        elif isinstance(
                            value,
                            (list, tuple),
                        ):
                            value = value[b]
                        row[key] = value
                    except Exception:
                        pass

            per_sample_rows.append(
                row
            )

            samples_evaluated += 1

        if (
            batch_idx % args.log_every == 0
            or batch_idx == len(test_loader)
        ):
            elapsed = time.time() - start_time

            current_rmse = math.sqrt(
                squared_error_sum
                / max(
                    hidden_pixel_count,
                    1,
                )
            )

            current_mae = (
                absolute_error_sum
                / max(
                    hidden_pixel_count,
                    1,
                )
            )

            print(
                f"Batch {batch_idx:,}/{len(test_loader):,} "
                f"| samples={samples_evaluated:,} "
                f"| hidden={hidden_pixel_count:,} "
                f"| RMSE={current_rmse:.6f} "
                f"| MAE={current_mae:.6f} "
                f"| elapsed={elapsed / 3600:.2f} h",
                flush=True,
            )

    # -----------------------------------------------------------------
    # Final global metrics
    # -----------------------------------------------------------------

    if hidden_pixel_count <= 0:
        raise RuntimeError(
            "No hidden pixels were evaluated."
        )

    rmse = math.sqrt(
        squared_error_sum
        / hidden_pixel_count
    )

    mae = (
        absolute_error_sum
        / hidden_pixel_count
    )

    n = float(hidden_pixel_count)

    numerator = (
        n * sum_cross
        - sum_pred * sum_target
    )

    denom_pred = (
        n * sum_pred_sq
        - sum_pred * sum_pred
    )

    denom_target = (
        n * sum_target_sq
        - sum_target * sum_target
    )

    denominator = math.sqrt(
        max(
            denom_pred,
            0.0,
        )
        * max(
            denom_target,
            0.0,
        )
    )

    if denominator > 0:
        correlation = (
            numerator / denominator
        )
    else:
        correlation = float("nan")

    mean_ssim = (
        float(np.mean(ssim_values))
        if ssim_values
        else float("nan")
    )

    median_ssim = (
        float(np.median(ssim_values))
        if ssim_values
        else float("nan")
    )

    elapsed_hours = (
        time.time() - start_time
    ) / 3600.0

    # -----------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else (
            args.checkpoint.parent
            / "evaluation"
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics = {
        "checkpoint": str(
            args.checkpoint
        ),
        "checkpoint_epoch": checkpoint.get(
            "epoch",
            None,
        ),
        "training_version": checkpoint_version,
        "model": "Mamba3",
        "split": "test",
        "aux_mode": aux_mode,
        "sequence_length": sequence_length,
        "channels_per_day": int(
            model.in_channels
        ),
        "samples_evaluated": int(
            samples_evaluated
        ),
        "test_samples_total": int(
            len(test_dataset)
        ),
        "hidden_pixels": int(
            hidden_pixel_count
        ),
        "rmse_mg_m3": float(
            rmse
        ),
        "mae_mg_m3": float(
            mae
        ),
        "pearson_correlation": float(
            correlation
        ),
        "mean_ssim": float(
            mean_ssim
        ),
        "median_ssim": float(
            median_ssim
        ),
        "ssim_samples": int(
            len(ssim_values)
        ),
        "seed": int(
            seed
        ),
        "amp": bool(
            args.amp
        ),
        "physical_clamp_enabled": bool(
            args.physical_clamp
        ),
        "chl_min": float(
            chl_min
        ),
        "chl_max": float(
            chl_max
        ),
        "train_mean": float(
            train_mean
        ),
        "train_std": float(
            train_std
        ),
        "evaluation_hours": float(
            elapsed_hours
        ),
        "fair_comparison_match": True,
    }

    metrics_path = (
        output_dir
        / "metrics.json"
    )

    with open(
        metrics_path,
        "w",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2,
        )

    csv_path = (
        output_dir
        / "per_sample_metrics.csv"
    )

    fieldnames = []
    for row in per_sample_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with open(
        csv_path,
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(
            per_sample_rows
        )

    # -----------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------

    print()
    print("=" * 78)
    print("MAMBA-3 TEST RESULTS")
    print("=" * 78)
    print(
        "Checkpoint epoch:",
        checkpoint.get(
            "epoch",
            "unknown",
        ),
    )
    print(
        "Aux mode:",
        aux_mode,
    )
    print(
        "Sequence length:",
        sequence_length,
    )
    print(
        "Samples:",
        f"{samples_evaluated:,}",
    )
    print(
        "Hidden pixels:",
        f"{hidden_pixel_count:,}",
    )
    print(
        f"RMSE:        {rmse:.9f} mg/m^3"
    )
    print(
        f"MAE:         {mae:.9f} mg/m^3"
    )
    print(
        f"Correlation: {correlation:.9f}"
    )
    print(
        f"Mean SSIM:   {mean_ssim:.9f}"
    )
    print(
        f"Median SSIM: {median_ssim:.9f}"
    )
    print(
        f"Elapsed:     {elapsed_hours:.3f} h"
    )
    print()
    print(
        "Saved:",
        metrics_path,
    )
    print(
        "Saved:",
        csv_path,
    )

    test_dataset.close()


if __name__ == "__main__":
    main()
