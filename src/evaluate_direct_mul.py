'''
Evaluate the multivariable chlorophyll-a DIRECT flow-matching reconstruction model.

This evaluator is matched to the multivariable versions of:
    src/chl_data.py
    src/models/direct.py
    src/train_direct.py

It supports checkpoints trained with:
    aux_mode = none
    aux_mode = sst
    aux_mode = bbp443
    aux_mode = sst_bbp443

Evaluation protocol
-------------------
- split = test
- input_mode = "direct"
- same deterministic artificial masks as the other benchmark models
- reverse flow sampling starts from Gaussian noise
- FUSE is applied at every reverse step
- default reverse steps Nk = 50
- default ensemble size N = 16
- ensemble mean is the final prediction
- metrics are computed ONLY on artificially hidden valid target pixels
- RMSE / MAE / Pearson correlation are computed in physical Chl-a units
- SSIM uses the same "completed reconstruction" protocol used for the
  U-Net/STIMP/L4 comparisons:
      * start from the ground-truth target
      * replace only artificially hidden pixels by the DIRECT prediction
      * invalid pixels are zeroed in both arrays
      * structural_similarity(..., gaussian_weights=True, sigma=1.5,
        use_sample_covariance=False)

DIRECT flow convention
----------------------
Training uses:
    x_k = (1-k) * x + k * epsilon
and the network predicts:
    u = x - epsilon

Therefore inference starts at k=1 (noise) and integrates toward k=0 with:
    x <- x + delta_k * v_theta(x, k)
where delta_k = 1 / Nk.

The current field is FUSE-rectified before every network evaluation and after
every update so visible centre observations stay fixed and land remains zero.

Notes
-----
The DIRECT thesis uses N=16 independent noise initializations and takes their
ensemble mean. Keep --ensemble-size 16 and --steps 50 for the final benchmark.
For a quick smoke test, use for example:
    --ensemble-size 1 --max-samples 10
'''

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
except ImportError as exc:
    raise ImportError(
        "evaluate_direct.py requires scikit-image. "
        "Install it with: pip install scikit-image"
    ) from exc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
# Helpers
# ---------------------------------------------------------------------

def make_loader(
    dataset,
    num_workers: int,
    prefetch_factor: int,
):
    kwargs = dict(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
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


def scalar_from_batch(
    value,
):
    if torch.is_tensor(value):
        return value.reshape(-1)[0].item()

    if isinstance(value, (list, tuple)):
        return value[0]

    return value


def inverse_normalized_log_chl(
    normalized: np.ndarray,
    train_mean: float,
    train_std: float,
) -> np.ndarray:
    log_chl = (
        normalized.astype(
            np.float64
        )
        * float(train_std)
        + float(train_mean)
    )

    return np.exp(log_chl)


def normalize_chl_bound(
    chl_value: float,
    train_mean: float,
    train_std: float,
) -> float:
    return (
        math.log(float(chl_value))
        - float(train_mean)
    ) / float(train_std)


def masked_sample_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
):
    values_p = prediction[mask]
    values_t = target[mask]

    if values_p.size == 0:
        return (
            math.nan,
            math.nan,
            math.nan,
        )

    diff = values_p - values_t

    rmse = float(
        np.sqrt(
            np.mean(diff ** 2)
        )
    )

    mae = float(
        np.mean(
            np.abs(diff)
        )
    )

    if (
        values_p.size >= 2
        and np.std(values_p) > 0
        and np.std(values_t) > 0
    ):
        corr = float(
            np.corrcoef(
                values_p,
                values_t,
            )[0, 1]
        )
    else:
        corr = math.nan

    return rmse, mae, corr


def completed_ssim(
    prediction: np.ndarray,
    target: np.ndarray,
    loss_mask: np.ndarray,
    target_valid: np.ndarray,
) -> float:
    """
    Same completed-image SSIM protocol used by the earlier benchmark scripts.
    """
    completed_prediction = (
        target.copy()
    )

    completed_prediction[
        loss_mask
    ] = prediction[
        loss_mask
    ]

    completed_target = (
        target.copy()
    )

    completed_prediction[
        ~target_valid
    ] = 0.0

    completed_target[
        ~target_valid
    ] = 0.0

    valid_values = completed_target[
        target_valid
    ]

    if valid_values.size == 0:
        return math.nan

    data_min = float(
        np.min(valid_values)
    )
    data_max = float(
        np.max(valid_values)
    )

    data_range = (
        data_max - data_min
    )

    if (
        not np.isfinite(data_range)
        or data_range <= 0
    ):
        return math.nan

    return float(
        structural_similarity(
            completed_target,
            completed_prediction,
            data_range=data_range,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
        )
    )



def build_auxiliary_context(
    batch: dict,
    aux_mode: str,
    device: torch.device,
) -> torch.Tensor | None:
    """
    Build the extra DIRECT conditioning channels in the same order used
    during multivariable training.

    none:
        None
    sst:
        [SST, SST-valid-mask]
    bbp443:
        [BBP443, BBP443-valid-mask]
    sst_bbp443:
        [SST, SST-valid-mask, BBP443, BBP443-valid-mask]
    """
    mode = str(aux_mode).lower()

    if mode == "none":
        return None

    channels = []

    if mode in {"sst", "sst_bbp443"}:
        channels.extend(
            [
                to_device(batch, "sst", device),
                to_device(batch, "sst_valid_mask", device),
            ]
        )

    if mode in {"bbp443", "sst_bbp443"}:
        channels.extend(
            [
                to_device(batch, "bbp443", device),
                to_device(batch, "bbp443_valid_mask", device),
            ]
        )

    if not channels:
        raise ValueError(f"Unsupported aux_mode: {aux_mode!r}")

    return torch.cat(channels, dim=1)


# ---------------------------------------------------------------------
# Streaming global Pearson correlation
# ---------------------------------------------------------------------

class StreamingMetrics:
    def __init__(self):
        self.n = 0
        self.sum_p = 0.0
        self.sum_t = 0.0
        self.sum_pp = 0.0
        self.sum_tt = 0.0
        self.sum_pt = 0.0
        self.sum_sq_error = 0.0
        self.sum_abs_error = 0.0

    def update(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        mask: np.ndarray,
    ):
        p = prediction[
            mask
        ].astype(
            np.float64,
            copy=False,
        )

        t = target[
            mask
        ].astype(
            np.float64,
            copy=False,
        )

        if p.size == 0:
            return

        diff = p - t

        self.n += int(
            p.size
        )

        self.sum_p += float(
            p.sum()
        )
        self.sum_t += float(
            t.sum()
        )
        self.sum_pp += float(
            np.dot(p, p)
        )
        self.sum_tt += float(
            np.dot(t, t)
        )
        self.sum_pt += float(
            np.dot(p, t)
        )

        self.sum_sq_error += float(
            np.dot(
                diff,
                diff,
            )
        )

        self.sum_abs_error += float(
            np.abs(diff).sum()
        )

    def finalize(self):
        if self.n == 0:
            return {
                "hidden_pixels": 0,
                "rmse": math.nan,
                "mae": math.nan,
                "pearson_correlation": (
                    math.nan
                ),
            }

        n = float(
            self.n
        )

        rmse = math.sqrt(
            self.sum_sq_error
            / n
        )

        mae = (
            self.sum_abs_error
            / n
        )

        cov_num = (
            self.sum_pt
            - (
                self.sum_p
                * self.sum_t
                / n
            )
        )

        var_p = (
            self.sum_pp
            - (
                self.sum_p
                * self.sum_p
                / n
            )
        )

        var_t = (
            self.sum_tt
            - (
                self.sum_t
                * self.sum_t
                / n
            )
        )

        denom = math.sqrt(
            max(
                var_p,
                0.0,
            )
            * max(
                var_t,
                0.0,
            )
        )

        corr = (
            cov_num / denom
            if denom > 0
            else math.nan
        )

        return {
            "hidden_pixels": (
                int(
                    self.n
                )
            ),
            "rmse": float(
                rmse
            ),
            "mae": float(
                mae
            ),
            "pearson_correlation": (
                float(corr)
            ),
        }


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------

def load_direct_model(
    checkpoint_path: Path,
    device: torch.device,
    requested_aux_mode: str | None = None,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    training_version = checkpoint.get(
        "training_version"
    )

    allowed_versions = {
        "direct_chla_flow_matching_v1",
        "direct_chla_flow_matching_multivar_v2",
    }

    if training_version not in allowed_versions:
        raise RuntimeError(
            "Unsupported DIRECT checkpoint training_version: "
            f"{training_version!r}. Expected one of {allowed_versions}."
        )

    ckpt_args = checkpoint.get(
        "args",
        {},
    )

    checkpoint_aux_mode = str(
        ckpt_args.get(
            "aux_mode",
            "none",
        )
    ).lower()

    # Original v1 checkpoints were CHL-only.
    if training_version == "direct_chla_flow_matching_v1":
        checkpoint_aux_mode = "none"

    aux_mode = (
        checkpoint_aux_mode
        if requested_aux_mode is None
        else str(requested_aux_mode).lower()
    )

    if aux_mode != checkpoint_aux_mode:
        raise RuntimeError(
            "Requested auxiliary mode does not match checkpoint. "
            f"checkpoint={checkpoint_aux_mode!r}, requested={aux_mode!r}"
        )

    conditioning_dim = int(
        ckpt_args.get(
            "conditioning_dim",
            512,
        )
    )

    k_embedding_dim = int(
        ckpt_args.get(
            "k_embedding_dim",
            256,
        )
    )

    attention_heads = int(
        ckpt_args.get(
            "attention_heads",
            8,
        )
    )

    dropout = float(
        ckpt_args.get(
            "dropout",
            0.0,
        )
    )

    group_norm_groups = int(
        ckpt_args.get(
            "group_norm_groups",
            32,
        )
    )

    aux_channels = auxiliary_channels_for_mode(
        aux_mode
    )

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
        conditioning_dim=conditioning_dim,
        k_embedding_dim=k_embedding_dim,
        attention_heads=attention_heads,
        dropout=dropout,
        group_norm_groups=group_norm_groups,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.to(device)
    model.eval()

    return (
        model,
        checkpoint,
        ckpt_args,
        aux_mode,
    )


# ---------------------------------------------------------------------
# Reverse DIRECT sampler
# ---------------------------------------------------------------------

@torch.no_grad()
def sample_direct_once(
    model: torch.nn.Module,
    current_masked: torch.Tensor,
    visible_mask: torch.Tensor,
    ocean_mask: torch.Tensor,
    temporal_context: torch.Tensor,
    temporal_masks: torch.Tensor,
    doy: torch.Tensor,
    auxiliary_context: torch.Tensor | None,
    steps: int,
    generator: torch.Generator,
    use_amp: bool,
    clamp_min_norm: float | None,
    clamp_max_norm: float | None,
) -> torch.Tensor:
    """
    One stochastic DIRECT reconstruction.

    Starts from N(0,I) at k=1 and Euler-integrates the learned reverse
    flow to k=0. FUSE is enforced at every step.
    """
    if steps < 1:
        raise ValueError(
            "steps must be >= 1"
        )

    x = torch.randn(
        current_masked.shape,
        generator=generator,
        device=current_masked.device,
        dtype=torch.float32,
    )

    x = fuse_rectify(
        noisy_center=x,
        observed_center=(
            current_masked
        ),
        visible_mask=(
            visible_mask
        ),
        ocean_mask=(
            ocean_mask
        ),
    )

    delta = (
        1.0 / float(steps)
    )

    batch_size = (
        current_masked.shape[0]
    )

    for step_idx in range(
        steps
    ):
        # Midpoint k is a slightly better numerical approximation than
        # evaluating exactly at the left/right end of each Euler interval.
        k_value = (
            1.0
            - (
                step_idx
                + 0.5
            )
            * delta
        )

        k = torch.full(
            (batch_size,),
            fill_value=k_value,
            device=x.device,
            dtype=torch.float32,
        )

        x = fuse_rectify(
            noisy_center=x,
            observed_center=(
                current_masked
            ),
            visible_mask=(
                visible_mask
            ),
            ocean_mask=(
                ocean_mask
            ),
        )

        with torch.autocast(
            device_type=x.device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            velocity = model(
                noisy_center=x,
                temporal_context=(
                    temporal_context
                ),
                temporal_masks=(
                    temporal_masks
                ),
                doy=doy,
                k=k,
                auxiliary_context=auxiliary_context,
            )

        if not torch.isfinite(
            velocity
        ).all():
            raise RuntimeError(
                "DIRECT produced a non-finite velocity "
                f"at reverse step {step_idx + 1}/{steps}."
            )

        x = (
            x
            + delta
            * velocity.float()
        )

        if (
            clamp_min_norm
            is not None
            and clamp_max_norm
            is not None
        ):
            # Clamp only the generated field. FUSE immediately restores exact
            # visible centre observations afterwards.
            x = torch.clamp(
                x,
                min=float(
                    clamp_min_norm
                ),
                max=float(
                    clamp_max_norm
                ),
            )

        x = fuse_rectify(
            noisy_center=x,
            observed_center=(
                current_masked
            ),
            visible_mask=(
                visible_mask
            ),
            ocean_mask=(
                ocean_mask
            ),
        )

    return x


@torch.no_grad()
def sample_direct_ensemble(
    model: torch.nn.Module,
    current_masked: torch.Tensor,
    visible_mask: torch.Tensor,
    ocean_mask: torch.Tensor,
    temporal_context: torch.Tensor,
    temporal_masks: torch.Tensor,
    doy: torch.Tensor,
    auxiliary_context: torch.Tensor | None,
    steps: int,
    ensemble_size: int,
    base_seed: int,
    sample_index: int,
    use_amp: bool,
    clamp_min_norm: float | None,
    clamp_max_norm: float | None,
):
    """
    Return ensemble mean and normalized-space ensemble standard deviation.

    Ensembles are sampled sequentially to keep GPU memory use low for the
    thesis-sized DIRECT U-Net.
    """
    if ensemble_size < 1:
        raise ValueError(
            "ensemble_size must be >= 1"
        )

    running_sum = None
    running_sum_sq = None

    for member_idx in range(
        ensemble_size
    ):
        seed = (
            int(base_seed)
            + 1_000_003
            * int(sample_index)
            + 10_007
            * int(member_idx)
        )

        generator = (
            torch.Generator(
                device=(
                    current_masked.device
                )
            )
        )

        generator.manual_seed(
            seed
        )

        reconstruction = (
            sample_direct_once(
                model=model,
                current_masked=(
                    current_masked
                ),
                visible_mask=(
                    visible_mask
                ),
                ocean_mask=(
                    ocean_mask
                ),
                temporal_context=(
                    temporal_context
                ),
                temporal_masks=(
                    temporal_masks
                ),
                doy=doy,
                auxiliary_context=auxiliary_context,
                steps=steps,
                generator=generator,
                use_amp=use_amp,
                clamp_min_norm=(
                    clamp_min_norm
                ),
                clamp_max_norm=(
                    clamp_max_norm
                ),
            )
        )

        r = reconstruction.float()

        if running_sum is None:
            running_sum = (
                r.clone()
            )
            running_sum_sq = (
                r * r
            )
        else:
            running_sum += r
            running_sum_sq += (
                r * r
            )

    mean = (
        running_sum
        / float(
            ensemble_size
        )
    )

    variance = (
        running_sum_sq
        / float(
            ensemble_size
        )
        - mean
        * mean
    ).clamp_min(0.0)

    std = torch.sqrt(
        variance
    )

    return mean, std


# ---------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------

def save_qualitative_figure(
    path: Path,
    current_masked_norm: np.ndarray,
    visible_mask: np.ndarray,
    prediction_physical: np.ndarray,
    target_physical: np.ndarray,
    loss_mask: np.ndarray,
    target_valid: np.ndarray,
    train_mean: float,
    train_std: float,
    date: str,
    sample_index: int,
):
    current_physical = (
        inverse_normalized_log_chl(
            current_masked_norm,
            train_mean,
            train_std,
        )
    )

    observed_display = (
        current_physical.copy()
    )

    observed_display[
        ~visible_mask
    ] = np.nan

    target_display = (
        target_physical.copy()
    )
    target_display[
        ~target_valid
    ] = np.nan

    prediction_completed = (
        target_physical.copy()
    )
    prediction_completed[
        loss_mask
    ] = prediction_physical[
        loss_mask
    ]
    prediction_completed[
        ~target_valid
    ] = np.nan

    error = np.full_like(
        target_physical,
        np.nan,
        dtype=np.float64,
    )
    error[
        loss_mask
    ] = np.abs(
        prediction_physical[
            loss_mask
        ]
        - target_physical[
            loss_mask
        ]
    )

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(18, 4.5),
        constrained_layout=True,
    )

    im0 = axes[0].imshow(
        observed_display
    )
    axes[0].set_title(
        "Masked input"
    )
    fig.colorbar(
        im0,
        ax=axes[0],
        fraction=0.046,
    )

    im1 = axes[1].imshow(
        prediction_completed
    )
    axes[1].set_title(
        "DIRECT reconstruction"
    )
    fig.colorbar(
        im1,
        ax=axes[1],
        fraction=0.046,
    )

    im2 = axes[2].imshow(
        target_display
    )
    axes[2].set_title(
        "Ground truth"
    )
    fig.colorbar(
        im2,
        ax=axes[2],
        fraction=0.046,
    )

    im3 = axes[3].imshow(
        error
    )
    axes[3].set_title(
        "Absolute error (hidden)"
    )
    fig.colorbar(
        im3,
        ax=axes[3],
        fraction=0.046,
    )

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"DIRECT | sample={sample_index} | {date}"
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        path,
        dpi=180,
    )
    plt.close(fig)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Evaluate DIRECT Chl-a reconstruction "
            "on the deterministic test split."
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
        help=(
            "Usually models/direct/best.pt"
        ),
    )

    p.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_DIR
            / "results"
            / "direct"
        ),
    )

    p.add_argument(
        "--steps",
        type=int,
        default=50,
        help=(
            "Reverse flow steps Nk. "
            "Final benchmark default: 50."
        ),
    )

    p.add_argument(
        "--ensemble-size",
        type=int,
        default=16,
        help=(
            "Independent Gaussian initializations. "
            "Final DIRECT benchmark default: 16."
        ),
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
            "Override checkpoint seed. Default uses the checkpoint seed, "
            "normally 42, preserving the deterministic benchmark masks."
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
            "Normally omit this. The evaluator reads aux_mode from the "
            "checkpoint. If provided, it must match the checkpoint."
        ),
    )

    p.add_argument(
        "--amp",
        action="store_true",
        help=(
            "Use FP16 autocast for model forward passes. "
            "Leave disabled if numerical stability is uncertain."
        ),
    )

    p.add_argument(
        "--no-clamp",
        action="store_true",
        help=(
            "Disable normalized-space clamping to preprocessing "
            "chl_min/chl_max."
        ),
    )

    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=(
            "Optional smoke-test limit. "
            "Omit for the full test set."
        ),
    )

    p.add_argument(
        "--num-figures",
        type=int,
        default=6,
    )

    p.add_argument(
        "--log-every",
        type=int,
        default=10,
    )

    return p.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    args = parse_args()

    seed_everything(42)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if device.type != "cuda":
        raise RuntimeError(
            "DIRECT evaluation should be run on a CUDA GPU."
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure_dir = (
        args.output_dir
        / "figures"
    )

    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 76)
    print("DIRECT CHL-A EVALUATION")
    print("=" * 76)
    print("Device:", device)
    print(
        "GPU:",
        torch.cuda.get_device_name(
            0
        ),
    )
    print(
        "Checkpoint:",
        args.checkpoint,
    )
    print(
        "Reverse steps:",
        args.steps,
    )
    print(
        "Ensemble size:",
        args.ensemble_size,
    )
    print(
        "AMP:",
        args.amp,
    )
    print()

    model, checkpoint, ckpt_args, aux_mode = (
        load_direct_model(
            checkpoint_path=args.checkpoint,
            device=device,
            requested_aux_mode=args.aux_mode,
        )
    )

    checkpoint_seed = int(
        ckpt_args.get(
            "seed",
            42,
        )
    )

    seed = (
        checkpoint_seed
        if args.seed is None
        else int(args.seed)
    )

    seed_everything(seed)

    dataset = (
        ChlorophyllReconstructionDataset(
            processed_dir=args.processed_dir,
            split="test",
            input_mode="direct",
            seed=seed,
            aux_mode=aux_mode,
        )
    )

    loader = make_loader(
        dataset=dataset,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )

    print(
        "Checkpoint epoch:",
        checkpoint.get(
            "epoch"
        ),
    )
    print(
        "Checkpoint best val loss:",
        checkpoint.get(
            "best_val_loss"
        ),
    )
    print(
        "Training version:",
        checkpoint.get(
            "training_version"
        ),
    )
    print(
        "Auxiliary mode:",
        aux_mode,
    )
    print(
        "DIRECT input channels:",
        getattr(model, "input_channels", "unknown"),
    )
    print(
        "Seed:",
        seed,
    )
    print()

    train_mean = float(
        dataset.train_mean
    )

    train_std = float(
        dataset.train_std
    )

    # The dataset already loaded preprocessing_config.json.
    # Use its public attributes when available; otherwise read the config.
    config_path = (
        Path(
            args.processed_dir
        )
        / "preprocessing_config.json"
    )

    with open(
        config_path,
        "r",
    ) as f:
        preprocessing_config = (
            json.load(f)
        )

    chl_min = float(
        preprocessing_config[
            "chl_min"
        ]
    )

    chl_max = float(
        preprocessing_config[
            "chl_max"
        ]
    )

    if args.no_clamp:
        clamp_min_norm = None
        clamp_max_norm = None
    else:
        clamp_min_norm = (
            normalize_chl_bound(
                chl_min,
                train_mean,
                train_std,
            )
        )
        clamp_max_norm = (
            normalize_chl_bound(
                chl_max,
                train_mean,
                train_std,
            )
        )

    n_total = len(
        dataset
    )

    n_eval = (
        min(
            n_total,
            args.max_samples,
        )
        if args.max_samples
        is not None
        else n_total
    )

    if n_eval < 1:
        raise ValueError(
            "No samples selected for evaluation."
        )

    if args.num_figures > 0:
        figure_indices = set(
            np.linspace(
                0,
                n_eval - 1,
                num=min(
                    args.num_figures,
                    n_eval,
                ),
                dtype=int,
            ).tolist()
        )
    else:
        figure_indices = set()

    print(
        "Test samples available:",
        f"{n_total:,}",
    )
    print(
        "Samples to evaluate:",
        f"{n_eval:,}",
    )
    print(
        "Normalization:",
        f"mean={train_mean:.6f}, "
        f"std={train_std:.6f}",
    )
    print(
        "Physical Chl-a clamp:",
        (
            "disabled"
            if args.no_clamp
            else (
                f"[{chl_min}, {chl_max}]"
            )
        ),
    )
    print()

    global_metrics = (
        StreamingMetrics()
    )

    ssim_values = []
    per_sample_rows = []

    start_time = (
        time.time()
    )

    expected_hidden_pixels = 0

    try:
        for sample_index, batch in enumerate(
            loader
        ):
            if sample_index >= n_eval:
                break

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

            prediction_norm, ensemble_std_norm = (
                sample_direct_ensemble(
                    model=model,
                    current_masked=(
                        current_masked
                    ),
                    visible_mask=(
                        visible_mask
                    ),
                    ocean_mask=(
                        ocean_mask
                    ),
                    temporal_context=(
                        temporal_context
                    ),
                    temporal_masks=(
                        temporal_masks
                    ),
                    doy=doy,
                    auxiliary_context=auxiliary_context,
                    steps=args.steps,
                    ensemble_size=(
                        args.ensemble_size
                    ),
                    base_seed=seed,
                    sample_index=(
                        sample_index
                    ),
                    use_amp=args.amp,
                    clamp_min_norm=(
                        clamp_min_norm
                    ),
                    clamp_max_norm=(
                        clamp_max_norm
                    ),
                )
            )

            prediction_norm_np = (
                prediction_norm[
                    0,
                    0,
                ]
                .detach()
                .cpu()
                .numpy()
            )

            ensemble_std_norm_np = (
                ensemble_std_norm[
                    0,
                    0,
                ]
                .detach()
                .cpu()
                .numpy()
            )

            target_norm_np = (
                batch[
                    "target"
                ][
                    0,
                    0,
                ]
                .numpy()
            )

            current_masked_np = (
                batch[
                    "current_masked"
                ][
                    0,
                    0,
                ]
                .numpy()
            )

            loss_mask_np = (
                batch[
                    "loss_mask"
                ][
                    0,
                    0,
                ]
                .numpy()
                > 0.5
            )

            target_valid_np = (
                batch[
                    "target_valid_mask"
                ][
                    0,
                    0,
                ]
                .numpy()
                > 0.5
            )

            visible_mask_np = (
                batch[
                    "current_visible_mask"
                ][
                    0,
                    0,
                ]
                .numpy()
                > 0.5
            )

            hidden_count = int(
                loss_mask_np.sum()
            )

            expected_hidden_pixels += (
                hidden_count
            )

            prediction_physical = (
                inverse_normalized_log_chl(
                    prediction_norm_np,
                    train_mean,
                    train_std,
                )
            )

            target_physical = (
                inverse_normalized_log_chl(
                    target_norm_np,
                    train_mean,
                    train_std,
                )
            )

            if not args.no_clamp:
                prediction_physical = (
                    np.clip(
                        prediction_physical,
                        chl_min,
                        chl_max,
                    )
                )

            global_metrics.update(
                prediction=(
                    prediction_physical
                ),
                target=(
                    target_physical
                ),
                mask=loss_mask_np,
            )

            (
                sample_rmse,
                sample_mae,
                sample_corr,
            ) = masked_sample_metrics(
                prediction=(
                    prediction_physical
                ),
                target=(
                    target_physical
                ),
                mask=loss_mask_np,
            )

            sample_ssim = completed_ssim(
                prediction=(
                    prediction_physical
                ),
                target=(
                    target_physical
                ),
                loss_mask=(
                    loss_mask_np
                ),
                target_valid=(
                    target_valid_np
                ),
            )

            ssim_values.append(
                sample_ssim
            )

            hidden_std_norm_mean = (
                float(
                    ensemble_std_norm_np[
                        loss_mask_np
                    ].mean()
                )
                if hidden_count > 0
                else math.nan
            )

            date = str(
                scalar_from_batch(
                    batch[
                        "date"
                    ]
                )
            )

            row = {
                "sample_index": (
                    sample_index
                ),
                "date": date,
                "time_idx": int(
                    scalar_from_batch(
                        batch[
                            "time_idx"
                        ]
                    )
                ),
                "patch_y": int(
                    scalar_from_batch(
                        batch[
                            "patch_y"
                        ]
                    )
                ),
                "patch_x": int(
                    scalar_from_batch(
                        batch[
                            "patch_x"
                        ]
                    )
                ),
                "hidden_pixels": (
                    hidden_count
                ),
                "rmse": (
                    sample_rmse
                ),
                "mae": (
                    sample_mae
                ),
                "pearson_correlation": (
                    sample_corr
                ),
                "ssim": (
                    sample_ssim
                ),
                "mean_hidden_ensemble_std_normalized": (
                    hidden_std_norm_mean
                ),
            }

            per_sample_rows.append(
                row
            )

            if (
                sample_index
                in figure_indices
            ):
                save_qualitative_figure(
                    path=(
                        figure_dir
                        / (
                            f"sample_"
                            f"{sample_index:04d}.png"
                        )
                    ),
                    current_masked_norm=(
                        current_masked_np
                    ),
                    visible_mask=(
                        visible_mask_np
                    ),
                    prediction_physical=(
                        prediction_physical
                    ),
                    target_physical=(
                        target_physical
                    ),
                    loss_mask=(
                        loss_mask_np
                    ),
                    target_valid=(
                        target_valid_np
                    ),
                    train_mean=(
                        train_mean
                    ),
                    train_std=(
                        train_std
                    ),
                    date=date,
                    sample_index=(
                        sample_index
                    ),
                )

            if (
                (sample_index + 1)
                % args.log_every
                == 0
                or sample_index == 0
                or (
                    sample_index + 1
                    == n_eval
                )
            ):
                elapsed = (
                    time.time()
                    - start_time
                )

                rate = (
                    elapsed
                    / (
                        sample_index
                        + 1
                    )
                )

                remaining = (
                    rate
                    * (
                        n_eval
                        - sample_index
                        - 1
                    )
                )

                print(
                    f"[{sample_index + 1:,}/"
                    f"{n_eval:,}] "
                    f"sample RMSE="
                    f"{sample_rmse:.6f} "
                    f"MAE={sample_mae:.6f} "
                    f"SSIM={sample_ssim:.6f} "
                    f"| avg {rate:.1f}s/sample "
                    f"| ETA "
                    f"{remaining / 3600:.2f} h",
                    flush=True,
                )

    finally:
        dataset.close()

    metrics = (
        global_metrics.finalize()
    )

    finite_ssim = np.asarray(
        [
            x
            for x in ssim_values
            if np.isfinite(x)
        ],
        dtype=np.float64,
    )

    metrics.update(
        {
            "model": "DIRECT",
            "split": "test",
            "checkpoint": str(
                args.checkpoint
            ),
            "checkpoint_epoch": (
                checkpoint.get(
                    "epoch"
                )
            ),
            "training_version": (
                checkpoint.get(
                    "training_version"
                )
            ),
            "samples_evaluated": int(
                n_eval
            ),
            "expected_hidden_pixels_from_masks": int(
                expected_hidden_pixels
            ),
            "mean_ssim": (
                float(
                    np.mean(
                        finite_ssim
                    )
                )
                if finite_ssim.size
                > 0
                else math.nan
            ),
            "median_ssim": (
                float(
                    np.median(
                        finite_ssim
                    )
                )
                if finite_ssim.size
                > 0
                else math.nan
            ),
            "ssim_samples": int(
                finite_ssim.size
            ),
            "reverse_steps": int(
                args.steps
            ),
            "ensemble_size": int(
                args.ensemble_size
            ),
            "ensemble_reduction": (
                "mean"
            ),
            "seed": int(
                seed
            ),
            "aux_mode": aux_mode,
            "auxiliary_channels": int(
                auxiliary_channels_for_mode(aux_mode)
            ),
            "amp": bool(
                args.amp
            ),
            "chl_min": float(
                chl_min
            ),
            "chl_max": float(
                chl_max
            ),
            "physical_clamp_enabled": (
                not args.no_clamp
            ),
            "train_mean": float(
                train_mean
            ),
            "train_std": float(
                train_std
            ),
            "elapsed_hours": float(
                (
                    time.time()
                    - start_time
                )
                / 3600.0
            ),
        }
    )

    metrics_path = (
        args.output_dir
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
            allow_nan=True,
        )

    csv_path = (
        args.output_dir
        / "per_sample_metrics.csv"
    )

    if per_sample_rows:
        with open(
            csv_path,
            "w",
            newline="",
        ) as f:
            writer = (
                csv.DictWriter(
                    f,
                    fieldnames=list(
                        per_sample_rows[
                            0
                        ].keys()
                    ),
                )
            )

            writer.writeheader()
            writer.writerows(
                per_sample_rows
            )

    print()
    print("=" * 76)
    print("DIRECT EVALUATION COMPLETE")
    print("=" * 76)
    print(
        "Samples:",
        f"{metrics['samples_evaluated']:,}",
    )
    print(
        "Hidden pixels:",
        f"{metrics['hidden_pixels']:,}",
    )
    print(
        "RMSE:",
        metrics["rmse"],
    )
    print(
        "MAE:",
        metrics["mae"],
    )
    print(
        "Pearson correlation:",
        metrics[
            "pearson_correlation"
        ],
    )
    print(
        "Mean SSIM:",
        metrics["mean_ssim"],
    )
    print(
        "Median SSIM:",
        metrics[
            "median_ssim"
        ],
    )
    print(
        "Elapsed hours:",
        f"{metrics['elapsed_hours']:.2f}",
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
    print(
        "Figures:",
        figure_dir,
    )


if __name__ == "__main__":
    main()

