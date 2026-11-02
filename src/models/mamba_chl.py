#!/usr/bin/env python3
"""
Spatiotemporal Mamba-3 model for chlorophyll-a reconstruction.

Architecture
------------
Input from chl_data.py:
    mamba_sequence: [B,T,3,H,W]

Per-day channels:
    0 normalized log-CHL
    1 observation-validity mask
    2 ocean mask

The model is intentionally adapted for a centered reconstruction problem
rather than used as a language model:

    shared spatial encoder
        ↓
    per-pixel temporal sequences at bottleneck resolution
        ↓
    bidirectional Mamba-3 blocks
        ↓
    target-day bottleneck representation
        ↓
    U-Net-style spatial decoder using target-day encoder skips
        ↓
    reconstructed normalized log-CHL [B,1,H,W]

Why bidirectional?
------------------
Our reconstruction sequence is centered on the target date and contains both
past and future context. A standard causal Mamba block evaluated at the centre
token would only use the past. We therefore run one Mamba-3 block forward and
one on the reversed temporal sequence, reverse the latter output back, and fuse
the two streams. This is a task-specific adaptation, not a claim from the
Mamba-3 paper.

Mamba-3 backend
---------------
This file uses the official `mamba_ssm.Mamba3` SISO block. The current
state-spaces/mamba API exposes arguments including:
    d_model, d_state, headdim, is_mimo, chunk_size, is_outproj_norm

We start with SISO (`is_mimo=False`) for easier installation and greater
stability on the V100 environment.

Recommended first configuration
-------------------------------
    sequence length: 15
    encoder channels: 64, 128, 256
    d_model: 256
    d_state: 64
    Mamba layers: 4
    headdim: 64
    bidirectional: True

The model predicts normalized log-CHL. Training/evaluation should use the same
target, artificial masks, split, and inverse transform as the existing models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from mamba_ssm import Mamba3
    _MAMBA3_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on HPC installation
    Mamba3 = None
    _MAMBA3_IMPORT_ERROR = exc


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _valid_group_count(
    channels: int,
    preferred: int = 8,
) -> int:
    for groups in range(
        min(preferred, channels),
        0,
        -1,
    ):
        if channels % groups == 0:
            return groups
    return 1


def _require_mamba3() -> None:
    if Mamba3 is None:
        raise ImportError(
            "Could not import mamba_ssm.Mamba3. Install/update the official "
            "state-spaces/mamba package in the active environment. "
            f"Original import error: {_MAMBA3_IMPORT_ERROR!r}"
        )


# ---------------------------------------------------------------------
# Spatial encoder / decoder blocks
# ---------------------------------------------------------------------

class ConvResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        group_norm_groups: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()

        g1 = _valid_group_count(
            in_channels,
            group_norm_groups,
        )
        g2 = _valid_group_count(
            out_channels,
            group_norm_groups,
        )

        self.norm1 = nn.GroupNorm(
            g1,
            in_channels,
        )
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

        self.norm2 = nn.GroupNorm(
            g2,
            out_channels,
        )
        self.dropout = nn.Dropout2d(
            float(dropout)
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
            )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        residual = self.shortcut(x)

        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + residual


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        group_norm_groups: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.res = ConvResidualBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            group_norm_groups=group_norm_groups,
            dropout=dropout,
        )

        self.down = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ):
        skip = self.res(x)
        down = self.down(skip)
        return skip, down


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        group_norm_groups: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.res = ConvResidualBlock(
            in_channels=(
                in_channels
                + skip_channels
            ),
            out_channels=out_channels,
            group_norm_groups=group_norm_groups,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        x = torch.cat(
            [x, skip],
            dim=1,
        )

        return self.res(x)


# ---------------------------------------------------------------------
# Temporal conditioning
# ---------------------------------------------------------------------

class TemporalMetadataEmbedding(nn.Module):
    """
    Projects per-time-step metadata into d_model.

    Input features:
      - day-of-year sin/cos                         2
      - mamba_time_valid                           1
      - normalized relative temporal offset        1

    Total: 4 features per day.
    """

    def __init__(
        self,
        d_model: int,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                4,
                d_model,
            ),
            nn.SiLU(),
            nn.Linear(
                d_model,
                d_model,
            ),
        )

    def forward(
        self,
        doy: torch.Tensor,
        time_valid: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        if doy.ndim != 3 or doy.shape[-1] != 2:
            raise ValueError(
                f"doy must be [B,T,2], got {tuple(doy.shape)}"
            )

        if time_valid.ndim != 2:
            raise ValueError(
                f"time_valid must be [B,T], got {tuple(time_valid.shape)}"
            )

        if offsets.ndim == 1:
            offsets = offsets[None, :].expand(
                doy.shape[0],
                -1,
            )

        if offsets.ndim != 2:
            raise ValueError(
                f"offsets must be [T] or [B,T], got {tuple(offsets.shape)}"
            )

        offsets = offsets.to(
            device=doy.device,
            dtype=doy.dtype,
        )

        denom = offsets.abs().amax(
            dim=1,
            keepdim=True,
        ).clamp_min(1.0)

        offsets_norm = (
            offsets / denom
        )

        features = torch.cat(
            [
                doy,
                time_valid[..., None].to(
                    dtype=doy.dtype
                ),
                offsets_norm[..., None],
            ],
            dim=-1,
        )

        return self.net(features)


# ---------------------------------------------------------------------
# Bidirectional Mamba-3 layer
# ---------------------------------------------------------------------

class BidirectionalMamba3Block(nn.Module):
    """
    Pre-norm bidirectional Mamba-3 residual block.

    Forward stream:
        sequence in chronological order.

    Backward stream:
        sequence reversed in time, processed by a separate Mamba-3 block,
        then reversed back.

    The two outputs are concatenated and projected to d_model.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        headdim: int = 64,
        expand: int = 2,
        chunk_size: int = 16,
        dropout: float = 0.0,
        bidirectional: bool = True,
        is_outproj_norm: bool = False,
    ):
        super().__init__()

        _require_mamba3()

        if (
            int(expand * d_model)
            % headdim
            != 0
        ):
            raise ValueError(
                "Mamba-3 requires expand*d_model divisible by headdim. "
                f"Got expand={expand}, d_model={d_model}, "
                f"headdim={headdim}."
            )

        self.d_model = int(d_model)
        self.bidirectional = bool(
            bidirectional
        )

        self.norm = nn.LayerNorm(
            d_model
        )

        self.forward_mamba = Mamba3(
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            headdim=headdim,
            is_mimo=False,
            chunk_size=chunk_size,
            is_outproj_norm=is_outproj_norm,
        )

        if self.bidirectional:
            self.backward_mamba = Mamba3(
                d_model=d_model,
                d_state=d_state,
                expand=expand,
                headdim=headdim,
                is_mimo=False,
                chunk_size=chunk_size,
                is_outproj_norm=is_outproj_norm,
            )

            self.fuse = nn.Linear(
                2 * d_model,
                d_model,
            )
        else:
            self.backward_mamba = None
            self.fuse = nn.Identity()

        self.dropout = nn.Dropout(
            float(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        time_valid: Optional[
            torch.Tensor
        ] = None,
    ) -> torch.Tensor:
        """
        x: [N,T,D]

        time_valid:
            optional [N,T] or broadcastable mask. Invalid padded positions
            are zeroed before and after the Mamba block. Natural cloudy days
            are NOT invalid; only sequence positions outside the split are.
        """
        residual = x
        h = self.norm(x)

        if time_valid is not None:
            valid = time_valid.to(
                device=h.device,
                dtype=h.dtype,
            )[..., None]

            h = h * valid
        else:
            valid = None

        fwd = self.forward_mamba(h)

        if self.bidirectional:
            rev_in = torch.flip(
                h,
                dims=[1],
            )

            rev = self.backward_mamba(
                rev_in
            )

            bwd = torch.flip(
                rev,
                dims=[1],
            )

            h = self.fuse(
                torch.cat(
                    [fwd, bwd],
                    dim=-1,
                )
            )
        else:
            h = fwd

        if valid is not None:
            h = h * valid

        return (
            residual
            + self.dropout(h)
        )


# ---------------------------------------------------------------------
# Complete Chl-a Mamba model
# ---------------------------------------------------------------------

class ChlorophyllMamba(nn.Module):
    """
    CNN encoder -> temporal Mamba-3 -> U-Net decoder.

    Parameters
    ----------
    in_channels:
        Per-day input channels. Current loader uses 3.

    encoder_channels:
        Three-level encoder widths. Default (64,128,256) reduces
        128x128 -> 64x64 -> 32x32 before Mamba.

    d_model:
        Temporal feature dimension. Default 256.

    num_mamba_layers:
        Number of bidirectional temporal Mamba layers.

    center_index:
        Optional fixed target-day index. If None, uses T//2.

    predict_residual:
        If True, output is interpreted as correction added to current
        masked centre CHL. Default False for straightforward regression.
    """

    def __init__(
        self,
        in_channels: int = 3,
        encoder_channels: tuple[
            int, int, int
        ] = (
            64,
            128,
            256,
        ),
        d_model: int = 256,
        d_state: int = 64,
        num_mamba_layers: int = 4,
        headdim: int = 64,
        expand: int = 2,
        chunk_size: int = 16,
        bidirectional: bool = True,
        dropout: float = 0.0,
        group_norm_groups: int = 8,
        use_temporal_metadata: bool = True,
        center_index: Optional[
            int
        ] = None,
        predict_residual: bool = False,
    ):
        super().__init__()

        _require_mamba3()

        if len(
            encoder_channels
        ) != 3:
            raise ValueError(
                "encoder_channels must contain exactly "
                "three levels."
            )

        c0, c1, c2 = [
            int(c)
            for c
            in encoder_channels
        ]

        self.in_channels = int(
            in_channels
        )
        self.encoder_channels = (
            c0,
            c1,
            c2,
        )
        self.d_model = int(
            d_model
        )
        self.center_index = (
            center_index
        )
        self.use_temporal_metadata = bool(
            use_temporal_metadata
        )
        self.predict_residual = bool(
            predict_residual
        )

        # ---------------------------------------------------------
        # Shared spatial encoder for each day independently.
        # ---------------------------------------------------------

        self.input_proj = nn.Conv2d(
            in_channels,
            c0,
            kernel_size=3,
            padding=1,
        )

        self.enc0 = DownBlock(
            in_channels=c0,
            out_channels=c0,
            group_norm_groups=group_norm_groups,
            dropout=dropout,
        )

        self.enc1 = DownBlock(
            in_channels=c0,
            out_channels=c1,
            group_norm_groups=group_norm_groups,
            dropout=dropout,
        )

        self.bottleneck_encoder = (
            ConvResidualBlock(
                in_channels=c1,
                out_channels=c2,
                group_norm_groups=(
                    group_norm_groups
                ),
                dropout=dropout,
            )
        )

        if c2 == d_model:
            self.to_mamba = nn.Identity()
            self.from_mamba = nn.Identity()
        else:
            self.to_mamba = nn.Conv2d(
                c2,
                d_model,
                kernel_size=1,
            )
            self.from_mamba = nn.Conv2d(
                d_model,
                c2,
                kernel_size=1,
            )

        if self.use_temporal_metadata:
            self.temporal_metadata = (
                TemporalMetadataEmbedding(
                    d_model=d_model
                )
            )
        else:
            self.temporal_metadata = None

        # ---------------------------------------------------------
        # Temporal Mamba stack.
        # ---------------------------------------------------------

        self.mamba_layers = nn.ModuleList(
            [
                BidirectionalMamba3Block(
                    d_model=d_model,
                    d_state=d_state,
                    headdim=headdim,
                    expand=expand,
                    chunk_size=(
                        chunk_size
                    ),
                    dropout=dropout,
                    bidirectional=(
                        bidirectional
                    ),
                )
                for _ in range(
                    num_mamba_layers
                )
            ]
        )

        self.temporal_output_norm = (
            nn.LayerNorm(
                d_model
            )
        )

        # ---------------------------------------------------------
        # Spatial decoder. Uses centre-day encoder skip maps.
        # ---------------------------------------------------------

        self.dec1 = UpBlock(
            in_channels=c2,
            skip_channels=c1,
            out_channels=c1,
            group_norm_groups=(
                group_norm_groups
            ),
            dropout=dropout,
        )

        self.dec0 = UpBlock(
            in_channels=c1,
            skip_channels=c0,
            out_channels=c0,
            group_norm_groups=(
                group_norm_groups
            ),
            dropout=dropout,
        )

        final_groups = (
            _valid_group_count(
                c0,
                group_norm_groups,
            )
        )

        self.output_head = nn.Sequential(
            nn.GroupNorm(
                final_groups,
                c0,
            ),
            nn.SiLU(),
            nn.Conv2d(
                c0,
                1,
                kernel_size=3,
                padding=1,
            ),
        )

    def _resolve_center(
        self,
        T: int,
    ) -> int:
        center = (
            self.center_index
            if self.center_index
            is not None
            else T // 2
        )

        if not (
            0 <= center < T
        ):
            raise ValueError(
                f"center_index={center} "
                f"invalid for T={T}"
            )

        return int(center)

    def _encode_all_frames(
        self,
        sequence: torch.Tensor,
    ):
        """
        sequence: [B,T,C,H,W]

        Returns:
            skips0: [B,T,c0,H,W]
            skips1: [B,T,c1,H/2,W/2]
            bottleneck: [B,T,d_model,H/4,W/4]
        """
        if sequence.ndim != 5:
            raise ValueError(
                "sequence must have shape "
                f"[B,T,C,H,W], got {tuple(sequence.shape)}"
            )

        B, T, C, H, W = (
            sequence.shape
        )

        if C != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} "
                f"channels per day, got {C}"
            )

        x = sequence.reshape(
            B * T,
            C,
            H,
            W,
        )

        x = self.input_proj(x)

        skip0, x = self.enc0(x)
        skip1, x = self.enc1(x)
        x = self.bottleneck_encoder(
            x
        )
        x = self.to_mamba(x)

        _, d_model, Hb, Wb = (
            x.shape
        )

        skip0 = skip0.reshape(
            B,
            T,
            skip0.shape[1],
            skip0.shape[2],
            skip0.shape[3],
        )

        skip1 = skip1.reshape(
            B,
            T,
            skip1.shape[1],
            skip1.shape[2],
            skip1.shape[3],
        )

        bottleneck = x.reshape(
            B,
            T,
            d_model,
            Hb,
            Wb,
        )

        return (
            skip0,
            skip1,
            bottleneck,
        )

    def _temporal_mamba(
        self,
        bottleneck: torch.Tensor,
        mamba_doy: Optional[
            torch.Tensor
        ],
        mamba_time_valid: Optional[
            torch.Tensor
        ],
        mamba_offsets: Optional[
            torch.Tensor
        ],
    ) -> torch.Tensor:
        """
        bottleneck: [B,T,D,Hb,Wb]

        Applies Mamba independently to each bottleneck spatial location:
            [B*Hb*Wb, T, D]
        """
        B, T, D, Hb, Wb = (
            bottleneck.shape
        )

        # [B,Hb,Wb,T,D]
        tokens = bottleneck.permute(
            0,
            3,
            4,
            1,
            2,
        )

        tokens = tokens.reshape(
            B * Hb * Wb,
            T,
            D,
        )

        # Add one metadata vector per day to every spatial position.
        if self.use_temporal_metadata:
            if (
                mamba_doy is None
                or mamba_time_valid
                is None
                or mamba_offsets
                is None
            ):
                raise ValueError(
                    "use_temporal_metadata=True requires "
                    "mamba_doy, mamba_time_valid, and "
                    "mamba_offsets."
                )

            metadata = (
                self.temporal_metadata(
                    doy=mamba_doy,
                    time_valid=(
                        mamba_time_valid
                    ),
                    offsets=(
                        mamba_offsets
                    ),
                )
            )  # [B,T,D]

            metadata = (
                metadata[:, None, None]
                .expand(
                    B,
                    Hb,
                    Wb,
                    T,
                    D,
                )
                .reshape(
                    B * Hb * Wb,
                    T,
                    D,
                )
            )

            tokens = (
                tokens
                + metadata
            )

        valid_for_tokens = None

        if (
            mamba_time_valid
            is not None
        ):
            valid_for_tokens = (
                mamba_time_valid[
                    :, None, None, :
                ]
                .expand(
                    B,
                    Hb,
                    Wb,
                    T,
                )
                .reshape(
                    B * Hb * Wb,
                    T,
                )
            )

        for layer in self.mamba_layers:
            tokens = layer(
                tokens,
                time_valid=(
                    valid_for_tokens
                ),
            )

        tokens = (
            self.temporal_output_norm(
                tokens
            )
        )

        tokens = tokens.reshape(
            B,
            Hb,
            Wb,
            T,
            D,
        )

        # [B,T,D,Hb,Wb]
        return tokens.permute(
            0,
            3,
            4,
            1,
            2,
        ).contiguous()

    def forward(
        self,
        mamba_sequence: torch.Tensor,
        mamba_doy: Optional[
            torch.Tensor
        ] = None,
        mamba_time_valid: Optional[
            torch.Tensor
        ] = None,
        mamba_offsets: Optional[
            torch.Tensor
        ] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        mamba_sequence:
            [B,T,3,H,W]

        mamba_doy:
            [B,T,2]

        mamba_time_valid:
            [B,T]

        mamba_offsets:
            [T] or [B,T]

        Returns
        -------
        prediction:
            [B,1,H,W], reconstructed normalized log-CHL.
        """
        if mamba_sequence.ndim != 5:
            raise ValueError(
                "mamba_sequence must be "
                f"[B,T,C,H,W], got "
                f"{tuple(mamba_sequence.shape)}"
            )

        B, T, _, H, W = (
            mamba_sequence.shape
        )

        center = self._resolve_center(
            T
        )

        (
            skip0,
            skip1,
            bottleneck,
        ) = self._encode_all_frames(
            mamba_sequence
        )

        temporal_features = (
            self._temporal_mamba(
                bottleneck=(
                    bottleneck
                ),
                mamba_doy=mamba_doy,
                mamba_time_valid=(
                    mamba_time_valid
                ),
                mamba_offsets=(
                    mamba_offsets
                ),
            )
        )

        # Target-day feature after it has incorporated both directions.
        x = temporal_features[
            :,
            center,
        ]

        x = self.from_mamba(x)

        center_skip1 = skip1[
            :,
            center,
        ]
        center_skip0 = skip0[
            :,
            center,
        ]

        x = self.dec1(
            x,
            center_skip1,
        )

        x = self.dec0(
            x,
            center_skip0,
        )

        prediction = (
            self.output_head(x)
        )

        if prediction.shape[-2:] != (
            H,
            W,
        ):
            prediction = F.interpolate(
                prediction,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )

        if self.predict_residual:
            center_chl = (
                mamba_sequence[
                    :,
                    center,
                    0:1,
                ]
            )

            prediction = (
                center_chl
                + prediction
            )

        return prediction


# ---------------------------------------------------------------------
# Alias
# ---------------------------------------------------------------------

MambaChlorophyllModel = (
    ChlorophyllMamba
)


# ---------------------------------------------------------------------
# Standalone sanity test
# ---------------------------------------------------------------------

def _sanity_check():
    _require_mamba3()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # Mamba-3 official kernels are intended for CUDA. This check remains
    # useful for shape validation when CUDA is available.
    if device.type != "cuda":
        print(
            "WARNING: Mamba-3 kernels normally require CUDA. "
            "Creating the model may work, but the forward pass "
            "may not run on CPU."
        )

    model = ChlorophyllMamba(
        in_channels=3,
        encoder_channels=(
            64,
            128,
            256,
        ),
        d_model=256,
        d_state=64,
        num_mamba_layers=4,
        headdim=64,
        expand=2,
        chunk_size=64,
        bidirectional=True,
        dropout=0.0,
    ).to(device)

    B = 1
    T = 15
    H = W = 128

    sequence = torch.randn(
        B,
        T,
        3,
        H,
        W,
        device=device,
    )

    # Make masks plausible.
    sequence[
        :,
        :,
        1,
    ] = (
        torch.rand(
            B,
            T,
            H,
            W,
            device=device,
        )
        > 0.4
    ).float()

    sequence[
        :,
        :,
        2,
    ] = 1.0

    doy_angles = torch.linspace(
        0.0,
        0.2,
        T,
        device=device,
    )

    doy = torch.stack(
        [
            torch.sin(
                doy_angles
            ),
            torch.cos(
                doy_angles
            ),
        ],
        dim=-1,
    )[None, ...]

    time_valid = torch.ones(
        B,
        T,
        device=device,
    )

    offsets = torch.arange(
        -(T // 2),
        T // 2 + 1,
        device=device,
    )[None, :]

    params = sum(
        p.numel()
        for p
        in model.parameters()
        if p.requires_grad
    )

    print("Device:", device)
    print(
        "Trainable parameters:",
        f"{params:,}",
    )

    if device.type == "cuda":
        with torch.no_grad():
            y = model(
                mamba_sequence=(
                    sequence
                ),
                mamba_doy=doy,
                mamba_time_valid=(
                    time_valid
                ),
                mamba_offsets=(
                    offsets
                ),
            )

        print(
            "Input:",
            tuple(
                sequence.shape
            ),
        )
        print(
            "Output:",
            tuple(
                y.shape
            ),
        )
        print(
            "Finite:",
            bool(
                torch.isfinite(
                    y
                ).all()
            ),
        )

        if y.shape != (
            B,
            1,
            H,
            W,
        ):
            raise AssertionError(
                f"Unexpected output "
                f"shape: {tuple(y.shape)}"
            )


if __name__ == "__main__":
    _sanity_check()
