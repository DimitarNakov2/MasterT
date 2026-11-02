#!/usr/bin/env python3
"""
DIRECT backbone adapted for CMEMS chlorophyll-a reconstruction.

The implementation follows the architecture described in Grega Rovšček's
DIRECT thesis as closely as possible from the written specification:

- four-level U-Net
- feature widths: 128, 256, 384, 512
- encoder residual-block counts: 5, 3, 3, 3
- FiLM conditioning in every residual block
- deepest encoder level: self-attention after every residual block
- bottleneck: residual block -> self-attention -> residual block
- symmetric decoder with skip connections
- attention at the corresponding deepest decoder level
- final GroupNorm -> SiLU -> 3x3 convolution
- conditioning dimension: 512
- conditioning = DoY embedding + continuous flow/diffusion-step embedding

DIRECT model input
------------------
The U-Net receives a concatenation of:

    noisy / FUSE-rectified centre field     1 channel
    temporal context [prev, centre, next]   3 channels
    temporal masks:
        previous source one-hot              3 channels
        centre observed/missing one-hot      2 channels
        future source one-hot                3 channels
                                               ----------
    total                                    12 channels

The temporal context preparation, prioritized t±1 -> t±2 -> t±3 fallback,
mask construction, artificial cloud masking, and FUSE rectification are data/
training concerns and are deliberately kept outside the network itself.

For the Chl-a adaptation:
- fields are normalized log-Chl-a rather than SST
- output is one velocity channel for the centre Chl-a field
- 128x128 patches work naturally with the three downsampling stages

IMPORTANT
---------
The thesis specifies where self-attention is applied but does not state the
number of attention heads in the architecture text. `attention_heads=8` is
therefore an explicit implementation choice and is configurable.

Likewise, this file implements the neural backbone and helper utilities only.
The flow-matching target, FUSE training construction, optimizer, scheduler,
ensemble sampling, and sequential DIRECTSEQ pass belong in train_direct.py /
evaluate_direct.py.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------

def _valid_group_count(channels: int, preferred: int = 32) -> int:
    """
    Largest convenient GroupNorm group count <= preferred that divides C.
    """
    for groups in range(min(preferred, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _broadcast_scalar_condition(
    value: torch.Tensor,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Normalize k input to [B].

    Accepts:
        scalar
        [B]
        [B,1]
    """
    value = torch.as_tensor(
        value,
        device=device,
        dtype=dtype,
    )

    if value.ndim == 0:
        value = value.expand(batch_size)

    elif value.ndim == 2 and value.shape[1] == 1:
        value = value[:, 0]

    elif value.ndim != 1:
        raise ValueError(
            f"Expected k as scalar, [B], or [B,1], got {tuple(value.shape)}"
        )

    if value.shape[0] == 1 and batch_size > 1:
        value = value.expand(batch_size)

    if value.shape[0] != batch_size:
        raise ValueError(
            f"k batch dimension {value.shape[0]} != input batch {batch_size}"
        )

    return value


# ---------------------------------------------------------------------
# Continuous sinusoidal embedding for k in [0,1]
# ---------------------------------------------------------------------

class ContinuousSinusoidalEmbedding(nn.Module):
    """
    Sinusoidal embedding for the continuous DIRECT flow coordinate k.

    k=0 corresponds to clean data.
    k=1 corresponds to Gaussian noise.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        max_period: float = 10_000.0,
    ):
        super().__init__()

        if embedding_dim < 2:
            raise ValueError("embedding_dim must be >= 2")

        self.embedding_dim = int(embedding_dim)
        self.max_period = float(max_period)

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        if k.ndim != 1:
            raise ValueError(
                f"ContinuousSinusoidalEmbedding expects [B], got {tuple(k.shape)}"
            )

        half = self.embedding_dim // 2

        if half == 1:
            frequencies = torch.ones(
                1,
                device=k.device,
                dtype=k.dtype,
            )
        else:
            frequencies = torch.exp(
                -math.log(self.max_period)
                * torch.arange(
                    half,
                    device=k.device,
                    dtype=k.dtype,
                )
                / (half - 1)
            )

        # Scaling by 2π gives useful variation across k∈[0,1].
        args = (
            2.0
            * math.pi
            * k[:, None]
            * frequencies[None, :]
        )

        emb = torch.cat(
            [
                torch.sin(args),
                torch.cos(args),
            ],
            dim=-1,
        )

        if emb.shape[-1] < self.embedding_dim:
            emb = F.pad(
                emb,
                (0, self.embedding_dim - emb.shape[-1]),
            )

        return emb


# ---------------------------------------------------------------------
# DIRECT conditioning
# ---------------------------------------------------------------------

class DIRECTConditioning(nn.Module):
    """
    e = DoY_MLP([sin(doy), cos(doy)]) + k_MLP(sinusoidal(k))

    The thesis uses a final conditioning dimension De=512.
    """

    def __init__(
        self,
        conditioning_dim: int = 512,
        k_embedding_dim: int = 256,
    ):
        super().__init__()

        self.conditioning_dim = int(conditioning_dim)

        self.k_embedding = ContinuousSinusoidalEmbedding(
            embedding_dim=k_embedding_dim,
        )

        self.k_mlp = nn.Sequential(
            nn.Linear(
                k_embedding_dim,
                conditioning_dim,
            ),
            nn.SiLU(),
            nn.Linear(
                conditioning_dim,
                conditioning_dim,
            ),
        )

        self.doy_mlp = nn.Sequential(
            nn.Linear(
                2,
                conditioning_dim,
            ),
            nn.SiLU(),
            nn.Linear(
                conditioning_dim,
                conditioning_dim,
            ),
        )

    def forward(
        self,
        doy: torch.Tensor,
        k: torch.Tensor,
    ) -> torch.Tensor:
        if doy.ndim != 2 or doy.shape[1] != 2:
            raise ValueError(
                f"doy must have shape [B,2], got {tuple(doy.shape)}"
            )

        k_emb = self.k_embedding(k)

        return (
            self.doy_mlp(doy)
            + self.k_mlp(k_emb)
        )


# ---------------------------------------------------------------------
# FiLM residual block
# ---------------------------------------------------------------------

class FiLMResidualBlock(nn.Module):
    """
    DIRECT-style residual block:

        GN -> SiLU -> Conv3x3
        GN -> FiLM(e) -> SiLU -> Conv3x3
        + residual shortcut

    The FiLM projection produces gamma and beta:

        FiLM(h,e) = GN(h) * (1 + gamma(e)) + beta(e)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conditioning_dim: int = 512,
        dropout: float = 0.0,
        group_norm_groups: int = 32,
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

        self.film = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                conditioning_dim,
                2 * out_channels,
            ),
        )

        self.dropout = nn.Dropout(
            p=float(dropout)
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
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        residual = self.shortcut(x)

        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        h = self.norm2(h)

        gamma_beta = self.film(
            conditioning
        )
        gamma, beta = torch.chunk(
            gamma_beta,
            2,
            dim=1,
        )

        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]

        h = h * (1.0 + gamma) + beta
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + residual


# ---------------------------------------------------------------------
# Resampling FiLM block
# ---------------------------------------------------------------------

class FiLMResampleBlock(nn.Module):
    """
    Down/upsampling variant described for DIRECT.

    Spatial resizing is applied before the residual block:
      down: 2x2 average pooling
      up:   nearest-neighbour interpolation

    Then a FiLM residual block changes feature width as needed.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conditioning_dim: int = 512,
        mode: str = "down",
        dropout: float = 0.0,
        group_norm_groups: int = 32,
    ):
        super().__init__()

        if mode not in {"down", "up"}:
            raise ValueError(
                "mode must be 'down' or 'up'"
            )

        self.mode = mode

        self.block = FiLMResidualBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            conditioning_dim=conditioning_dim,
            dropout=dropout,
            group_norm_groups=group_norm_groups,
        )

    def forward(
        self,
        x: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        if self.mode == "down":
            x = F.avg_pool2d(
                x,
                kernel_size=2,
                stride=2,
            )
        else:
            x = F.interpolate(
                x,
                scale_factor=2.0,
                mode="nearest",
            )

        return self.block(
            x,
            conditioning,
        )


# ---------------------------------------------------------------------
# Spatial self-attention
# ---------------------------------------------------------------------

class SpatialSelfAttention(nn.Module):
    """
    Self-attention across HxW spatial positions.

    Used only at the deepest 512-channel resolution and bottleneck,
    so memory usage remains manageable for 128x128 patches.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        group_norm_groups: int = 32,
    ):
        super().__init__()

        if channels % num_heads != 0:
            raise ValueError(
                f"channels={channels} must be divisible by num_heads={num_heads}"
            )

        groups = _valid_group_count(
            channels,
            group_norm_groups,
        )

        self.norm = nn.GroupNorm(
            groups,
            channels,
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        b, c, h, w = x.shape

        y = self.norm(x)
        y = y.flatten(2).transpose(1, 2)  # [B,HW,C]

        y, _ = self.attn(
            y,
            y,
            y,
            need_weights=False,
        )

        y = y.transpose(1, 2).reshape(
            b,
            c,
            h,
            w,
        )

        return x + y


# ---------------------------------------------------------------------
# Repeated residual stages
# ---------------------------------------------------------------------

class FiLMStage(nn.Module):
    """
    Sequence of FiLM residual blocks, optionally followed by attention
    after every block.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        conditioning_dim: int,
        dropout: float,
        group_norm_groups: int,
        attention: bool = False,
        attention_heads: int = 8,
    ):
        super().__init__()

        if num_blocks < 1:
            raise ValueError(
                "num_blocks must be >= 1"
            )

        blocks = []
        attentions = []

        current_channels = in_channels

        for _ in range(num_blocks):
            blocks.append(
                FiLMResidualBlock(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    conditioning_dim=conditioning_dim,
                    dropout=dropout,
                    group_norm_groups=group_norm_groups,
                )
            )

            if attention:
                attentions.append(
                    SpatialSelfAttention(
                        channels=out_channels,
                        num_heads=attention_heads,
                        group_norm_groups=group_norm_groups,
                    )
                )
            else:
                attentions.append(
                    nn.Identity()
                )

            current_channels = out_channels

        self.blocks = nn.ModuleList(
            blocks
        )
        self.attentions = nn.ModuleList(
            attentions
        )

    def forward(
        self,
        x: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        for block, attention in zip(
            self.blocks,
            self.attentions,
        ):
            x = block(
                x,
                conditioning,
            )
            x = attention(x)

        return x


# ---------------------------------------------------------------------
# DIRECT U-Net
# ---------------------------------------------------------------------

class DIRECT(nn.Module):
    """
    FiLM-conditioned U-Net backbone for DIRECT.

    Default architecture from the thesis:
        channels = [128, 256, 384, 512]
        blocks   = [5,   3,   3,   3]
        cond dim = 512

    Input:
        noisy_center:
            [B,1,H,W]
        temporal_context:
            [B,3,H,W] = [filled t-1, masked/current t, filled t+1]
        temporal_masks:
            [B,8,H,W]:
                3 previous-source one-hot
                2 centre observed/missing one-hot
                3 future-source one-hot
        doy:
            [B,2] sine/cosine day-of-year
        k:
            scalar/[B]/[B,1], continuous flow coordinate in [0,1]

    Output:
        [B,1,H,W] predicted DIRECT velocity field.
    """

    def __init__(
        self,
        input_channels: int = 12,
        output_channels: int = 1,
        channels: tuple[int, int, int, int] = (
            128,
            256,
            384,
            512,
        ),
        blocks_per_level: tuple[int, int, int, int] = (
            5,
            3,
            3,
            3,
        ),
        conditioning_dim: int = 512,
        k_embedding_dim: int = 256,
        attention_heads: int = 8,
        dropout: float = 0.0,
        group_norm_groups: int = 32,
    ):
        super().__init__()

        if len(channels) != 4:
            raise ValueError(
                "DIRECT expects exactly four channel levels"
            )

        if len(blocks_per_level) != 4:
            raise ValueError(
                "DIRECT expects exactly four block counts"
            )

        self.input_channels = int(
            input_channels
        )
        self.output_channels = int(
            output_channels
        )
        self.channels = tuple(
            int(c) for c in channels
        )
        self.blocks_per_level = tuple(
            int(n)
            for n in blocks_per_level
        )
        self.conditioning_dim = int(
            conditioning_dim
        )

        c0, c1, c2, c3 = self.channels
        n0, n1, n2, n3 = (
            self.blocks_per_level
        )

        self.conditioning = DIRECTConditioning(
            conditioning_dim=conditioning_dim,
            k_embedding_dim=k_embedding_dim,
        )

        self.input_conv = nn.Conv2d(
            self.input_channels,
            c0,
            kernel_size=3,
            padding=1,
        )

        # ---------------- Encoder ----------------

        self.enc0 = FiLMStage(
            in_channels=c0,
            out_channels=c0,
            num_blocks=n0,
            conditioning_dim=conditioning_dim,
            dropout=dropout,
            group_norm_groups=group_norm_groups,
            attention=False,
        )

        self.down0 = FiLMResampleBlock(
            in_channels=c0,
            out_channels=c1,
            conditioning_dim=conditioning_dim,
            mode="down",
            dropout=dropout,
            group_norm_groups=group_norm_groups,
        )

        self.enc1 = FiLMStage(
            in_channels=c1,
            out_channels=c1,
            num_blocks=n1,
            conditioning_dim=conditioning_dim,
            dropout=dropout,
            group_norm_groups=group_norm_groups,
            attention=False,
        )

        self.down1 = FiLMResampleBlock(
            in_channels=c1,
            out_channels=c2,
            conditioning_dim=conditioning_dim,
            mode="down",
            dropout=dropout,
            group_norm_groups=group_norm_groups,
        )

        self.enc2 = FiLMStage(
            in_channels=c2,
            out_channels=c2,
            num_blocks=n2,
            conditioning_dim=conditioning_dim,
            dropout=dropout,
            group_norm_groups=group_norm_groups,
            attention=False,
        )

        self.down2 = FiLMResampleBlock(
            in_channels=c2,
            out_channels=c3,
            conditioning_dim=conditioning_dim,
            mode="down",
            dropout=dropout,
            group_norm_groups=group_norm_groups,
        )

        # Deepest encoder: attention after every block.
        self.enc3 = FiLMStage(
            in_channels=c3,
            out_channels=c3,
            num_blocks=n3,
            conditioning_dim=conditioning_dim,
            dropout=dropout,
            group_norm_groups=group_norm_groups,
            attention=True,
            attention_heads=attention_heads,
        )

        # ---------------- Bottleneck ----------------

        self.mid_block1 = FiLMResidualBlock(
            in_channels=c3,
            out_channels=c3,
            conditioning_dim=conditioning_dim,
            dropout=dropout,
            group_norm_groups=group_norm_groups,
        )

        self.mid_attention = SpatialSelfAttention(
            channels=c3,
            num_heads=attention_heads,
            group_norm_groups=group_norm_groups,
        )

        self.mid_block2 = FiLMResidualBlock(
            in_channels=c3,
            out_channels=c3,
            conditioning_dim=conditioning_dim,
            dropout=dropout,
            group_norm_groups=group_norm_groups,
        )

        # ---------------- Decoder ----------------
        #
        # Skip at every resolution. After concatenation, the first block
        # of each decoder stage reduces concatenated channels back to the
        # stage width; remaining blocks mirror the encoder depth.

        self.dec3 = FiLMStage(
            in_channels=2 * c3,
            out_channels=c3,
            num_blocks=n3,
            conditioning_dim=conditioning_dim,
            dropout=dropout,
            group_norm_groups=group_norm_groups,
            attention=True,
            attention_heads=attention_heads,
        )

        self.up2 = FiLMResampleBlock(
            in_channels=c3,
            out_channels=c2,
            conditioning_dim=conditioning_dim,
            mode="up",
            dropout=dropout,
            group_norm_groups=group_norm_groups,
        )

        self.dec2 = FiLMStage(
            in_channels=2 * c2,
            out_channels=c2,
            num_blocks=n2,
            conditioning_dim=conditioning_dim,
            dropout=dropout,
            group_norm_groups=group_norm_groups,
            attention=False,
        )

        self.up1 = FiLMResampleBlock(
            in_channels=c2,
            out_channels=c1,
            conditioning_dim=conditioning_dim,
            mode="up",
            dropout=dropout,
            group_norm_groups=group_norm_groups,
        )

        self.dec1 = FiLMStage(
            in_channels=2 * c1,
            out_channels=c1,
            num_blocks=n1,
            conditioning_dim=conditioning_dim,
            dropout=dropout,
            group_norm_groups=group_norm_groups,
            attention=False,
        )

        self.up0 = FiLMResampleBlock(
            in_channels=c1,
            out_channels=c0,
            conditioning_dim=conditioning_dim,
            mode="up",
            dropout=dropout,
            group_norm_groups=group_norm_groups,
        )

        self.dec0 = FiLMStage(
            in_channels=2 * c0,
            out_channels=c0,
            num_blocks=n0,
            conditioning_dim=conditioning_dim,
            dropout=dropout,
            group_norm_groups=group_norm_groups,
            attention=False,
        )

        final_groups = _valid_group_count(
            c0,
            group_norm_groups,
        )

        self.output_norm = nn.GroupNorm(
            final_groups,
            c0,
        )

        self.output_conv = nn.Conv2d(
            c0,
            self.output_channels,
            kernel_size=3,
            padding=1,
        )

    @staticmethod
    def build_network_input(
        noisy_center: torch.Tensor,
        temporal_context: torch.Tensor,
        temporal_masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Concatenate the canonical 12 DIRECT channels.

        Expected:
            noisy_center      [B,1,H,W]
            temporal_context  [B,3,H,W]
            temporal_masks    [B,8,H,W]
        """
        if noisy_center.ndim != 4:
            raise ValueError(
                "noisy_center must be [B,1,H,W]"
            )
        if temporal_context.ndim != 4:
            raise ValueError(
                "temporal_context must be [B,3,H,W]"
            )
        if temporal_masks.ndim != 4:
            raise ValueError(
                "temporal_masks must be [B,8,H,W]"
            )

        if noisy_center.shape[1] != 1:
            raise ValueError(
                f"Expected 1 noisy-centre channel, got {noisy_center.shape[1]}"
            )

        if temporal_context.shape[1] != 3:
            raise ValueError(
                f"Expected 3 temporal-context channels, got {temporal_context.shape[1]}"
            )

        if temporal_masks.shape[1] != 8:
            raise ValueError(
                f"Expected 8 temporal-mask channels, got {temporal_masks.shape[1]}"
            )

        b, _, h, w = noisy_center.shape

        for name, tensor in (
            ("temporal_context", temporal_context),
            ("temporal_masks", temporal_masks),
        ):
            if (
                tensor.shape[0] != b
                or tensor.shape[-2:] != (h, w)
            ):
                raise ValueError(
                    f"{name} does not match noisy_center spatial/batch dimensions"
                )

        return torch.cat(
            [
                noisy_center,
                temporal_context,
                temporal_masks,
            ],
            dim=1,
        )

    def forward(
        self,
        noisy_center: torch.Tensor,
        temporal_context: torch.Tensor,
        temporal_masks: torch.Tensor,
        doy: torch.Tensor,
        k: torch.Tensor,
    ) -> torch.Tensor:
        network_input = self.build_network_input(
            noisy_center=noisy_center,
            temporal_context=temporal_context,
            temporal_masks=temporal_masks,
        )

        if (
            network_input.shape[1]
            != self.input_channels
        ):
            raise ValueError(
                f"Model configured for {self.input_channels} input channels, "
                f"but received {network_input.shape[1]}"
            )

        batch_size = network_input.shape[0]

        k = _broadcast_scalar_condition(
            value=k,
            batch_size=batch_size,
            device=network_input.device,
            dtype=network_input.dtype,
        )

        doy = doy.to(
            device=network_input.device,
            dtype=network_input.dtype,
        )

        conditioning = self.conditioning(
            doy=doy,
            k=k,
        )

        # ---------------- Encoder ----------------

        h = self.input_conv(
            network_input
        )

        skip0 = self.enc0(
            h,
            conditioning,
        )

        h = self.down0(
            skip0,
            conditioning,
        )

        skip1 = self.enc1(
            h,
            conditioning,
        )

        h = self.down1(
            skip1,
            conditioning,
        )

        skip2 = self.enc2(
            h,
            conditioning,
        )

        h = self.down2(
            skip2,
            conditioning,
        )

        skip3 = self.enc3(
            h,
            conditioning,
        )

        # ---------------- Bottleneck ----------------

        h = self.mid_block1(
            skip3,
            conditioning,
        )
        h = self.mid_attention(h)
        h = self.mid_block2(
            h,
            conditioning,
        )

        # ---------------- Decoder ----------------

        h = torch.cat(
            [h, skip3],
            dim=1,
        )
        h = self.dec3(
            h,
            conditioning,
        )

        h = self.up2(
            h,
            conditioning,
        )
        h = torch.cat(
            [h, skip2],
            dim=1,
        )
        h = self.dec2(
            h,
            conditioning,
        )

        h = self.up1(
            h,
            conditioning,
        )
        h = torch.cat(
            [h, skip1],
            dim=1,
        )
        h = self.dec1(
            h,
            conditioning,
        )

        h = self.up0(
            h,
            conditioning,
        )
        h = torch.cat(
            [h, skip0],
            dim=1,
        )
        h = self.dec0(
            h,
            conditioning,
        )

        h = self.output_norm(h)
        h = F.silu(h)

        return self.output_conv(h)


# ---------------------------------------------------------------------
# DIRECT / FUSE helper utilities
# ---------------------------------------------------------------------

def fuse_rectify(
    noisy_center: torch.Tensor,
    observed_center: torch.Tensor,
    visible_mask: torch.Tensor,
    ocean_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Observation-guided rectification (FUSE), adapted to our mask semantics.

    Our masks:
        visible_mask = 1 where centre Chl-a is visible to the model
        ocean_mask   = 1 for ocean, 0 for land

    Therefore:
        rectified =
            [noisy * (1-visible) + observed * visible] * ocean

    Shapes may be [B,1,H,W] or broadcast-compatible.
    """
    visible = visible_mask.to(
        dtype=noisy_center.dtype
    )
    ocean = ocean_mask.to(
        dtype=noisy_center.dtype
    )

    return (
        noisy_center
        * (1.0 - visible)
        + observed_center
        * visible
    ) * ocean


def make_center_mask_one_hot(
    visible_mask: torch.Tensor,
    ocean_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Build DIRECT central two-channel mask Mt.

    Channel 0: observed/visible ocean pixel.
    Channel 1: missing OR land.

    Returns [B,2,H,W].
    """
    visible = visible_mask.float()

    if ocean_mask is not None:
        visible = (
            visible
            * ocean_mask.float()
        )

    missing_or_land = (
        1.0 - visible
    )

    return torch.cat(
        [
            visible,
            missing_or_land,
        ],
        dim=1,
    )


def source_to_direct_one_hot(
    source: torch.Tensor,
) -> torch.Tensor:
    """
    Convert source offsets {0,1,2,3} to DIRECT's documented 3-channel order:

        source=1 -> [0,0,1]
        source=2 -> [0,1,0]
        source=3 -> [1,0,0]
        source=0 -> [0,0,0]

    Input:
        source [B,H,W] or [B,1,H,W]

    Output:
        [B,3,H,W]
    """
    if source.ndim == 4:
        if source.shape[1] != 1:
            raise ValueError(
                "4D source tensor must have one channel"
            )
        source = source[:, 0]

    if source.ndim != 3:
        raise ValueError(
            f"source must be [B,H,W] or [B,1,H,W], got {tuple(source.shape)}"
        )

    source = source.long()

    return torch.stack(
        [
            (source == 3).float(),
            (source == 2).float(),
            (source == 1).float(),
        ],
        dim=1,
    )


def build_temporal_masks(
    previous_source_mask: torch.Tensor,
    center_visible_mask: torch.Tensor,
    future_source_mask: torch.Tensor,
    ocean_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Build the 8 temporal-mask channels expected by DIRECT:

        0..2 previous source one-hot
        3..4 centre observed/missing one-hot
        5..7 future source one-hot
    """
    previous_oh = source_to_direct_one_hot(
        previous_source_mask
    )

    center_oh = make_center_mask_one_hot(
        visible_mask=center_visible_mask,
        ocean_mask=ocean_mask,
    )

    future_oh = source_to_direct_one_hot(
        future_source_mask
    )

    return torch.cat(
        [
            previous_oh,
            center_oh,
            future_oh,
        ],
        dim=1,
    )


# ---------------------------------------------------------------------
# Simple standalone sanity check
# ---------------------------------------------------------------------

def _sanity_check():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = DIRECT().to(device)

    # Use B=1 because that is the likely first safe training configuration.
    b, h, w = 1, 128, 128

    noisy_center = torch.randn(
        b,
        1,
        h,
        w,
        device=device,
    )

    temporal_context = torch.randn(
        b,
        3,
        h,
        w,
        device=device,
    )

    temporal_masks = torch.zeros(
        b,
        8,
        h,
        w,
        device=device,
    )

    doy = torch.tensor(
        [[0.5, -0.8660254]],
        dtype=torch.float32,
        device=device,
    )

    k = torch.tensor(
        [0.5],
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():
        output = model(
            noisy_center=noisy_center,
            temporal_context=temporal_context,
            temporal_masks=temporal_masks,
            doy=doy,
            k=k,
        )

    params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("DIRECT sanity check")
    print("-------------------")
    print("device:", device)
    print("parameters:", f"{params:,}")
    print("input:", (b, 12, h, w))
    print("output:", tuple(output.shape))
    print(
        "finite:",
        bool(torch.isfinite(output).all()),
    )

    if output.shape != (
        b,
        1,
        h,
        w,
    ):
        raise AssertionError(
            f"Unexpected output shape: {tuple(output.shape)}"
        )


if __name__ == "__main__":
    _sanity_check()
