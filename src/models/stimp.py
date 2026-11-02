"""
STIMP / STDDM model for chlorophyll-a spatiotemporal imputation.

This implementation follows the architecture described in:
Zhang et al. (2025), "AI-powered spatiotemporal imputation and prediction
of chlorophyll-a concentration in coastal ecosystems".

Implemented components:
- value embedding of concatenated perturbed values X_l and partial observations X_ob
- sinusoidal diffusion-step embedding + two-layer SiLU projection
- Temporal Linear Transformer (TLT)
- Heterogeneous Spatial Graph Neural Network (HSGNN)
- GroupNorm residual spatial block
- CSDI-style gated activation
- imputation projection predicting clean normalized Chl-a X

Expected tensors
----------------
x_ob : [B, T, H, W] or [B, T, N]
x_l  : [B, T, H, W] or [B, T, N]
step : [B] diffusion step
coords: [B, H, W, 2], [H, W, 2], [B, N, 2], or [N, 2]
ocean_mask: [B, H, W], [H, W], [B, N], or [N]

The model internally flattens ocean pixels to graph nodes N and returns
the prediction in the same spatial layout as x_ob.

Important adaptation for the Mediterranean CMEMS dataset:
A dense N x N graph is not materialized. Graph aggregation is performed
with sparse edge_index connectivity constructed from geographic coordinates.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Diffusion-step embedding
# ---------------------------------------------------------------------

class SinusoidalDiffusionEmbedding(nn.Module):
    """Standard Transformer-style sinusoidal embedding for diffusion step l."""

    def __init__(self, dim: int):
        super().__init__()
        if dim < 4:
            raise ValueError("diffusion embedding dimension must be >= 4")
        self.dim = int(dim)

    def forward(self, step: torch.Tensor) -> torch.Tensor:
        if step.ndim == 0:
            step = step[None]
        step = step.float()

        half = self.dim // 2
        scale = math.log(10000.0) / max(half - 1, 1)
        frequencies = torch.exp(
            -scale * torch.arange(half, device=step.device, dtype=step.dtype)
        )
        angles = step[:, None] * frequencies[None, :]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


class DiffusionEmbeddingProjection(nn.Module):
    """
    Paper: PE(l) is projected with two fully connected layers and SiLU.
    """

    def __init__(self, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.pe = SinusoidalDiffusionEmbedding(embedding_dim)
        self.proj = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, step: torch.Tensor) -> torch.Tensor:
        return self.proj(self.pe(step))


# ---------------------------------------------------------------------
# Temporal Linear Transformer (TLT)
# ---------------------------------------------------------------------

class TemporalLinearTransformer(nn.Module):
    """
    Temporal dependency learning module.

    Self-attention is applied along T independently for every spatial node.
    Input/output: [B, T, N, D].

    The paper specifies MSA + position-wise feed-forward + normalization.
    PyTorch MultiheadAttention is used for the MSA implementation.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        ff_multiplier: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        ff_dim = hidden_dim * ff_multiplier
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # [B,T,N,D] -> [B*N,T,D]
        b, t, n, d = h.shape
        z = h.permute(0, 2, 1, 3).reshape(b * n, t, d)

        attn_out, _ = self.attn(
            z, z, z,
            need_weights=False,
        )
        z = self.norm1(z + attn_out)

        ff_out = self.ff(z)
        z = self.norm2(z + ff_out)

        return z.reshape(b, n, t, d).permute(0, 2, 1, 3).contiguous()


# ---------------------------------------------------------------------
# Sparse graph utilities
# ---------------------------------------------------------------------

def build_spatial_edges(
    coords: torch.Tensor,
    node_mask: Optional[torch.Tensor] = None,
    threshold_deg: float = 0.05,
    include_self: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build sparse graph edges following the STIMP paper's default rule:

        G_ij = 1 when both |lat_i-lat_j| < threshold
        and |lon_i-lon_j| < threshold.

    For our regular 1-km CMEMS patches, we avoid constructing dense G.

    Returns
    -------
    edge_index : LongTensor [2, E]
        source -> destination edges in the compact valid-node indexing.
    valid_flat_indices : LongTensor [N]
        positions of graph nodes in the original flattened H*W grid.

    Notes
    -----
    This routine is intended to be called once per spatial patch/grid and
    cached by the training script, not rebuilt every optimization step.
    """

    if coords.ndim == 3:
        coords = coords.reshape(-1, 2)
    elif coords.ndim != 2:
        raise ValueError("coords must have shape [H,W,2] or [N,2]")

    if node_mask is None:
        valid = torch.ones(coords.shape[0], dtype=torch.bool, device=coords.device)
    else:
        valid = node_mask.reshape(-1).bool().to(coords.device)

    valid &= torch.isfinite(coords).all(dim=-1)
    valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
    c = coords[valid_idx]

    if c.numel() == 0:
        raise ValueError("No valid graph nodes found.")

    # Coordinates are on a regular grid, so use sorted unique lat/lon values
    # to generate nearby candidate offsets instead of O(N^2) dense distances.
    lat = c[:, 0]
    lon = c[:, 1]

    # General sparse construction using bins of threshold width.
    # Candidate comparisons occur only within neighboring coordinate bins.
    bin_size = float(threshold_deg)
    lat_bin = torch.floor(lat / bin_size).long()
    lon_bin = torch.floor(lon / bin_size).long()

    # CPU dictionaries are cheap here because graph construction is one-time.
    lat_cpu = lat.detach().cpu()
    lon_cpu = lon.detach().cpu()
    lb_cpu = lat_bin.detach().cpu()
    lob_cpu = lon_bin.detach().cpu()

    bins = {}
    for i in range(c.shape[0]):
        key = (int(lb_cpu[i]), int(lob_cpu[i]))
        bins.setdefault(key, []).append(i)

    src = []
    dst = []

    for i in range(c.shape[0]):
        bi = int(lb_cpu[i])
        bj = int(lob_cpu[i])

        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                candidates = bins.get((bi + di, bj + dj), ())
                for j in candidates:
                    if not include_self and i == j:
                        continue
                    if (
                        abs(float(lat_cpu[i] - lat_cpu[j])) < threshold_deg
                        and abs(float(lon_cpu[i] - lon_cpu[j])) < threshold_deg
                    ):
                        src.append(j)
                        dst.append(i)

    edge_index = torch.tensor(
        [src, dst],
        dtype=torch.long,
        device=coords.device,
    )

    return edge_index, valid_idx


def normalized_neighbor_aggregate(
    h: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """
    Sparse mean aggregation corresponding to multiplication by a
    row-normalized spatial adjacency.

    h: [B,T,N,D]
    edge_index: [2,E], source -> destination
    """
    if edge_index.numel() == 0:
        return h

    b, t, n, d = h.shape
    src, dst = edge_index

    # Flatten B and T while retaining N.
    z = h.reshape(b * t, n, d)
    out = torch.zeros_like(z)

    # index_add over graph nodes.
    out.index_add_(1, dst, z[:, src, :])

    degree = torch.zeros(n, device=h.device, dtype=h.dtype)
    degree.index_add_(
        0,
        dst,
        torch.ones(dst.numel(), device=h.device, dtype=h.dtype),
    )
    degree = degree.clamp_min(1.0)

    out = out / degree[None, :, None]
    return out.reshape(b, t, n, d)


# ---------------------------------------------------------------------
# Heterogeneous Spatial Graph Neural Network (HSGNN)
# ---------------------------------------------------------------------

class HeterogeneousSpatialGNN(nn.Module):
    """
    HSGNN adaptation following Eqs. 22-25 of the STIMP paper.

    The paper creates location-specific parameters from a small prototype
    parameter pool P. Query Q is determined from:
        location, temporal mean, temporal variance.

    Here "location" is represented by normalized latitude and longitude,
    giving the same geographic information while being suitable for the
    2-D Mediterranean grid.

    Input/output: [B,T,N,D].
    """

    def __init__(
        self,
        hidden_dim: int,
        num_prototypes: int = 4,
    ):
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.num_prototypes = int(num_prototypes)

        # Q is produced from [lat, lon, mean, variance].
        self.query = nn.Linear(4, num_prototypes)

        # P in R^{k x d x d}
        self.prototype_pool = nn.Parameter(
            torch.empty(num_prototypes, hidden_dim, hidden_dim)
        )
        nn.init.xavier_uniform_(self.prototype_pool)

        self.spatial_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        h: torch.Tensor,
        coords: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:

        b, t, n, d = h.shape

        if coords.ndim == 2:
            coords = coords.unsqueeze(0).expand(b, -1, -1)
        if coords.shape[:2] != (b, n):
            raise ValueError(
                f"coords must be [B,N,2] or [N,2]; got {tuple(coords.shape)} "
                f"for h={tuple(h.shape)}"
            )

        coords = coords.to(device=h.device, dtype=h.dtype)

        # Normalize geographic coordinates per sample.
        coord_mean = coords.mean(dim=1, keepdim=True)
        coord_std = coords.std(dim=1, keepdim=True).clamp_min(1e-6)
        coord_features = (coords - coord_mean) / coord_std

        # Spatial heterogeneity statistics across the time dimension.
        temporal_mean = h.mean(dim=1).mean(dim=-1, keepdim=True)
        temporal_var = h.var(
            dim=1,
            unbiased=False,
        ).mean(dim=-1, keepdim=True)

        query_features = torch.cat(
            [coord_features, temporal_mean, temporal_var],
            dim=-1,
        )  # [B,N,4]

        q = torch.softmax(self.query(query_features), dim=-1)  # [B,N,K]

        # W_spa = Q^T P conceptually: each node receives its own D x D matrix.
        w_spa = torch.einsum(
            "bnk,kde->bnde",
            q,
            self.prototype_pool,
        )  # [B,N,D,D]

        # H_temp * G: sparse graph aggregation first.
        agg = normalized_neighbor_aggregate(h, edge_index)

        # Apply location-specific parameters.
        spatial = torch.einsum(
            "btnd,bnde->btne",
            agg,
            w_spa,
        )

        return self.spatial_projection(spatial)


# ---------------------------------------------------------------------
# Group normalization over latent channels
# ---------------------------------------------------------------------

class LatentGroupNorm(nn.Module):
    def __init__(self, hidden_dim: int, num_groups: int = 8):
        super().__init__()

        groups = min(num_groups, hidden_dim)
        while hidden_dim % groups != 0 and groups > 1:
            groups -= 1

        self.norm = nn.GroupNorm(groups, hidden_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # [B,T,N,D] -> GroupNorm over D
        z = h.permute(0, 3, 1, 2)
        z = self.norm(z)
        return z.permute(0, 2, 3, 1).contiguous()


# ---------------------------------------------------------------------
# STDDM
# ---------------------------------------------------------------------

class STDDM(nn.Module):
    """
    Spatiotemporal Denoising Diffusion Model used by STIMP for imputation.

    This network predicts the clean normalized Chl-a X rather than noise.

    Paper flow:
        concat(X_l, X_ob)
            -> 1x1 value embedding
            + diffusion-step embedding
            -> TLT
            -> GN(TLT + HSGNN(TLT, G))
            -> sigmoid(conv) * tanh(conv)
            -> imputation projection
            -> X_hat
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        diffusion_embedding_dim: int = 128,
        num_heads: int = 4,
        ff_multiplier: int = 2,
        num_prototypes: int = 4,
        group_norm_groups: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.hidden_dim = int(hidden_dim)

        # 1x1 convolution over the two input values [X_l, X_ob].
        # Linear is mathematically the same point-wise operation once spatial
        # locations have been flattened into nodes.
        self.value_embedding = nn.Linear(2, hidden_dim)

        self.diffusion_embedding = DiffusionEmbeddingProjection(
            diffusion_embedding_dim,
            hidden_dim,
        )

        self.tlt = TemporalLinearTransformer(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            ff_multiplier=ff_multiplier,
            dropout=dropout,
        )

        self.hsgnn = HeterogeneousSpatialGNN(
            hidden_dim=hidden_dim,
            num_prototypes=num_prototypes,
        )

        self.group_norm = LatentGroupNorm(
            hidden_dim=hidden_dim,
            num_groups=group_norm_groups,
        )

        # CSDI-style gated activation from Eq. 18.
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.filter = nn.Linear(hidden_dim, hidden_dim)

        # F_imp
        self.imputation_projection = nn.Linear(hidden_dim, 1)

    def forward_nodes(
        self,
        x_ob: torch.Tensor,
        x_l: torch.Tensor,
        diffusion_step: torch.Tensor,
        coords: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Node-form forward pass.

        x_ob, x_l: [B,T,N]
        coords: [B,N,2] or [N,2]
        output: [B,T,N]
        """

        if x_ob.shape != x_l.shape:
            raise ValueError(
                f"x_ob and x_l must have identical shape, got "
                f"{tuple(x_ob.shape)} and {tuple(x_l.shape)}"
            )
        if x_ob.ndim != 3:
            raise ValueError("forward_nodes expects [B,T,N] tensors")

        inp = torch.stack([x_l, x_ob], dim=-1)  # [B,T,N,2]
        h = self.value_embedding(inp)

        step_emb = self.diffusion_embedding(diffusion_step)
        h = h + step_emb[:, None, None, :]

        h_temp = self.tlt(h)
        h_spatial = self.hsgnn(h_temp, coords, edge_index)

        h_spa = self.group_norm(h_temp + h_spatial)

        h_out = torch.sigmoid(self.gate(h_spa)) * torch.tanh(
            self.filter(h_spa)
        )

        return self.imputation_projection(h_out).squeeze(-1)

    def forward(
        self,
        x_ob: torch.Tensor,
        x_l: torch.Tensor,
        diffusion_step: torch.Tensor,
        coords: torch.Tensor,
        ocean_mask: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Patch-form convenience forward pass.

        x_ob/x_l can be [B,T,H,W] or [B,T,N].

        For [B,T,H,W], only ocean nodes are passed through the graph model;
        output is restored to [B,T,H,W]. Land remains zero.

        For [B,T,N], all N nodes are assumed to correspond to coords.
        """

        if x_ob.ndim == 3:
            # Already [B,T,N]
            b, t, n = x_ob.shape

            if coords.ndim == 2:
                node_coords = coords
            elif coords.ndim == 3:
                node_coords = coords
            else:
                raise ValueError("coords incompatible with node-form input")

            if edge_index is None:
                base_coords = node_coords[0] if node_coords.ndim == 3 else node_coords
                base_mask = (
                    ocean_mask[0] if ocean_mask.ndim == 2 else ocean_mask
                )
                edge_index, valid_idx = build_spatial_edges(
                    base_coords,
                    base_mask,
                )
                if valid_idx.numel() != n:
                    raise ValueError(
                        "For node-form input, ocean_mask must describe exactly "
                        "the supplied N nodes."
                    )

            return self.forward_nodes(
                x_ob=x_ob,
                x_l=x_l,
                diffusion_step=diffusion_step,
                coords=node_coords,
                edge_index=edge_index.to(x_ob.device),
            )

        if x_ob.ndim != 4:
            raise ValueError(
                "x_ob must have shape [B,T,H,W] or [B,T,N]"
            )

        if x_ob.shape != x_l.shape:
            raise ValueError("x_ob and x_l must have the same shape")

        b, t, h, w = x_ob.shape

        # stimp_data.py returns the same fixed ocean mask for a given patch
        # across all sequence timesteps.
        if ocean_mask.ndim == 4:
            ocean_mask = ocean_mask[:, 0]
        if ocean_mask.ndim == 2:
            ocean_mask = ocean_mask.unsqueeze(0).expand(b, -1, -1)

        # DataLoader batches coords as [B,H,W,2].
        if coords.ndim == 3:
            coords = coords.unsqueeze(0).expand(b, -1, -1, -1)

        flat_mask = ocean_mask[0].reshape(-1).bool()
        valid_idx = torch.nonzero(flat_mask, as_tuple=False).squeeze(-1)

        if valid_idx.numel() == 0:
            return torch.zeros_like(x_ob)

        xob_flat = x_ob.reshape(b, t, h * w)[:, :, valid_idx]
        xl_flat = x_l.reshape(b, t, h * w)[:, :, valid_idx]

        coords_flat = coords.reshape(b, h * w, 2)[:, valid_idx, :]

        if edge_index is None:
            # Correct but slower. Training should precompute/cache edge_index
            # per patch geometry whenever possible.
            edge_index, _ = build_spatial_edges(
                coords[0],
                ocean_mask[0],
            )

        pred_nodes = self.forward_nodes(
            x_ob=xob_flat,
            x_l=xl_flat,
            diffusion_step=diffusion_step,
            coords=coords_flat,
            edge_index=edge_index.to(x_ob.device),
        )

        out = torch.zeros(
            (b, t, h * w),
            device=x_ob.device,
            dtype=pred_nodes.dtype,
        )
        out[:, :, valid_idx] = pred_nodes

        return out.reshape(b, t, h, w)


# ---------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------

def masked_stddm_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Eq. 8-style MSE: optimize only artificially hidden known observations.
    """
    mask = loss_mask.to(dtype=prediction.dtype)

    denom = mask.sum()
    if denom <= 0:
        # Keep graph connected to prediction while contributing zero gradient.
        return prediction.sum() * 0.0

    return (((prediction - target) ** 2) * mask).sum() / denom


# ---------------------------------------------------------------------
# Lightweight sanity check
# ---------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    # Small synthetic example; the real 128x128 graph should be exercised
    # through train_stimp.py with stimp_data.py.
    b, t, h, w = 2, 15, 8, 8

    x_ob = torch.randn(b, t, h, w)
    x_l = torch.randn_like(x_ob)
    ocean = torch.ones(b, h, w)

    lat = torch.linspace(40.0, 40.07, h)
    lon = torch.linspace(15.0, 15.07, w)
    lat_grid, lon_grid = torch.meshgrid(lat, lon, indexing="ij")
    coords = torch.stack([lat_grid, lon_grid], dim=-1)
    coords = coords.unsqueeze(0).expand(b, -1, -1, -1).contiguous()

    step = torch.tensor([10, 25], dtype=torch.long)

    model = STDDM(
        hidden_dim=32,
        diffusion_embedding_dim=64,
        num_heads=4,
        num_prototypes=4,
    )

    y = model(
        x_ob=x_ob,
        x_l=x_l,
        diffusion_step=step,
        coords=coords,
        ocean_mask=ocean,
    )

    print("STDDM sanity check")
    print("------------------")
    print("x_ob:", tuple(x_ob.shape))
    print("x_l:", tuple(x_l.shape))
    print("output:", tuple(y.shape))
    print("finite:", bool(torch.isfinite(y).all()))
    print("parameters:", f"{sum(p.numel() for p in model.parameters()):,}")

    assert y.shape == x_ob.shape
    assert torch.isfinite(y).all()

    print("Sanity checks passed.")
