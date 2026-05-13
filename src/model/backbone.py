"""Full encoder backbone: patching + linear projection + spatial encoder + temporal Transformer.

This is the composition of components that feeds the alignment loss (and,
when the momentum encoder is wired up next session, the EMA target). The
architecture follows proposal section 3:

    raw EEG (B, M, T)
        |
        +-> patcher (non-overlapping 250 ms windows by default)
        |       -> (B, M, T_p, patch_samples)
        |
        +-> linear projection (per-patch -> d_model, ELECTRODE-AGNOSTIC)
        |       -> (B, M, T_p, d_model)
        |
        +-> spatial encoder (per-time-step, CLS-pooled)
        |       -> (B, T_p, d_model)              [summaries S_t]
        |
        +-> temporal Transformer (positional + L_T blocks)
                -> (B, T_p, d_model)              [contextualized summaries]

Design notes
------------
1. The linear projection is electrode-agnostic by design. The transductive
   variant's codex add lives INSIDE `TransductiveSpatialEncoder.forward`,
   not at projection time (confirmed session 5). So the same projection
   weights apply to every electrode patch, regardless of which spatial
   encoder is wired up downstream.

2. The spatial encoder is called over a flattened (B * T_p) batch rather
   than in a Python loop over t. Both `GeometricSpatialEncoder.forward`
   and `TransductiveSpatialEncoder.forward` only care about the leading
   batch dim, and the geometry tables are independent of t (geometry is
   constant within a montage), so they can be shared across all time steps.

3. The temporal Transformer uses learned positional embeddings by default
   (proposal section 3). `sinusoidal` and `none` are reserved options
   threaded through ArchConfig.

4. The temporal Transformer is a vanilla pre-norm Transformer encoder.
   Implementing it in this file rather than importing `nn.TransformerEncoder`
   is a deliberate choice -- the spatial encoders are hand-rolled for
   geometry-conditioning, and matching the style makes the FLOPs/parameter
   accounting consistent across the model.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.geometric_attention import GeometricSpatialEncoder
from src.model.transductive_baseline import TransductiveSpatialEncoder


SpatialEncoder = Union[GeometricSpatialEncoder, TransductiveSpatialEncoder]


# ---------------------------------------------------------------------------
# Patcher
# ---------------------------------------------------------------------------

class Patcher(nn.Module):
    """Non-overlapping (by default) time-axis patcher.

    Input:  (B, M, T)
    Output: (B, M, T_p, patch_samples)

    Implemented via `Tensor.unfold` so arbitrary stride is supported. With
    the proposal-default stride == patch_samples == 50 (at 200 Hz, i.e.
    250 ms), an epoch of T=800 samples yields T_p=16 patches.

    The patcher has no parameters. It's a module so it shows up in the
    model graph and can be replaced (e.g. with a learned conv-stem patcher)
    without touching the Backbone forward.
    """

    def __init__(self, patch_samples: int, patch_stride: int):
        super().__init__()
        if patch_samples <= 0 or patch_stride <= 0:
            raise ValueError(
                f"patch_samples and patch_stride must be positive: "
                f"got {patch_samples}, {patch_stride}"
            )
        self.patch_samples = patch_samples
        self.patch_stride = patch_stride

    def n_patches(self, T: int) -> int:
        if T < self.patch_samples:
            return 0
        return 1 + (T - self.patch_samples) // self.patch_stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, M, T)
        Returns:
            patches: (B, M, T_p, patch_samples)
        """
        if x.dim() != 3:
            raise ValueError(f"expected (B, M, T), got shape {tuple(x.shape)}")
        T = x.shape[-1]
        T_p = self.n_patches(T)
        if T_p == 0:
            raise ValueError(
                f"T={T} < patch_samples={self.patch_samples}; no patches"
            )
        # Tensor.unfold(dim, size, step): adds a new last dim of `size`
        # by sliding `step`-strided windows along `dim`.
        # (B, M, T) -unfold(2, P, S)-> (B, M, T_p, P)
        return x.unfold(dimension=-1, size=self.patch_samples, step=self.patch_stride)


# ---------------------------------------------------------------------------
# Temporal positional embedding
# ---------------------------------------------------------------------------

class TemporalPositionalEmbedding(nn.Module):
    """Positional embedding for the temporal Transformer.

    Modes:
        - "learned"    : nn.Parameter of shape (max_len, d_model). Default.
        - "sinusoidal" : fixed sin/cos schedule from Vaswani et al. Reserved
                         for a future ablation; included so the ArchConfig
                         option is functional.
        - "none"       : no positional information; useful as a sanity check
                         that the temporal Transformer is actually using
                         position info.

    A `max_len` larger than T_p is fine -- only the first T_p rows are used.
    """

    def __init__(self, mode: str, d_model: int, max_len: int = 64):
        super().__init__()
        if mode not in {"learned", "sinusoidal", "none"}:
            raise ValueError(f"unknown mode: {mode}")
        self.mode = mode
        self.d_model = d_model
        self.max_len = max_len

        if mode == "learned":
            self.pe = nn.Parameter(torch.zeros(max_len, d_model))
            nn.init.trunc_normal_(self.pe, std=0.02)
        elif mode == "sinusoidal":
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float32)
                * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe, persistent=False)
        # "none" stores nothing.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T_p, d_model)
        Returns:
            x + positional embedding (or x unchanged if mode == "none")
        """
        if self.mode == "none":
            return x
        T_p = x.shape[1]
        if T_p > self.max_len:
            raise ValueError(
                f"sequence length {T_p} exceeds positional max_len={self.max_len}; "
                f"bump max_len in ArchConfig or shorten the epoch"
            )
        return x + self.pe[:T_p].unsqueeze(0)


# ---------------------------------------------------------------------------
# Vanilla temporal Transformer block (pre-norm, no geometry)
# ---------------------------------------------------------------------------

class _TemporalAttention(nn.Module):
    """Sibling of `PlainMultiheadAttention` in transductive_baseline, kept
    local to this file so the Backbone is a single import. Vanilla scaled
    dot-product self-attention with optional time-step masking.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, _ = x.shape
        H, dh = self.n_heads, self.d_head
        q = self.q_proj(x).view(B, N, H, dh).transpose(1, 2)
        k = self.k_proj(x).view(B, N, H, dh).transpose(1, 2)
        v = self.v_proj(x).view(B, N, H, dh).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(dh)
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, self.d_model)
        return self.out_proj(out)


class _TemporalBlock(nn.Module):
    """Pre-norm temporal Transformer block."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = _TemporalAttention(d_model, n_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), key_padding_mask=key_padding_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class TemporalTransformer(nn.Module):
    """Stack of `n_layers` pre-norm Transformer blocks over time-step tokens."""

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        posemb_mode: str = "learned",
        max_len: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.posemb = TemporalPositionalEmbedding(posemb_mode, d_model, max_len=max_len)
        self.blocks = nn.ModuleList([
            _TemporalBlock(d_model, n_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T_p, d_model). Per-time-step CLS summaries from the spatial encoder.
            key_padding_mask: (B, T_p) bool. True = mask. Used for temporal
                masking during pretraining (proposal section 4 O4).
        Returns:
            y: (B, T_p, d_model). Contextualized time-step summaries.
        """
        x = self.posemb(x)
        for blk in self.blocks:
            x = blk(x, key_padding_mask=key_padding_mask)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Backbone -- the composition
# ---------------------------------------------------------------------------

class Backbone(nn.Module):
    """End-to-end encoder: raw EEG -> contextualized time-step summaries.

    Takes a spatial encoder instance as a dependency (either
    `GeometricSpatialEncoder` or `TransductiveSpatialEncoder`) so the same
    Backbone class wires up both the geometric and transductive variants.
    Build it like:

        # Geometric variant (default):
        spatial = GeometricSpatialEncoder(
            d_model=arch.d_model, n_heads=arch.n_heads_spatial,
            n_layers=arch.n_layers_spatial,
            geometry_injection="g1",
            geom_mlp_hidden=arch.geom_mlp_hidden, dropout=arch.dropout,
        )
        backbone = Backbone(spatial, arch=arch)
        bias_tabs, val_tabs = backbone.precompute_geom_tables(ch_pos)
        y = backbone(x, bias_tabs, val_tabs)         # (B, T_p, d_model)

        # Transductive variant:
        spatial = TransductiveSpatialEncoder(
            d_model=arch.d_model, n_heads=arch.n_heads_spatial,
            n_layers=arch.n_layers_spatial, n_electrodes=M,
            dropout=arch.dropout,
        )
        backbone = Backbone(spatial, arch=arch)
        y = backbone(x)                              # (B, T_p, d_model)
    """

    def __init__(self, spatial_encoder: SpatialEncoder, arch):
        super().__init__()
        self.spatial = spatial_encoder
        self.arch = arch

        # Sanity: the spatial encoder's d_model must agree with arch.d_model
        # (the projection has to feed it).
        if spatial_encoder.d_model != arch.d_model:
            raise ValueError(
                f"spatial_encoder.d_model={spatial_encoder.d_model} != "
                f"arch.d_model={arch.d_model}"
            )

        self.patcher = Patcher(arch.patch_samples, arch.patch_stride)

        # Linear projection: per-patch (patch_samples -> d_model), shared
        # across electrodes. Electrode-agnostic by design (see module
        # docstring).
        self.proj = nn.Linear(arch.patch_samples, arch.d_model, bias=True)

        # Temporal Transformer over the (B, T_p, d_model) sequence of
        # per-time-step CLS summaries.
        self.temporal = TemporalTransformer(
            d_model=arch.d_model,
            n_heads=arch.n_heads_temporal,
            n_layers=arch.n_layers_temporal,
            mlp_ratio=arch.mlp_ratio,
            dropout=arch.dropout,
            posemb_mode=arch.temporal_posemb,
        )

    # ------------------------------------------------------------------
    # Capability flags / delegation helpers
    # ------------------------------------------------------------------

    @property
    def uses_geometry(self) -> bool:
        return isinstance(self.spatial, GeometricSpatialEncoder)

    def precompute_geom_tables(self, ch_pos: torch.Tensor):
        """Delegate to the spatial encoder when it expects geometry tables.

        For the transductive variant this raises -- there are no tables to
        precompute, and the caller shouldn't be asking. The asymmetry is
        intentional and matches the asymmetry baked into the architecture
        comparison.
        """
        if not self.uses_geometry:
            raise RuntimeError(
                "precompute_geom_tables called on a Backbone wired with "
                "TransductiveSpatialEncoder; the transductive variant has "
                "no geometry tables. This is the expected asymmetry."
            )
        return self.spatial.precompute_geom_tables(ch_pos)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        bias_tables: Optional[list] = None,
        value_tables: Optional[list] = None,
        spatial_mask: Optional[torch.Tensor] = None,
        temporal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, M, T). Raw (preprocessed) EEG.
            bias_tables, value_tables: lists of length n_layers_spatial from
                `precompute_geom_tables`. Required for the geometric variant,
                must be None for the transductive variant.
            spatial_mask: (B, M) bool, True = mask out this electrode. Same
                mask is applied at every time step. (Per-time-step varying
                masks are supported by reshaping to (B*T_p, M), but the
                default proposal masking scheme is consistent across t.)
            temporal_mask: (B, T_p) bool, True = mask out this time step at
                the temporal Transformer.

        Returns:
            y: (B, T_p, d_model). Contextualized per-time-step summaries.
        """
        if x.dim() != 3:
            raise ValueError(f"expected (B, M, T), got {tuple(x.shape)}")
        B, M, T = x.shape

        # 1. Patch:  (B, M, T) -> (B, M, T_p, P)
        patches = self.patcher(x)
        T_p = patches.shape[2]

        # 2. Project per patch:  (B, M, T_p, P) -> (B, M, T_p, d_model)
        feats = self.proj(patches)

        # 3. Spatial encoder over each time step.
        # Reshape (B, M, T_p, d) -> (B*T_p, M, d) so the encoder runs once
        # over a (B*T_p) batch. This is faster than a Python loop over t
        # and works for both spatial encoders (they only care about the
        # leading batch dim).
        d = feats.shape[-1]
        # (B, M, T_p, d) -> (B, T_p, M, d) -> (B*T_p, M, d)
        feats_bt = feats.permute(0, 2, 1, 3).reshape(B * T_p, M, d)

        if spatial_mask is not None:
            if spatial_mask.shape != (B, M):
                raise ValueError(
                    f"spatial_mask shape {tuple(spatial_mask.shape)} != ({B}, {M})"
                )
            # Broadcast the same electrode mask across all T_p time steps.
            sp_mask_bt = spatial_mask.unsqueeze(1).expand(B, T_p, M).reshape(B * T_p, M)
        else:
            sp_mask_bt = None

        if self.uses_geometry:
            if bias_tables is None or value_tables is None:
                raise ValueError(
                    "geometric Backbone requires bias_tables and value_tables; "
                    "call precompute_geom_tables(ch_pos) once per montage"
                )
            s_bt = self.spatial(
                feats_bt,
                bias_tables=bias_tables,
                value_tables=value_tables,
                key_padding_mask=sp_mask_bt,
            )
        else:
            if bias_tables is not None or value_tables is not None:
                raise ValueError(
                    "transductive Backbone does not accept geometry tables"
                )
            s_bt = self.spatial(feats_bt, key_padding_mask=sp_mask_bt)

        # s_bt: (B*T_p, d) -> (B, T_p, d)
        s = s_bt.view(B, T_p, d)

        # 4. Temporal Transformer.
        y = self.temporal(s, key_padding_mask=temporal_mask)
        return y


# ---------------------------------------------------------------------------
# Smoke assertion block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.config import Config

    torch.manual_seed(0)
    cfg = Config()
    arch = cfg.arch

    # Tiny config for the smoke run.
    M, T = 8, 800
    B = 2
    ch_pos = torch.randn(M, 3)
    ch_pos = ch_pos - ch_pos.mean(0, keepdim=True)
    ch_pos = ch_pos / ch_pos.norm(dim=-1).max().clamp_min(1e-8)

    # --- Geometric Backbone ---
    spatial_g = GeometricSpatialEncoder(
        d_model=arch.d_model,
        n_heads=arch.n_heads_spatial,
        n_layers=arch.n_layers_spatial,
        geometry_injection="g1",
        geom_mlp_hidden=arch.geom_mlp_hidden,
        dropout=arch.dropout,
    )
    bb_g = Backbone(spatial_g, arch=arch)
    bt, vt = bb_g.precompute_geom_tables(ch_pos)
    x = torch.randn(B, M, T, requires_grad=True)
    y = bb_g(x, bt, vt)
    assert y.shape == (B, arch.n_patches(T), arch.d_model), y.shape
    y.sum().backward()
    assert x.grad is not None
    print(f"geometric Backbone smoke OK; output shape {tuple(y.shape)}")

    # --- Transductive Backbone ---
    spatial_t = TransductiveSpatialEncoder(
        d_model=arch.d_model,
        n_heads=arch.n_heads_spatial,
        n_layers=arch.n_layers_spatial,
        n_electrodes=M,
        dropout=arch.dropout,
    )
    bb_t = Backbone(spatial_t, arch=arch)
    x2 = torch.randn(B, M, T, requires_grad=True)
    y2 = bb_t(x2)
    assert y2.shape == (B, arch.n_patches(T), arch.d_model), y2.shape
    y2.sum().backward()
    assert x2.grad is not None
    print(f"transductive Backbone smoke OK; output shape {tuple(y2.shape)}")
