"""Geometry-conditioned self-attention over electrodes.

This is the core architectural contribution: spatial self-attention with no
per-electrode parameters, conditioned instead on a geometric descriptor
`g_ij` computed from 3-D scalp coordinates. A model pretrained on one montage
applies the same function to any other montage given only the new electrodes'
coordinates.

Implements the G1/G2/G3 ablation from `project_direction_summary.md` §4 O1.

Design note on G1 (deviation from the literal GAT-style formulation in the
proposal). The proposal text gives G1 as
    e_{ij} = a^T LeakyReLU(W_a [W_Q x_i || W_K x_j || W_g g_{ij}])
which is additive attention in the style of GAT. I implement the
dot-product-plus-bias variant instead:
    e_{ij} = (W_Q x_i)^T (W_K x_j) / sqrt(d_h) + b_h(g_{ij})
because the headline comparison (geometric vs codex) needs both variants to
share the same attention kernel so the comparison isolates inductive-vs-
transductive geometry handling rather than confounding it with dot-product-
vs-additive attention. This is the same design choice T5/ALiBi make for
relative position bias. The literal GAT form is recoverable by swapping the
attention kernel; I have not implemented it.

Convention. Tokens are electrode patches at a single time step. The CLS
token is prepended; it has no geometry, so its row/column in the geometric
bias is zero. The encoder is applied independently per time step by the
caller (the temporal transformer comes later).
"""

from __future__ import annotations

import math
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


GeomInjection = Literal["g1", "g2", "g3", "none"]


# ---------------------------------------------------------------------------
# Geometric descriptor
# ---------------------------------------------------------------------------

def build_geometric_descriptor(ch_pos: torch.Tensor) -> torch.Tensor:
    """Build pairwise geometric descriptors g_ij from 3-D coordinates.

    Per proposal §3:
        g_ij = [ ||p_i - p_j|| ,  p_i - p_j ]  in R^4
    Asymmetric in (i, j) by design — the signed displacement lets attention
    express directional preferences (e.g. left-to-right hemisphere).

    Args:
        ch_pos: (M, 3) electrode coordinates, already normalized as in
            `preprocess.normalize_coordinates` (centroid-subtracted,
            unit-max-norm scaled). Either numpy or torch is fine.

    Returns:
        g: (M, M, 4) tensor of descriptors. g[i, j] = [dist, dx, dy, dz].
        Self-pairs g[i, i] are all zero.
    """
    if not isinstance(ch_pos, torch.Tensor):
        ch_pos = torch.as_tensor(ch_pos, dtype=torch.float32)
    if ch_pos.dim() != 2 or ch_pos.shape[1] != 3:
        raise ValueError(f"ch_pos must have shape (M, 3), got {tuple(ch_pos.shape)}")
    # (M, 1, 3) - (1, M, 3) -> (M, M, 3); displacement p_i - p_j with i on dim 0.
    disp = ch_pos.unsqueeze(1) - ch_pos.unsqueeze(0)
    dist = disp.norm(dim=-1, keepdim=True)  # (M, M, 1)
    g = torch.cat([dist, disp], dim=-1)  # (M, M, 4)
    return g


# ---------------------------------------------------------------------------
# Geometric multi-head attention
# ---------------------------------------------------------------------------

class GeometricMultiheadAttention(nn.Module):
    """Multi-head self-attention with optional geometric conditioning.

    Modes (from project_direction_summary.md §4 O1):
        - "g1": geometry enters the score as an additive per-head bias.
                e_{ij} = (W_Q x_i)^T (W_K x_j) / sqrt(d_h) + b_h(g_ij)
                where b_h is a shared MLP R^4 -> R^H.
        - "g2": geometry enters the value as a per-pair, per-head increment.
                v_{j -> i} = W_V x_j + W_{V,g}_h(g_ij)
        - "g3": both bias and value-increment.
        - "none": vanilla scaled dot-product attention. Used for the
                  transductive baseline (see TransductiveMultiheadAttention).

    The CLS token is handled by the caller: it should appear at index 0 of
    the token sequence, and the corresponding row/column of `geom_bias_table`
    / `geom_value_table` is zero (i.e. CLS has no geometric relationship
    with any electrode). This module does NOT inject the CLS; it receives
    a token sequence of length N = (1 + M) when CLS is in use.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        geometry_injection: GeomInjection = "g1",
        geom_mlp_hidden: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        if geometry_injection not in ("g1", "g2", "g3", "none"):
            raise ValueError(f"geometry_injection={geometry_injection!r} invalid")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.geometry_injection = geometry_injection
        self.dropout_p = dropout

        # Standard Q/K/V projections (shared with the transductive baseline).
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        # G1 / G3: per-head geometric attention bias. Small MLP R^4 -> R^H.
        if geometry_injection in ("g1", "g3"):
            self.geom_bias_mlp = nn.Sequential(
                nn.Linear(4, geom_mlp_hidden),
                nn.GELU(),
                nn.Linear(geom_mlp_hidden, n_heads),
            )
        else:
            self.geom_bias_mlp = None

        # G2 / G3: per-pair value increment. MLP R^4 -> R^{d_model}, reshaped
        # to (n_heads, d_head). Done per-pair, applied to message j -> i.
        if geometry_injection in ("g2", "g3"):
            self.geom_value_mlp = nn.Sequential(
                nn.Linear(4, geom_mlp_hidden),
                nn.GELU(),
                nn.Linear(geom_mlp_hidden, d_model),
            )
        else:
            self.geom_value_mlp = None

        self.dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    # Geometry tables — caller precomputes once per montage and reuses
    # ------------------------------------------------------------------

    def build_geom_bias_table(self, g: torch.Tensor) -> torch.Tensor:
        """Compute the (M, M, H) additive bias table from descriptor g.

        The caller should compute this once per montage and pad to (N, N, H)
        with zeros for the CLS row/column before passing to `forward`.
        Returns zeros for modes that don't use a score-side bias.
        """
        M = g.shape[0]
        if self.geom_bias_mlp is None:
            return g.new_zeros(M, M, self.n_heads)
        return self.geom_bias_mlp(g)  # (M, M, H)

    def build_geom_value_table(self, g: torch.Tensor) -> torch.Tensor:
        """Compute the (M, M, H, d_head) value-increment table from g.

        Caller pads CLS row/column to zero before passing to `forward`.
        Returns zeros for modes that don't use a value-side increment.
        """
        M = g.shape[0]
        if self.geom_value_mlp is None:
            return g.new_zeros(M, M, self.n_heads, self.d_head)
        v_inc = self.geom_value_mlp(g)  # (M, M, d_model)
        return v_inc.view(M, M, self.n_heads, self.d_head)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        geom_bias_table: Optional[torch.Tensor] = None,
        geom_value_table: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, d_model). Token sequence. If a CLS token is used it is
                already at index 0.
            geom_bias_table: (N, N, H) per-head additive bias, or None.
                Required for g1/g3; ignored for g2/none.
            geom_value_table: (N, N, H, d_head) per-pair value increment, or
                None. Required for g2/g3; ignored for g1/none.
            key_padding_mask: (B, N) boolean. True = mask this key out.
                Used for spatial masking during pretraining (proposal §3
                masked-patch reconstruction).

        Returns:
            y: (B, N, d_model)
        """
        B, N, _ = x.shape
        H, dh = self.n_heads, self.d_head

        q = self.q_proj(x).view(B, N, H, dh).transpose(1, 2)  # (B, H, N, dh)
        k = self.k_proj(x).view(B, N, H, dh).transpose(1, 2)
        v = self.v_proj(x).view(B, N, H, dh).transpose(1, 2)

        # Scores: (B, H, N_q, N_k)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(dh)

        if self.geometry_injection in ("g1", "g3"):
            if geom_bias_table is None:
                raise ValueError(
                    f"geometry_injection={self.geometry_injection!r} requires "
                    "geom_bias_table"
                )
            if geom_bias_table.shape != (N, N, H):
                raise ValueError(
                    f"geom_bias_table shape {tuple(geom_bias_table.shape)} "
                    f"!= expected (N={N}, N={N}, H={H})"
                )
            # (N, N, H) -> (1, H, N, N) for broadcast over batch.
            bias = geom_bias_table.permute(2, 0, 1).unsqueeze(0)
            scores = scores + bias

        if key_padding_mask is not None:
            # mask: (B, N) -> (B, 1, 1, N), broadcast over heads and queries.
            mask = key_padding_mask[:, None, None, :]
            scores = scores.masked_fill(mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        # If an entire row was masked out (e.g. mask hides every key), softmax
        # yields NaN. This shouldn't happen during training but defend anyway.
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        # Value path. Standard: out_h = attn @ v   -> (B, H, N, dh)
        out = torch.matmul(attn, v)

        if self.geometry_injection in ("g2", "g3"):
            if geom_value_table is None:
                raise ValueError(
                    f"geometry_injection={self.geometry_injection!r} requires "
                    "geom_value_table"
                )
            if geom_value_table.shape != (N, N, H, dh):
                raise ValueError(
                    f"geom_value_table shape {tuple(geom_value_table.shape)} "
                    f"!= expected (N={N}, N={N}, H={H}, dh={dh})"
                )
            # geom_value_table: (N_q, N_k, H, dh) -> (H, N_q, N_k, dh)
            # attn: (B, H, N_q, N_k); we want sum_k attn[b,h,i,k] * g_v[i,k,h,:]
            gv = geom_value_table.permute(2, 0, 1, 3)  # (H, N_q, N_k, dh)
            # einsum reads: b h i k , h i k d -> b h i d
            geom_msg = torch.einsum("bhik,hikd->bhid", attn, gv)
            out = out + geom_msg

        # (B, H, N, dh) -> (B, N, d_model)
        out = out.transpose(1, 2).contiguous().view(B, N, self.d_model)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Geometric Transformer encoder block (pre-norm, attention + MLP)
# ---------------------------------------------------------------------------

class GeometricEncoderBlock(nn.Module):
    """One pre-norm Transformer block with geometric attention."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        geometry_injection: GeomInjection = "g1",
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        geom_mlp_hidden: int = 32,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = GeometricMultiheadAttention(
            d_model=d_model,
            n_heads=n_heads,
            geometry_injection=geometry_injection,
            geom_mlp_hidden=geom_mlp_hidden,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(d_model)
        mlp_hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        geom_bias_table: Optional[torch.Tensor] = None,
        geom_value_table: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.norm1(x),
            geom_bias_table=geom_bias_table,
            geom_value_table=geom_value_table,
            key_padding_mask=key_padding_mask,
        )
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Full spatial encoder (stack of blocks + CLS handling + geometry padding)
# ---------------------------------------------------------------------------

class GeometricSpatialEncoder(nn.Module):
    """Stack of GeometricEncoderBlocks with CLS token and montage handling.

    Per project_direction_summary.md §3, defaults are L_S=2, H=8, d=256.

    Usage:
        enc = GeometricSpatialEncoder(d_model=256, n_heads=8, n_layers=2)
        # Once per montage:
        bias_tab, value_tab = enc.precompute_geom_tables(ch_pos)
        # Per forward pass (one time step at a time, or a flattened batch):
        s_t = enc(x_t, bias_tab, value_tab)   # CLS-pooled summary
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        geometry_injection: GeomInjection = "g1",
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        geom_mlp_hidden: int = 32,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.geometry_injection = geometry_injection

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.blocks = nn.ModuleList([
            GeometricEncoderBlock(
                d_model=d_model,
                n_heads=n_heads,
                geometry_injection=geometry_injection,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                geom_mlp_hidden=geom_mlp_hidden,
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    # ------------------------------------------------------------------
    # Geometry tables (precompute once per montage)
    # ------------------------------------------------------------------

    def precompute_geom_tables(self, ch_pos: torch.Tensor):
        """Build the bias and value tables, padded for the CLS token at index 0.

        Returns:
            bias_table: (N, N, H) where N = 1 + M. Row/col 0 = zeros (CLS).
            value_table: (N, N, H, d_head). Same padding scheme.

            For modes that don't use a table, the corresponding tensor is
            zeros and the attention module will ignore it.
        """
        if not isinstance(ch_pos, torch.Tensor):
            ch_pos = torch.as_tensor(ch_pos, dtype=torch.float32)
        ch_pos = ch_pos.to(self.cls_token.device, dtype=self.cls_token.dtype)
        g = build_geometric_descriptor(ch_pos)  # (M, M, 4)
        M = g.shape[0]
        H = self.n_heads
        dh = self.d_model // H

        # Both modules share weights across blocks of different types? No —
        # each block has its own attention MLP. So we compute the table by
        # querying the first block's attention module and trust that all
        # blocks use the same `geometry_injection` mode. But the MLPs DIFFER
        # per layer (they're separate nn.Modules), so we need a per-layer
        # table.
        bias_tables, value_tables = [], []
        for blk in self.blocks:
            attn = blk.attn
            b_inner = attn.build_geom_bias_table(g)  # (M, M, H)
            v_inner = attn.build_geom_value_table(g)  # (M, M, H, dh)
            # Pad to (N, N, ...) with N = M + 1; the CLS row/col is zero.
            N = M + 1
            b = b_inner.new_zeros(N, N, H)
            b[1:, 1:, :] = b_inner
            v = v_inner.new_zeros(N, N, H, dh)
            v[1:, 1:, :, :] = v_inner
            bias_tables.append(b)
            value_tables.append(v)
        # Return as lists; the forward pass indexes per-block.
        return bias_tables, value_tables

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        bias_tables: Optional[list] = None,
        value_tables: Optional[list] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode one time step.

        Args:
            x: (B, M, d_model). Patch features for M electrodes at one time step.
            bias_tables, value_tables: lists of length n_layers, from
                `precompute_geom_tables`. Required unless geometry_injection
                is "none".
            key_padding_mask: (B, M) electrode-level mask (True = masked).
                The CLS token is never masked.

        Returns:
            s: (B, d_model). CLS-pooled summary.
        """
        B, M, _ = x.shape
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        tokens = torch.cat([cls, x], dim=1)  # (B, M+1, d_model)

        if key_padding_mask is not None:
            cls_pad = key_padding_mask.new_zeros(B, 1)  # CLS never masked
            full_mask = torch.cat([cls_pad, key_padding_mask], dim=1)
        else:
            full_mask = None

        if self.geometry_injection == "none":
            bias_tables = [None] * len(self.blocks)
            value_tables = [None] * len(self.blocks)
        else:
            if bias_tables is None or value_tables is None:
                raise ValueError(
                    "bias_tables and value_tables required when "
                    f"geometry_injection={self.geometry_injection!r}"
                )
            if len(bias_tables) != len(self.blocks):
                raise ValueError(
                    f"got {len(bias_tables)} bias tables for "
                    f"{len(self.blocks)} blocks"
                )

        for blk, bt, vt in zip(self.blocks, bias_tables, value_tables):
            tokens = blk(
                tokens,
                geom_bias_table=bt,
                geom_value_table=vt,
                key_padding_mask=full_mask,
            )

        tokens = self.norm(tokens)
        return tokens[:, 0]  # CLS-pooled summary, (B, d_model)


# ---------------------------------------------------------------------------
# Smoke assertion block — run this file directly to sanity-check shapes
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    B, M, d = 2, 8, 64
    ch_pos = torch.randn(M, 3)
    enc = GeometricSpatialEncoder(
        d_model=d, n_heads=4, n_layers=2, geometry_injection="g1"
    )
    bts, vts = enc.precompute_geom_tables(ch_pos)
    x = torch.randn(B, M, d, requires_grad=True)
    s = enc(x, bts, vts)
    assert s.shape == (B, d), f"got {s.shape}"
    s.sum().backward()
    assert x.grad is not None
    print("geometric_attention.py smoke OK; output shape", tuple(s.shape))
