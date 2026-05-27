"""v2 geometric attention: 10-D descriptor, per-electrode tokens (no CLS pool).

Two changes vs v1's `src/model/geometric_attention.py`:

1. **Descriptor expanded from R^4 to R^10.** v1 fed
   ``g_ij = [||p_i - p_j||, p_i - p_j]`` to the geometric MLP, which is
   translation-invariant by construction. The trained v1 ``b`` collapsed
   to a per-head linear ramp in distance -- diagnostic in v2_scope.md.
   v2.0 feeds

       g_ij = [p_i, p_j, p_i - p_j, ||p_i - p_j||]  in R^{3+3+3+1} = R^10

   which is a strict superset: the MLP can recover v1's behavior by
   zeroing the weights on the first six dims, or it can learn
   position-conditional structure (e.g. frontal pairs behave differently
   from equidistant occipital pairs) if the training signal supports it.
   Coordinates must be in a shared frame (`src/v2/preprocess.py`).

2. **No CLS pool.** v1's spatial encoder prepended a CLS token and
   returned only the CLS row, collapsing the electrode dimension. v2.1
   returns the full per-electrode token sequence ``(B, M, d)`` so the
   downstream loss can align per-channel.

Everything else (g1/g2/g3 modes, geometry-table precomputation, pre-norm
blocks) mirrors v1. The descriptor builder is in this file; the spatial
encoder is the natural ``cls=False`` counterpart of v1's.
"""

from __future__ import annotations

import math
from typing import List, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


GeomInjection = Literal["g1", "g2", "g3", "none"]

# Width of the v2 descriptor.
G_DIM_V2 = 10


# ---------------------------------------------------------------------------
# Geometric descriptor (v2: 10-D)
# ---------------------------------------------------------------------------

def build_geometric_descriptor_v2(ch_pos: torch.Tensor) -> torch.Tensor:
    """Build pairwise 10-D geometric descriptors from 3-D coordinates.

    ``g_ij = [p_i, p_j, p_i - p_j, ||p_i - p_j||]``

    The first six dimensions break translation invariance: the same
    relative displacement at the frontal vs occipital scalp produces
    different ``g_ij`` values, so ``b`` can learn position-conditional
    biases. Recovers v1 if the MLP's first-layer weights on dims 0..5
    are zero.

    Args:
        ch_pos: (M, 3) coordinates in a shared frame
            (see `src/v2/preprocess.py`). Either numpy or torch is fine.

    Returns:
        g: (M, M, 10). g[i, j] = [p_i, p_j, p_i - p_j, ||p_i - p_j||].
        Self-pairs g[i, i] have zero displacement and zero distance but
        nonzero p_i / p_j components.
    """
    if not isinstance(ch_pos, torch.Tensor):
        ch_pos = torch.as_tensor(ch_pos, dtype=torch.float32)
    if ch_pos.dim() != 2 or ch_pos.shape[1] != 3:
        raise ValueError(f"ch_pos must have shape (M, 3), got {tuple(ch_pos.shape)}")
    M = ch_pos.shape[0]
    p_i = ch_pos.unsqueeze(1).expand(M, M, 3)   # (M, M, 3)
    p_j = ch_pos.unsqueeze(0).expand(M, M, 3)   # (M, M, 3)
    disp = p_i - p_j                            # (M, M, 3)
    dist = disp.norm(dim=-1, keepdim=True)      # (M, M, 1)
    g = torch.cat([p_i, p_j, disp, dist], dim=-1)  # (M, M, 10)
    return g


# ---------------------------------------------------------------------------
# Multi-head attention with geometric conditioning (v2: 10-D MLP inputs)
# ---------------------------------------------------------------------------

class GeometricMultiheadAttentionV2(nn.Module):
    """Multi-head self-attention with optional 10-D-conditioned geometry.

    Modes mirror v1:
        - "g1": geometry enters the score as additive per-head bias.
        - "g2": geometry enters the value as a per-pair per-head increment.
        - "g3": both.
        - "none": vanilla scaled dot-product (no geometry tables).

    The geometric MLPs accept a 10-D descriptor instead of v1's 4-D, but
    the rest of the math is unchanged.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        geometry_injection: GeomInjection = "g3",
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

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        if geometry_injection in ("g1", "g3"):
            self.geom_bias_mlp = nn.Sequential(
                nn.Linear(G_DIM_V2, geom_mlp_hidden),
                nn.GELU(),
                nn.Linear(geom_mlp_hidden, n_heads),
            )
        else:
            self.geom_bias_mlp = None

        if geometry_injection in ("g2", "g3"):
            self.geom_value_mlp = nn.Sequential(
                nn.Linear(G_DIM_V2, geom_mlp_hidden),
                nn.GELU(),
                nn.Linear(geom_mlp_hidden, d_model),
            )
        else:
            self.geom_value_mlp = None

        self.dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    # Geometry tables (caller precomputes once per montage)
    # ------------------------------------------------------------------

    def build_geom_bias_table(self, g: torch.Tensor) -> torch.Tensor:
        """(M, M, 10) -> (M, M, H) score-side per-head bias."""
        M = g.shape[0]
        if self.geom_bias_mlp is None:
            return g.new_zeros(M, M, self.n_heads)
        return self.geom_bias_mlp(g)

    def build_geom_value_table(self, g: torch.Tensor) -> torch.Tensor:
        """(M, M, 10) -> (M, M, H, d_head) value-side per-pair increment."""
        M = g.shape[0]
        if self.geom_value_mlp is None:
            return g.new_zeros(M, M, self.n_heads, self.d_head)
        v_inc = self.geom_value_mlp(g)
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
            x: (B, M, d_model). v2 has no CLS token; tokens are electrodes.
            geom_bias_table: (M, M, H) per-head additive bias, or None.
                Required for g1/g3; ignored otherwise.
            geom_value_table: (M, M, H, d_head) per-pair value increment.
                Required for g2/g3; ignored otherwise.
            key_padding_mask: (B, M) bool. True = mask this electrode out.

        Returns:
            y: (B, M, d_model)
        """
        B, M, _ = x.shape
        H, dh = self.n_heads, self.d_head

        q = self.q_proj(x).view(B, M, H, dh).transpose(1, 2)
        k = self.k_proj(x).view(B, M, H, dh).transpose(1, 2)
        v = self.v_proj(x).view(B, M, H, dh).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(dh)

        if self.geometry_injection in ("g1", "g3"):
            if geom_bias_table is None:
                raise ValueError(
                    f"geometry_injection={self.geometry_injection!r} requires "
                    "geom_bias_table"
                )
            if geom_bias_table.shape != (M, M, H):
                raise ValueError(
                    f"geom_bias_table shape {tuple(geom_bias_table.shape)} "
                    f"!= expected ({M}, {M}, {H})"
                )
            bias = geom_bias_table.permute(2, 0, 1).unsqueeze(0)
            scores = scores + bias

        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :]
            scores = scores.masked_fill(mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)

        if self.geometry_injection in ("g2", "g3"):
            if geom_value_table is None:
                raise ValueError(
                    f"geometry_injection={self.geometry_injection!r} requires "
                    "geom_value_table"
                )
            if geom_value_table.shape != (M, M, H, dh):
                raise ValueError(
                    f"geom_value_table shape {tuple(geom_value_table.shape)} "
                    f"!= expected ({M}, {M}, {H}, {dh})"
                )
            gv = geom_value_table.permute(2, 0, 1, 3)  # (H, M, M, dh)
            geom_msg = torch.einsum("bhik,hikd->bhid", attn, gv)
            out = out + geom_msg

        out = out.transpose(1, 2).contiguous().view(B, M, self.d_model)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Encoder block + stack
# ---------------------------------------------------------------------------

class GeometricEncoderBlockV2(nn.Module):
    """Pre-norm Transformer block with v2 geometric attention."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        geometry_injection: GeomInjection = "g3",
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        geom_mlp_hidden: int = 32,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = GeometricMultiheadAttentionV2(
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


class GeometricSpatialEncoderV2(nn.Module):
    """Per-electrode spatial encoder. Returns (B, M, d), no CLS.

    Per-layer geometric MLPs (one set per block); `precompute_geom_tables`
    returns lists of length `n_layers` to match.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        geometry_injection: GeomInjection = "g3",
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        geom_mlp_hidden: int = 32,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.geometry_injection = geometry_injection

        self.blocks = nn.ModuleList([
            GeometricEncoderBlockV2(
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

    def precompute_geom_tables(self, ch_pos: torch.Tensor):
        """Build per-layer (bias_table, value_table) lists from coords.

        Returns:
            (bias_tables, value_tables), each a list of length n_layers.
            Shapes: bias (M, M, H), value (M, M, H, d_head). No CLS padding.
        """
        if not isinstance(ch_pos, torch.Tensor):
            ch_pos = torch.as_tensor(ch_pos, dtype=torch.float32)
        # Get device/dtype from the first parameter so we don't depend on
        # whether the encoder has a CLS token.
        ref = next(self.parameters())
        ch_pos = ch_pos.to(ref.device, dtype=ref.dtype)
        g = build_geometric_descriptor_v2(ch_pos)  # (M, M, 10)

        bias_tables: List[torch.Tensor] = []
        value_tables: List[torch.Tensor] = []
        for blk in self.blocks:
            attn = blk.attn
            bias_tables.append(attn.build_geom_bias_table(g))
            value_tables.append(attn.build_geom_value_table(g))
        return bias_tables, value_tables

    def forward(
        self,
        x: torch.Tensor,
        bias_tables: Optional[list] = None,
        value_tables: Optional[list] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, M, d_model). Per-electrode patch features at one time step.
            bias_tables / value_tables: from `precompute_geom_tables`.
            key_padding_mask: (B, M) bool, True = masked electrode.

        Returns:
            y: (B, M, d_model). Per-electrode contextualized features.
        """
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
            x = blk(
                x,
                geom_bias_table=bt,
                geom_value_table=vt,
                key_padding_mask=key_padding_mask,
            )
        return self.norm(x)


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    B, M, d = 2, 8, 64
    ch_pos = torch.randn(M, 3)
    # Apply a v2-style fixed normalization (not the real REF_* constants,
    # but a consistent shared frame for the smoke).
    ch_pos = (ch_pos - ch_pos.mean(0, keepdim=True)) / ch_pos.norm(dim=-1).max()

    for mode in ("g1", "g2", "g3"):
        enc = GeometricSpatialEncoderV2(
            d_model=d, n_heads=4, n_layers=2, geometry_injection=mode
        )
        bts, vts = enc.precompute_geom_tables(ch_pos)
        assert len(bts) == 2 and bts[0].shape == (M, M, 4)
        assert vts[0].shape == (M, M, 4, d // 4)
        x = torch.randn(B, M, d, requires_grad=True)
        y = enc(x, bts, vts)
        assert y.shape == (B, M, d), f"mode={mode}: got {y.shape}"
        y.sum().backward()
        assert x.grad is not None
        print(f"  {mode}: output shape {tuple(y.shape)} OK")

    # 10-D descriptor sanity
    g = build_geometric_descriptor_v2(ch_pos)
    assert g.shape == (M, M, 10)
    # Self-pair should have zero displacement and zero distance, but the
    # p_i / p_j components are the electrode's own position twice.
    for i in range(M):
        assert torch.allclose(g[i, i, 6:9], torch.zeros(3), atol=1e-6)
        assert g[i, i, 9].item() < 1e-6
        assert torch.allclose(g[i, i, 0:3], ch_pos[i])
        assert torch.allclose(g[i, i, 3:6], ch_pos[i])
    print("geometric_attention_v2.py smoke OK")
