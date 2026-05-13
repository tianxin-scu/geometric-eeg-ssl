"""Transductive (codex-based) baseline spatial encoder.

This is the head-to-head comparison target for the geometric encoder (see
`project_direction_summary.md` §4 O2). The architecture is identical
*except* that:

  (a) attention is vanilla scaled dot-product, with no geometric conditioning,
      and
  (b) a learned per-electrode embedding c_i (the "codex") is added to each
      electrode's patch projection before the encoder.

The codex makes the encoder "transductive" in the GraphSAGE sense: it
embeds each electrode by name. New electrodes get no embedding; the model
can't transfer to a different montage without retraining or substitution.

Parameter matching note. The headline comparison must be parameter-matched
to be honest. The geometric model spends extra parameters on per-layer
geometry MLPs (R^4 -> R^H for G1, R^4 -> R^d for G2). The codex model spends
extra parameters on a learned (M, d_model) table. For M = 64 and d_model = 256
this gives the codex roughly 16k parameters; the geometric model with G1 and
d_model=256, n_heads=8, geom_hidden=32, L_S=2 spends 2 * (4*32 + 32*8) +
biases ~= 800. To bring them into the same ballpark we either (a) accept
that the codex is slightly larger (which is the conservative direction for
us — we want to show geometric WINS at parameter parity or better) or (b)
shrink the codex by tying or low-ranking it. For now we expose `codex_rank`
to let the codex be a low-rank (M -> r -> d_model) factorization so the
student can tune for exact parameter match if desired. Default `codex_rank
= None` gives the full per-electrode embedding (the standard formulation).

Use `count_parameters_excluding_proj_and_norm` to compare the two
architectures consistently; it excludes shared modules so the comparison
isolates the differing parameters.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Codex
# ---------------------------------------------------------------------------

class ElectrodeCodex(nn.Module):
    """Per-electrode learned embedding c_i in R^{d_model}.

    Indexed by electrode position in the input tensor (0..M-1). At training
    time M is fixed by the dataset; if M changes between train and eval
    (cross-montage), the codex doesn't apply and the model must either be
    retrained or use a kNN substitution heuristic. The geometric encoder
    avoids this problem entirely.

    Optional low-rank factorization: c_i = U V[i], where U is (rank, d_model)
    and V is (M, rank). Useful if you want exact parameter parity with the
    geometric model.
    """

    def __init__(self, n_electrodes: int, d_model: int, rank: Optional[int] = None):
        super().__init__()
        self.n_electrodes = n_electrodes
        self.d_model = d_model
        self.rank = rank
        if rank is None:
            self.table = nn.Parameter(torch.zeros(n_electrodes, d_model))
            nn.init.trunc_normal_(self.table, std=0.02)
        else:
            self.V = nn.Parameter(torch.zeros(n_electrodes, rank))
            self.U = nn.Parameter(torch.zeros(rank, d_model))
            nn.init.trunc_normal_(self.V, std=0.02)
            nn.init.trunc_normal_(self.U, std=0.02)

    def forward(self) -> torch.Tensor:
        """Return the full (M, d_model) codex table."""
        if self.rank is None:
            return self.table
        return self.V @ self.U


# ---------------------------------------------------------------------------
# Plain dot-product multi-head attention (no geometry)
# ---------------------------------------------------------------------------

class PlainMultiheadAttention(nn.Module):
    """Vanilla scaled dot-product multi-head self-attention.

    Implemented as a sibling of GeometricMultiheadAttention so both share
    the exact same Q/K/V/out_proj structure (and therefore identical
    parameter count there). The only difference is the absence of the
    geometric MLPs.
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
            mask = key_padding_mask[:, None, None, :]
            scores = scores.masked_fill(mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, self.d_model)
        return self.out_proj(out)


class PlainEncoderBlock(nn.Module):
    """Pre-norm Transformer block, no geometry."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = PlainMultiheadAttention(d_model, n_heads, dropout=dropout)
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
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), key_padding_mask=key_padding_mask)
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Transductive spatial encoder = codex + vanilla Transformer
# ---------------------------------------------------------------------------

class TransductiveSpatialEncoder(nn.Module):
    """Sibling of GeometricSpatialEncoder; uses codex instead of geometry.

    Same defaults (L_S=2, H=8, d=256), same CLS-pooling, same block layout
    except for the absence of geometric MLPs and the presence of the codex.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        n_electrodes: int = 64,
        codex_rank: Optional[int] = None,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_electrodes = n_electrodes

        self.codex = ElectrodeCodex(
            n_electrodes=n_electrodes, d_model=d_model, rank=codex_rank
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.blocks = nn.ModuleList([
            PlainEncoderBlock(
                d_model=d_model,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
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
            x: (B, M, d_model). Patch features for M electrodes at one time
                step. M MUST equal n_electrodes (codex is transductive).
            key_padding_mask: (B, M) electrode-level mask.

        Returns:
            s: (B, d_model) CLS-pooled summary.
        """
        B, M, _ = x.shape
        if M != self.n_electrodes:
            raise ValueError(
                f"x has M={M} but codex was built for M={self.n_electrodes}. "
                "The transductive baseline does not support cross-montage "
                "transfer; this is the whole point of the comparison."
            )
        codex = self.codex()  # (M, d_model)
        x = x + codex.unsqueeze(0)  # broadcast over batch

        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, x], dim=1)
        if key_padding_mask is not None:
            cls_pad = key_padding_mask.new_zeros(B, 1)
            full_mask = torch.cat([cls_pad, key_padding_mask], dim=1)
        else:
            full_mask = None

        for blk in self.blocks:
            tokens = blk(tokens, key_padding_mask=full_mask)
        tokens = self.norm(tokens)
        return tokens[:, 0]


# ---------------------------------------------------------------------------
# Parameter-counting helper for the headline comparison
# ---------------------------------------------------------------------------

def count_distinguishing_parameters(model: nn.Module) -> dict:
    """Break down parameter counts by source so the geometric vs codex
    comparison can be reported honestly.

    Returns a dict with:
        total: total trainable parameters
        codex: parameters in the ElectrodeCodex (transductive only)
        geom_bias: parameters in geom_bias_mlp across layers (geometric only)
        geom_value: parameters in geom_value_mlp across layers (geometric only)
        attn_qkvo: parameters in q/k/v/out projections (SHARED between variants)
        mlp: parameters in FFN MLPs (SHARED)
        norm: parameters in LayerNorms (SHARED)
        cls: parameters in CLS token (SHARED)
        other: anything not classified above
    """
    out = {
        "total": 0,
        "codex": 0,
        "geom_bias": 0,
        "geom_value": 0,
        "attn_qkvo": 0,
        "mlp": 0,
        "norm": 0,
        "cls": 0,
        "other": 0,
    }
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n = p.numel()
        out["total"] += n
        if "codex" in name:
            out["codex"] += n
        elif "geom_bias_mlp" in name:
            out["geom_bias"] += n
        elif "geom_value_mlp" in name:
            out["geom_value"] += n
        elif any(s in name for s in ("q_proj", "k_proj", "v_proj", "out_proj")):
            out["attn_qkvo"] += n
        elif "mlp" in name:
            out["mlp"] += n
        elif "norm" in name:
            out["norm"] += n
        elif "cls_token" in name:
            out["cls"] += n
        else:
            out["other"] += n
    return out


# ---------------------------------------------------------------------------
# Smoke assertion block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    B, M, d = 2, 8, 64
    enc = TransductiveSpatialEncoder(
        d_model=d, n_heads=4, n_layers=2, n_electrodes=M
    )
    x = torch.randn(B, M, d, requires_grad=True)
    s = enc(x)
    assert s.shape == (B, d), f"got {s.shape}"
    s.sum().backward()
    assert x.grad is not None
    print("transductive_baseline.py smoke OK; output shape", tuple(s.shape))
    print("param breakdown:", count_distinguishing_parameters(enc))
