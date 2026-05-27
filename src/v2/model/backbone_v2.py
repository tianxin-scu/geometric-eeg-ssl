"""v2 backbone: raw EEG -> per-electrode contextualized features.

Key difference vs v1's `src/model/backbone.py`:

    v1: (B, M, T)
        -> patcher -> proj -> spatial(CLS-pool) -> (B, T_p, d)
        -> temporal -> (B, T_p, d)            # electrode dim is gone

    v2: (B, M, T)
        -> patcher -> proj -> spatial_v2      -> (B, M, T_p, d)
        -> temporal (per-electrode)           -> (B, M, T_p, d)

The temporal Transformer runs over each electrode's time series
independently by reshaping (B, M, T_p, d) -> (B*M, T_p, d), running the
existing TemporalTransformer (imported from v1 unchanged), then reshaping
back. This costs M times more temporal compute than v1, but it's the
only way to keep electrode identity through to the loss for the v2.1
per-channel target.

Patcher, linear projection, and TemporalTransformer are imported from v1.
They are not modified.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.model.backbone import Patcher, TemporalTransformer
from src.v2.model.geometric_attention_v2 import GeometricSpatialEncoderV2


class BackboneV2(nn.Module):
    """End-to-end v2 encoder: raw EEG -> per-electrode time-step features.

    Built with the v2 spatial encoder (`GeometricSpatialEncoderV2`).
    Per-electrode temporal context comes from running v1's
    `TemporalTransformer` over each electrode's patch sequence.

    Args:
        spatial: `GeometricSpatialEncoderV2` instance (no CLS pool).
        arch: ArchConfig from `src.config`.

    Forward returns (B, M, T_p, d_model).
    """

    def __init__(self, spatial: GeometricSpatialEncoderV2, arch):
        super().__init__()
        if not isinstance(spatial, GeometricSpatialEncoderV2):
            raise TypeError(
                "BackboneV2 expects a GeometricSpatialEncoderV2; got "
                f"{type(spatial).__name__}. Use v1's Backbone for v1 encoders."
            )
        if spatial.d_model != arch.d_model:
            raise ValueError(
                f"spatial.d_model={spatial.d_model} != arch.d_model={arch.d_model}"
            )

        self.spatial = spatial
        self.arch = arch
        self.patcher = Patcher(arch.patch_samples, arch.patch_stride)
        self.proj = nn.Linear(arch.patch_samples, arch.d_model, bias=True)
        self.temporal = TemporalTransformer(
            d_model=arch.d_model,
            n_heads=arch.n_heads_temporal,
            n_layers=arch.n_layers_temporal,
            mlp_ratio=arch.mlp_ratio,
            dropout=arch.dropout,
            posemb_mode=arch.temporal_posemb,
        )

    def precompute_geom_tables(self, ch_pos: torch.Tensor):
        return self.spatial.precompute_geom_tables(ch_pos)

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
            x: (B, M, T) raw EEG.
            bias_tables / value_tables: from `precompute_geom_tables`.
            spatial_mask: (B, M) bool, True = mask electrode.
            temporal_mask: (B, T_p) bool, True = mask time step.

        Returns:
            y: (B, M, T_p, d_model). Per-electrode per-time-step features.
        """
        if x.dim() != 3:
            raise ValueError(f"expected (B, M, T), got {tuple(x.shape)}")
        B, M, T = x.shape

        # 1) Patch + project per patch (electrode-agnostic).
        patches = self.patcher(x)                        # (B, M, T_p, P)
        T_p = patches.shape[2]
        feats = self.proj(patches)                       # (B, M, T_p, d)
        d = feats.shape[-1]

        # 2) Spatial encoder per time step (geometry-conditioned).
        # (B, M, T_p, d) -> (B, T_p, M, d) -> (B*T_p, M, d)
        feats_bt = feats.permute(0, 2, 1, 3).reshape(B * T_p, M, d)
        if spatial_mask is not None:
            if spatial_mask.shape != (B, M):
                raise ValueError(
                    f"spatial_mask shape {tuple(spatial_mask.shape)} != ({B}, {M})"
                )
            sp_mask_bt = (
                spatial_mask.unsqueeze(1).expand(B, T_p, M).reshape(B * T_p, M)
            )
        else:
            sp_mask_bt = None

        s_bt = self.spatial(
            feats_bt,
            bias_tables=bias_tables,
            value_tables=value_tables,
            key_padding_mask=sp_mask_bt,
        )                                                # (B*T_p, M, d)

        # 3) Temporal encoder per electrode.
        # (B*T_p, M, d) -> (B, T_p, M, d) -> (B, M, T_p, d) -> (B*M, T_p, d)
        s = s_bt.view(B, T_p, M, d).permute(0, 2, 1, 3).contiguous()  # (B, M, T_p, d)
        s_bm = s.reshape(B * M, T_p, d)

        if temporal_mask is not None:
            # Same temporal mask applies to every electrode of the same batch
            # item, so broadcast (B, T_p) -> (B, M, T_p) -> (B*M, T_p).
            tmask_bm = (
                temporal_mask.unsqueeze(1).expand(B, M, T_p).reshape(B * M, T_p)
            )
        else:
            tmask_bm = None

        y_bm = self.temporal(s_bm, key_padding_mask=tmask_bm)  # (B*M, T_p, d)
        return y_bm.view(B, M, T_p, d)


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

    from src.config import Config

    torch.manual_seed(0)
    cfg = Config()
    arch = cfg.arch
    M, T, B = 6, arch.patch_samples * 4, 2
    ch_pos = torch.randn(M, 3)
    ch_pos = (ch_pos - ch_pos.mean(0)) / ch_pos.norm(dim=-1).max()

    spatial = GeometricSpatialEncoderV2(
        d_model=arch.d_model,
        n_heads=arch.n_heads_spatial,
        n_layers=arch.n_layers_spatial,
        geometry_injection="g3",
        geom_mlp_hidden=arch.geom_mlp_hidden,
        dropout=arch.dropout,
    )
    backbone = BackboneV2(spatial, arch=arch)
    bts, vts = backbone.precompute_geom_tables(ch_pos)

    x = torch.randn(B, M, T, requires_grad=True)
    y = backbone(x, bts, vts)
    T_p = arch.n_patches(T)
    assert y.shape == (B, M, T_p, arch.d_model), f"got {y.shape}"
    y.sum().backward()
    assert x.grad is not None

    # With temporal mask
    tmask = torch.zeros(B, T_p, dtype=torch.bool)
    tmask[:, ::2] = True
    y2 = backbone(x.detach(), bts, vts, temporal_mask=tmask)
    assert y2.shape == y.shape

    # With spatial mask
    smask = torch.zeros(B, M, dtype=torch.bool)
    smask[:, 0] = True
    y3 = backbone(x.detach(), bts, vts, spatial_mask=smask)
    assert y3.shape == y.shape

    n_params = sum(p.numel() for p in backbone.parameters())
    print(f"backbone_v2 smoke OK; output {tuple(y.shape)}, params={n_params:,}")
