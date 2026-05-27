"""v2 pretraining model: per-channel alignment + per-channel reconstruction.

What v1 does (`src/model/momentum_encoder.py`):
    - Backbone returns (B, T_p, d). Electrodes are CLS-pooled inside the
      spatial encoder.
    - L_align = -2 * cos(pred, z_momentum) over (B, T_p, d). The target
      is spatially pooled, so there is no per-electrode signal to learn.
    - L_recon target is `patches.mean(dim=1)` -- also spatially pooled.

This is the precise mechanism that let v1's b collapse to a per-head
linear ramp in distance: no per-electrode gradient signal means no
incentive to route attention by direction. See `docs/v2/v2_scope.md`
v2.1.

v2.1:
    - Backbone returns (B, M, T_p, d) per-electrode tokens.
    - L_align = -2 * cos(pred, z_momentum) averaged over (B, M, T_p).
    - L_recon target = raw per-electrode patches (no spatial mean).
    - Recon head: Linear(d, patch_samples), applied per (B, M, T_p).
    - Spatial masking is dropped; temporal masking only.

EMA + cosine tau schedule are reused unchanged in spirit from v1; the
implementation lives here to keep v2 self-contained.
"""

from __future__ import annotations

import copy
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.v2.model.backbone_v2 import BackboneV2


# ---------------------------------------------------------------------------
# Prediction MLP (online side; never EMA-copied)
# ---------------------------------------------------------------------------

class PredictionMLPV2(nn.Module):
    """2-layer prediction head, applied per-token over (..., d_model)."""

    def __init__(self, d_model: int, hidden_ratio: float = 4.0):
        super().__init__()
        hidden = int(d_model * hidden_ratio)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# PretrainModelV2
# ---------------------------------------------------------------------------

class PretrainModelV2(nn.Module):
    """Online + EMA momentum backbone with per-channel alignment + recon.

    Usage::

        backbone = BackboneV2(spatial, arch)
        bias_tabs, val_tabs = backbone.precompute_geom_tables(ch_pos)
        model = PretrainModelV2(backbone, lambda_R=1.0)

        for x in loader:                               # (B, M, T)
            T_p = arch.n_patches(x.shape[-1])
            tmask = torch.rand(B, T_p) < 0.5
            L, L_A, L_R = model(x, tmask, bias_tabs, val_tabs)
            L.backward(); opt.step(); opt.zero_grad()
            tau = PretrainModelV2.cosine_tau(step, total_steps,
                                             cfg.train.tau_base,
                                             cfg.train.tau_final)
            model.update_ema(tau=tau)
    """

    def __init__(
        self,
        backbone: BackboneV2,
        lambda_R: float = 1.0,
        pred_hidden_ratio: float = 4.0,
    ):
        super().__init__()
        d = backbone.arch.d_model
        patch_samples = backbone.arch.patch_samples

        self.online_backbone = backbone
        self.momentum_backbone = copy.deepcopy(backbone)
        for p in self.momentum_backbone.parameters():
            p.requires_grad_(False)

        self.pred_mlp = PredictionMLPV2(d, hidden_ratio=pred_hidden_ratio)
        # Per-electrode reconstruction: predict the raw patch waveform
        # (patch_samples) at each (electrode, time-step). No spatial mean.
        self.recon_head = nn.Linear(d, patch_samples)

        self.lambda_R = lambda_R

    # ------------------------------------------------------------------
    # Parameter groups + EMA
    # ------------------------------------------------------------------

    def online_parameters(self):
        return (
            list(self.online_backbone.parameters())
            + list(self.pred_mlp.parameters())
            + list(self.recon_head.parameters())
        )

    @staticmethod
    def cosine_tau(
        step: int,
        total_steps: int,
        tau_base: float = 0.996,
        tau_final: float = 1.0,
    ) -> float:
        frac = step / max(total_steps, 1)
        return tau_final - (tau_final - tau_base) * (math.cos(math.pi * frac) + 1) / 2

    @torch.no_grad()
    def update_ema(self, tau: float = 0.996) -> None:
        for p_o, p_m in zip(
            self.online_backbone.parameters(),
            self.momentum_backbone.parameters(),
        ):
            p_m.data.mul_(tau).add_(p_o.data, alpha=1.0 - tau)

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mask_input(
        x: torch.Tensor, temporal_mask: torch.Tensor, patch_samples: int
    ) -> torch.Tensor:
        """Zero out raw EEG samples in temporally-masked patch windows.

        Same logic as v1: assumes non-overlapping patches.
        """
        B, T_p = temporal_mask.shape
        expected_T = T_p * patch_samples
        if x.shape[-1] != expected_T:
            raise ValueError(
                f"_mask_input expects T={expected_T} (T_p={T_p} * P={patch_samples}), "
                f"got T={x.shape[-1]}."
            )
        keep = (~temporal_mask).float()                              # (B, T_p)
        keep_time = (
            keep.unsqueeze(-1).expand(B, T_p, patch_samples).reshape(B, expected_T)
        )
        return x * keep_time.unsqueeze(1)                            # broadcast over M

    @staticmethod
    def _cosine_loss_per_channel(
        pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """v2.1 per-channel cosine loss.

        Args:
            pred:   (B, M, T_p, d_model)
            target: (B, M, T_p, d_model) stop-grad
        Returns:
            scalar loss in [-2, 0]; mean over (B, M, T_p).
        """
        pred_n = F.normalize(pred, dim=-1)
        tgt_n = F.normalize(target, dim=-1)
        return (-2.0 * (pred_n * tgt_n).sum(dim=-1)).mean()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        temporal_mask: Optional[torch.Tensor] = None,
        bias_tables: Optional[list] = None,
        value_tables: Optional[list] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute v2 pretraining losses for one batch.

        Args:
            x: (B, M, T). Clean EEG; temporal masking applied internally.
            temporal_mask: (B, T_p) bool, True = mask. If None, L_R = 0.
            bias_tables / value_tables: per-layer geometry tables.

        Returns:
            (L_total, L_align, L_recon).
        """
        if x.dim() != 3:
            raise ValueError(f"expected (B, M, T), got {tuple(x.shape)}")

        arch = self.online_backbone.arch

        # Online: masked input + temporal mask passed downstream.
        if temporal_mask is not None:
            x_online = self._mask_input(x, temporal_mask, arch.patch_samples)
        else:
            x_online = x

        z_online = self.online_backbone(
            x_online,
            bias_tables=bias_tables,
            value_tables=value_tables,
            spatial_mask=None,             # v2 drops spatial masking
            temporal_mask=temporal_mask,
        )                                  # (B, M, T_p, d)
        pred = self.pred_mlp(z_online)     # (B, M, T_p, d)

        with torch.no_grad():
            z_momentum = self.momentum_backbone(
                x,
                bias_tables=bias_tables,
                value_tables=value_tables,
                spatial_mask=None,
                temporal_mask=None,        # momentum sees full sequence
            )                              # (B, M, T_p, d)

        L_A = self._cosine_loss_per_channel(pred, z_momentum)

        if self.lambda_R > 0.0 and temporal_mask is not None and temporal_mask.any():
            # Per-electrode patches as the recon target -- NO channel mean.
            recon = self.recon_head(z_online)                # (B, M, T_p, P)
            with torch.no_grad():
                target = self.online_backbone.patcher(x)     # (B, M, T_p, P)
            # MSE only over masked time steps; averaged across (B, M) for
            # those time steps so a batch with more channels doesn't get
            # a bigger loss.
            mask_f = temporal_mask.float()                   # (B, T_p)
            loss_grid = F.mse_loss(recon, target, reduction="none")  # (B,M,T_p,P)
            # Mean over the patch_samples dim; then mean over electrodes;
            # then mask-weighted sum/normalize over time steps.
            per_pos = loss_grid.mean(dim=-1).mean(dim=1)     # (B, T_p)
            L_R = (per_pos * mask_f).sum() / mask_f.sum().clamp_min(1.0)
        else:
            L_R = x.new_zeros(()).squeeze()

        L_total = L_A + self.lambda_R * L_R
        return L_total, L_A, L_R


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from src.config import Config
    from src.v2.model.geometric_attention_v2 import GeometricSpatialEncoderV2

    torch.manual_seed(42)
    cfg = Config()
    arch = cfg.arch

    M, T, B = 6, arch.patch_samples * 8, 2
    ch_pos = torch.randn(M, 3)
    ch_pos = (ch_pos - ch_pos.mean(0)) / ch_pos.norm(dim=-1).max()
    x = torch.randn(B, M, T)
    T_p = arch.n_patches(T)
    tmask = torch.zeros(B, T_p, dtype=torch.bool)
    tmask[:, ::2] = True

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

    model = PretrainModelV2(backbone, lambda_R=1.0)
    L, L_A, L_R = model(x, tmask, bts, vts)
    L.backward()

    online_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.online_backbone.parameters()
    )
    momentum_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.momentum_backbone.parameters()
    )
    assert online_has_grad, "online backbone got no gradient"
    assert not momentum_has_grad, "momentum backbone leaked gradient"

    # EMA update doesn't crash and moves momentum toward online when they
    # differ. Momentum is deep-copied from online at init, so we perturb
    # online slightly first to make the EMA step observable.
    with torch.no_grad():
        next(model.online_backbone.parameters()).add_(1.0)
    before = next(model.momentum_backbone.parameters()).detach().clone()
    model.update_ema(tau=0.5)
    after = next(model.momentum_backbone.parameters()).detach()
    assert not torch.equal(before, after), "EMA update didn't change momentum params"

    print(
        f"v2/losses smoke OK; L={L.item():.4f} (L_A={L_A.item():.4f}, "
        f"L_R={L_R.item():.4f})"
    )
