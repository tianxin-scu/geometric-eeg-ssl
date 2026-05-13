"""Momentum encoder, prediction MLP, and pretraining losses.

Implements the BYOL-style momentum encoder alignment and optional masked-patch
reconstruction described in proposal §3.

Architecture
------------
Online path (masked input):
    x_masked → Backbone → (B, T_p, d) =: z_online
    pred_mlp(z_online) → pred  (for alignment loss)
    recon_head(z_online[masked]) → patch recon  (for reconstruction loss)

Momentum path (clean input, no gradient):
    x → Backbone_EMA → (B, T_p, d) =: z_momentum  [stop-grad target]

Losses:
    L_A = -2 · mean cosine_similarity(pred, z_momentum)   (BYOL §3.2)
    L_R = MSE(recon_head(z_online), channel_mean_patches)  at masked positions
    L   = L_A + λ_R · L_R

Masking
-------
Masking is applied at the raw-signal level before the backbone: the time-domain
samples of each masked patch window are zeroed (assumes non-overlapping patches,
patch_stride == patch_samples, enforced in Config.__post_init__).  The temporal
Transformer additionally receives temporal_mask as a key_padding_mask so masked
time steps cannot contribute as keys/values to unmasked positions, but can still
gather context from unmasked positions (standard masked-prediction behaviour).

Reconstruction target
---------------------
The target for the reconstruction head is the channel-mean of the original
(unmasked) patches: patches.mean(dim=1) → (B, T_p, patch_samples).  This
collapses the electrode dimension to match the single linear head
d_model → patch_samples.  Per-electrode reconstruction would require the
backbone to expose pre-pooling features; that is out of scope.
"""

from __future__ import annotations

import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.backbone import Backbone


# ---------------------------------------------------------------------------
# Prediction MLP (online side only — never copied to momentum encoder)
# ---------------------------------------------------------------------------

class PredictionMLP(nn.Module):
    """2-layer prediction head: Linear → LayerNorm → GELU → Linear.

    Using LayerNorm rather than BatchNorm for consistency with the rest of
    the architecture (all transformers here use LayerNorm).

    Operates on any shape (..., d_model); the linear layers broadcast
    over leading dims so no reshape is needed.
    """

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
# PretrainModel — online + momentum backbone with losses
# ---------------------------------------------------------------------------

class PretrainModel(nn.Module):
    """Online encoder + EMA momentum encoder with BYOL alignment and
    optional masked-patch reconstruction.

    Usage (geometric variant)::

        spatial = GeometricSpatialEncoder(...)
        backbone = Backbone(spatial, arch=cfg.arch)
        bias_tabs, val_tabs = backbone.precompute_geom_tables(ch_pos)

        model = PretrainModel(backbone, lambda_R=1.0)
        optimizer = torch.optim.AdamW(model.online_parameters(), lr=1e-4)

        for x_batch in loader:                               # (B, M, T)
            T_p = cfg.arch.n_patches(x_batch.shape[-1])
            # 50 % random temporal masking
            mask = torch.rand(B, T_p) < 0.5
            L, L_A, L_R = model(x_batch, mask, bias_tabs, val_tabs)
            L.backward()
            optimizer.step(); optimizer.zero_grad()
            model.update_ema(tau=0.99)

    For the transductive variant, omit bias_tabs and val_tabs.

    Args:
        backbone: fully-constructed Backbone (geometric or transductive).
            The momentum backbone is a deepcopy created at init time.
        lambda_R: reconstruction loss weight. 0 → alignment only.
        pred_hidden_ratio: hidden-to-d_model ratio for the prediction MLP.
    """

    def __init__(
        self,
        backbone: Backbone,
        lambda_R: float = 1.0,
        pred_hidden_ratio: float = 4.0,
    ):
        super().__init__()
        d = backbone.arch.d_model
        patch_samples = backbone.arch.patch_samples

        self.online_backbone = backbone

        # Momentum backbone: deepcopy with gradients disabled.
        self.momentum_backbone = copy.deepcopy(backbone)
        for p in self.momentum_backbone.parameters():
            p.requires_grad_(False)

        # Online-only modules — never EMA-copied.
        self.pred_mlp = PredictionMLP(d, hidden_ratio=pred_hidden_ratio)
        self.recon_head = nn.Linear(d, patch_samples)

        self.lambda_R = lambda_R

    # ------------------------------------------------------------------
    # Parameter groups
    # ------------------------------------------------------------------

    def online_parameters(self):
        """Parameters that receive gradients (online backbone + pred + recon)."""
        return (
            list(self.online_backbone.parameters())
            + list(self.pred_mlp.parameters())
            + list(self.recon_head.parameters())
        )

    # ------------------------------------------------------------------
    # EMA update
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_ema(self, tau: float = 0.99) -> None:
        """EMA update of momentum backbone parameters.

        θ_m ← τ · θ_m + (1 − τ) · θ_o

        Call once per optimiser step, AFTER the backward pass and optimiser
        update.
        """
        for p_online, p_mom in zip(
            self.online_backbone.parameters(),
            self.momentum_backbone.parameters(),
        ):
            p_mom.data.mul_(tau).add_(p_online.data, alpha=1.0 - tau)

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mask_input(
        x: torch.Tensor,
        temporal_mask: torch.Tensor,
        patch_samples: int,
    ) -> torch.Tensor:
        """Zero out raw EEG samples at temporally-masked patch windows.

        Assumes non-overlapping patches (patch_stride == patch_samples),
        which is enforced by Config.__post_init__.  T must equal T_p *
        patch_samples.

        Args:
            x:             (B, M, T)
            temporal_mask: (B, T_p) bool.  True = zero this time step.
            patch_samples: samples per patch P.
        Returns:
            x_masked: (B, M, T) — same as x but zeroed at masked windows.
        """
        B, T_p = temporal_mask.shape
        expected_T = T_p * patch_samples
        if x.shape[-1] != expected_T:
            raise ValueError(
                f"_mask_input expects T={expected_T} (T_p={T_p} × P={patch_samples}), "
                f"got T={x.shape[-1]}.  Non-overlapping patching required."
            )
        # (B, T_p) → (B, T_p, P) → (B, T) — 1 where we keep, 0 where we zero.
        keep = (~temporal_mask).float()  # (B, T_p)
        keep_time = keep.unsqueeze(-1).expand(B, T_p, patch_samples).reshape(B, expected_T)
        return x * keep_time.unsqueeze(1)  # broadcast over M

    @staticmethod
    def _cosine_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """BYOL alignment loss: mean of −2 · cosine_similarity over (B, T_p).

        Args:
            pred:   (B, T_p, d_model) — output of pred_mlp (online).
            target: (B, T_p, d_model) — output of momentum encoder (stop-grad).
        Returns:
            scalar loss ∈ [−2, 0].
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
        spatial_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute pretraining losses for one batch.

        Args:
            x:             (B, M, T).  Clean (unmasked) EEG; masking is
                           applied internally so both paths share the same
                           input tensor.
            temporal_mask: (B, T_p) bool.  True = mask this time step.
                           If None, no masking; L_R is 0.
            bias_tables, value_tables: geometry tables from
                           ``backbone.precompute_geom_tables(ch_pos)``.
                           Required for the geometric variant, None for
                           transductive.
            spatial_mask:  (B, M) bool electrode mask passed to both
                           backbones.

        Returns:
            (L_total, L_alignment, L_reconstruction) — all scalar tensors.
            L_total = L_alignment + lambda_R * L_reconstruction.
        """
        if x.dim() != 3:
            raise ValueError(f"expected (B, M, T), got shape {tuple(x.shape)}")

        arch = self.online_backbone.arch

        # ---- Online path (masked input) --------------------------------
        if temporal_mask is not None:
            x_online = self._mask_input(x, temporal_mask, arch.patch_samples)
        else:
            x_online = x

        z_online = self.online_backbone(
            x_online,
            bias_tables=bias_tables,
            value_tables=value_tables,
            spatial_mask=spatial_mask,
            # Pass temporal_mask to the temporal Transformer so masked time
            # steps are excluded as keys/values — they can still attend to
            # unmasked context (standard masked-prediction semantics).
            temporal_mask=temporal_mask,
        )  # (B, T_p, d_model)

        pred = self.pred_mlp(z_online)  # (B, T_p, d_model)

        # ---- Momentum path (clean input, no gradient) ------------------
        with torch.no_grad():
            z_momentum = self.momentum_backbone(
                x,
                bias_tables=bias_tables,
                value_tables=value_tables,
                spatial_mask=spatial_mask,
                temporal_mask=None,  # momentum encoder sees full sequence
            )  # (B, T_p, d_model)

        # ---- Alignment loss (over all time steps) ----------------------
        L_A = self._cosine_loss(pred, z_momentum)

        # ---- Reconstruction loss (masked time steps only) --------------
        if self.lambda_R > 0.0 and temporal_mask is not None and temporal_mask.any():
            recon = self.recon_head(z_online)  # (B, T_p, patch_samples)
            # Target: channel-mean of the original (unmasked) patches.
            with torch.no_grad():
                patches = self.online_backbone.patcher(x)  # (B, M, T_p, P)
                target = patches.mean(dim=1)               # (B, T_p, P)
            # MSE weighted by the mask; only masked positions count.
            mask_f = temporal_mask.float()                  # (B, T_p)
            loss_grid = F.mse_loss(recon, target, reduction="none")  # (B, T_p, P)
            L_R = (loss_grid.mean(dim=-1) * mask_f).sum() / mask_f.sum().clamp_min(1.0)
        else:
            L_R = x.new_zeros(()).squeeze()

        L_total = L_A + self.lambda_R * L_R
        return L_total, L_A, L_R


# ---------------------------------------------------------------------------
# Smoke assertion block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from src.config import Config
    from src.model.geometric_attention import GeometricSpatialEncoder
    from src.model.transductive_baseline import TransductiveSpatialEncoder

    torch.manual_seed(42)
    cfg = Config()
    arch = cfg.arch

    M, T, B = 8, 800, 3
    ch_pos = torch.randn(M, 3)
    ch_pos = ch_pos - ch_pos.mean(0, keepdim=True)
    ch_pos = ch_pos / ch_pos.norm(dim=-1).max().clamp_min(1e-8)
    x = torch.randn(B, M, T)

    T_p = arch.n_patches(T)
    temporal_mask = torch.zeros(B, T_p, dtype=torch.bool)
    temporal_mask[:, ::2] = True  # mask every other time step

    # --- Geometric PretrainModel ---
    spatial_g = GeometricSpatialEncoder(
        d_model=arch.d_model, n_heads=arch.n_heads_spatial,
        n_layers=arch.n_layers_spatial, geometry_injection="g1",
        geom_mlp_hidden=arch.geom_mlp_hidden, dropout=arch.dropout,
    )
    backbone_g = Backbone(spatial_g, arch=arch)
    bias_tabs, val_tabs = backbone_g.precompute_geom_tables(ch_pos)

    model_g = PretrainModel(backbone_g, lambda_R=1.0)

    x_in = x.clone().requires_grad_(False)
    L, L_A, L_R = model_g(x_in, temporal_mask, bias_tabs, val_tabs)
    L.backward()

    # Online params should have gradients; momentum should not.
    online_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model_g.online_backbone.parameters()
    )
    mom_has_grad = any(
        p.grad is not None for p in model_g.momentum_backbone.parameters()
    )
    assert online_has_grad, "online backbone has no gradients"
    assert not mom_has_grad, "momentum backbone should not have gradients"
    assert L_R.item() > 0.0, "reconstruction loss should be nonzero with masking"
    print(
        f"geometric PretrainModel smoke OK | "
        f"L={L.item():.4f}  L_A={L_A.item():.4f}  L_R={L_R.item():.4f}"
    )

    # EMA update should move momentum params.
    p_before = next(model_g.momentum_backbone.parameters()).data.clone()
    model_g.update_ema(tau=0.99)
    p_after = next(model_g.momentum_backbone.parameters()).data.clone()
    assert not torch.equal(p_before, p_after), "EMA update did not change momentum params"
    print("EMA update smoke OK")
