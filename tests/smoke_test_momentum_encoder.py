"""Smoke tests for PretrainModel (momentum encoder + losses).

Tests:
1.  Momentum encoder params have requires_grad=False.
2.  Gradients flow through the online backbone, not the momentum backbone.
3.  update_ema() moves momentum params toward online params.
4.  Reconstruction loss is nonzero when there are masked positions.
5.  Reconstruction loss is zero when temporal_mask is all-False.
6.  L_total == L_A + lambda_R * L_R (loss decomposition).
7.  lambda_R=0 disables the reconstruction head (L_R == 0).
8.  Works for transductive variant (no geometry tables).
9.  Alignment loss is in [−2, 0] (cosine similarity range property).
10. update_ema with tau=1 leaves momentum params unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.config import Config
from src.model.backbone import Backbone
from src.model.geometric_attention import GeometricSpatialEncoder
from src.model.momentum_encoder import PretrainModel
from src.model.transductive_baseline import TransductiveSpatialEncoder


def _make_geometric_model(arch, M: int):
    spatial = GeometricSpatialEncoder(
        d_model=arch.d_model,
        n_heads=arch.n_heads_spatial,
        n_layers=arch.n_layers_spatial,
        geometry_injection="g1",
        geom_mlp_hidden=arch.geom_mlp_hidden,
        dropout=arch.dropout,
    )
    return Backbone(spatial, arch=arch)


def _make_transductive_model(arch, M: int):
    spatial = TransductiveSpatialEncoder(
        d_model=arch.d_model,
        n_heads=arch.n_heads_spatial,
        n_layers=arch.n_layers_spatial,
        n_electrodes=M,
        dropout=arch.dropout,
    )
    return Backbone(spatial, arch=arch)


def _random_ch_pos(M: int):
    ch_pos = torch.randn(M, 3)
    ch_pos = ch_pos - ch_pos.mean(0, keepdim=True)
    ch_pos = ch_pos / ch_pos.norm(dim=-1).max().clamp_min(1e-8)
    return ch_pos


torch.manual_seed(0)
cfg = Config()
arch = cfg.arch
M, T, B = 8, 800, 4
T_p = arch.n_patches(T)  # 16 patches for default config
ch_pos = _random_ch_pos(M)

# Geometric variant setup
backbone_g = _make_geometric_model(arch, M)
bias_tabs, val_tabs = backbone_g.precompute_geom_tables(ch_pos)
model_g = PretrainModel(backbone_g, lambda_R=1.0)

# A standard 50 % temporal mask
mask_50 = torch.zeros(B, T_p, dtype=torch.bool)
mask_50[:, ::2] = True   # 8 of 16 time steps masked

x = torch.randn(B, M, T)


# -----------------------------------------------------------------
# 1. Momentum params have requires_grad=False
# -----------------------------------------------------------------
for p in model_g.momentum_backbone.parameters():
    assert not p.requires_grad, "momentum parameter should not require grad"
print("PASS  1: momentum params have requires_grad=False")


# -----------------------------------------------------------------
# 2. Gradients flow through online backbone, not momentum backbone
# -----------------------------------------------------------------
# Fresh identical model to avoid grad accumulation from later tests.
bb_fresh = _make_geometric_model(arch, M)
bt2, vt2 = bb_fresh.precompute_geom_tables(ch_pos)
m2 = PretrainModel(bb_fresh, lambda_R=1.0)

L, L_A, L_R = m2(x, mask_50, bt2, vt2)
L.backward()

online_grad_total = sum(
    p.grad.abs().sum().item()
    for p in m2.online_backbone.parameters()
    if p.grad is not None
)
mom_any_grad = any(
    p.grad is not None for p in m2.momentum_backbone.parameters()
)
assert online_grad_total > 0.0, "no gradient in online backbone"
assert not mom_any_grad, "momentum backbone should not accumulate gradients"
print("PASS  2: gradients flow only through online backbone")


# -----------------------------------------------------------------
# 3. update_ema moves momentum params toward online params
# -----------------------------------------------------------------
# Snapshot momentum params before update.
mom_snap = {
    n: p.data.clone()
    for n, p in m2.momentum_backbone.named_parameters()
}
m2.update_ema(tau=0.99)
changed = 0
for n, p in m2.momentum_backbone.named_parameters():
    if not torch.equal(p.data, mom_snap[n]):
        changed += 1
assert changed > 0, "update_ema did not change any momentum parameter"
print(f"PASS  3: update_ema changed {changed} momentum parameter tensors")


# -----------------------------------------------------------------
# 4. Reconstruction loss is nonzero when there are masked positions
# -----------------------------------------------------------------
m4 = PretrainModel(_make_geometric_model(arch, M), lambda_R=1.0)
bt4, vt4 = m4.online_backbone.precompute_geom_tables(ch_pos)
_, _, L_R4 = m4(x, mask_50, bt4, vt4)
assert L_R4.item() > 0.0, f"expected L_R > 0 with masking, got {L_R4.item()}"
print(f"PASS  4: L_R={L_R4.item():.4f} > 0 with masked positions")


# -----------------------------------------------------------------
# 5. Reconstruction loss is zero when no positions are masked
# -----------------------------------------------------------------
mask_none = torch.zeros(B, T_p, dtype=torch.bool)
_, _, L_R5 = m4(x, mask_none, bt4, vt4)
assert L_R5.item() == 0.0, f"expected L_R == 0 with empty mask, got {L_R5.item()}"
print(f"PASS  5: L_R={L_R5.item():.4f} == 0 with no masked positions")


# -----------------------------------------------------------------
# 6. L_total == L_A + lambda_R * L_R
# -----------------------------------------------------------------
L6, L_A6, L_R6 = m4(x, mask_50, bt4, vt4)
expected = L_A6 + m4.lambda_R * L_R6
assert torch.isclose(L6, expected, atol=1e-5), (
    f"loss decomposition mismatch: {L6.item()} != {expected.item()}"
)
print(f"PASS  6: L_total={L6.item():.4f} == L_A + λ_R·L_R")


# -----------------------------------------------------------------
# 7. lambda_R=0 disables reconstruction (L_R == 0)
# -----------------------------------------------------------------
m7 = PretrainModel(_make_geometric_model(arch, M), lambda_R=0.0)
bt7, vt7 = m7.online_backbone.precompute_geom_tables(ch_pos)
_, _, L_R7 = m7(x, mask_50, bt7, vt7)
assert L_R7.item() == 0.0, f"expected L_R == 0 when lambda_R=0, got {L_R7.item()}"
print(f"PASS  7: L_R=0 when lambda_R=0")


# -----------------------------------------------------------------
# 8. Transductive variant works (no geometry tables)
# -----------------------------------------------------------------
bb_t = _make_transductive_model(arch, M)
m8 = PretrainModel(bb_t, lambda_R=1.0)
L8, L_A8, L_R8 = m8(x, mask_50)
assert L8.shape == torch.Size([]), "transductive loss should be scalar"
assert L_R8.item() > 0.0, "transductive reconstruction loss should be nonzero"
print(f"PASS  8: transductive PretrainModel OK | L={L8.item():.4f}")


# -----------------------------------------------------------------
# 9. Alignment loss is in [−2, 2]  (−2·cosine_sim, cosine ∈ [−1, 1])
# -----------------------------------------------------------------
for seed in range(5):
    torch.manual_seed(seed)
    xi = torch.randn(B, M, T)
    m9 = PretrainModel(_make_geometric_model(arch, M), lambda_R=0.0)
    bt9, vt9 = m9.online_backbone.precompute_geom_tables(_random_ch_pos(M))
    _, L_A9, _ = m9(xi, mask_50, bt9, vt9)
    assert -2.0 - 1e-4 <= L_A9.item() <= 2.0 + 1e-4, (
        f"L_A={L_A9.item():.4f} outside [−2, 2]"
    )
print(f"PASS  9: alignment loss in [−2, 2] across 5 seeds")


# -----------------------------------------------------------------
# 10. update_ema with tau=1 leaves momentum params unchanged
# -----------------------------------------------------------------
m10 = PretrainModel(_make_geometric_model(arch, M), lambda_R=1.0)
bt10, vt10 = m10.online_backbone.precompute_geom_tables(ch_pos)
# Run a forward to have valid state.
m10(x, mask_50, bt10, vt10)

snap = {n: p.data.clone() for n, p in m10.momentum_backbone.named_parameters()}
m10.update_ema(tau=1.0)  # full momentum — should not change anything
for n, p in m10.momentum_backbone.named_parameters():
    assert torch.equal(p.data, snap[n]), f"tau=1 changed momentum param {n}"
print("PASS 10: update_ema(tau=1.0) leaves momentum params unchanged")


# ---------------------------------------------------------------------------
# 11. cosine_tau schedule properties
# ---------------------------------------------------------------------------
import math

tau_base, tau_final = 0.996, 1.0
K = 1000

# Boundary values
assert abs(PretrainModel.cosine_tau(0, K, tau_base, tau_final) - tau_base) < 1e-9, \
    "cosine_tau at step 0 should equal tau_base"
assert abs(PretrainModel.cosine_tau(K, K, tau_base, tau_final) - tau_final) < 1e-9, \
    "cosine_tau at step K should equal tau_final"

# Monotone increasing
taus = [PretrainModel.cosine_tau(k, K, tau_base, tau_final) for k in range(0, K + 1, 100)]
assert all(taus[i] <= taus[i + 1] for i in range(len(taus) - 1)), \
    "cosine_tau must be non-decreasing"

# All values in [tau_base, tau_final]
assert all(tau_base <= t <= tau_final for t in taus), \
    "cosine_tau must stay within [tau_base, tau_final]"

# total_steps=0 edge case must not raise (guard against div-by-zero)
v = PretrainModel.cosine_tau(0, 0, tau_base, tau_final)
assert isinstance(v, float), "cosine_tau with total_steps=0 should return a float"

print("PASS 11: cosine_tau schedule — boundaries, monotonicity, range, edge case")


print("\nAll smoke tests passed.")
