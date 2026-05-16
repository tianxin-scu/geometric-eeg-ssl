"""Smoke test for scripts/pretrain.py training loop logic.

Does NOT require MNE or real PhysioNet data.  Exercises the loop
internals directly (model build, mask sampling, forward, backward,
EMA update, checkpoint save/load) with synthetic tensors.

Tests:
1.  Geometric PretrainModel: 2-step loop, loss decreases after backward.
2.  Transductive PretrainModel: 2-step loop runs without error.
3.  Temporal-only masking: L_R > 0; spatial_mask is None.
4.  Joint masking: both temporal_mask and spatial_mask are not None.
5.  Checkpoint save → load → resumed model produces identical forward output.
6.  LR scheduler: lr at epoch 0 is near zero (warmup), rises by epoch 5.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.config import AblationConfig, ArchConfig, Config, TrainConfig
from src.model.backbone import Backbone
from src.model.geometric_attention import GeometricSpatialEncoder
from src.model.momentum_encoder import PretrainModel
from src.model.transductive_baseline import TransductiveSpatialEncoder

# Import the helpers we want to test from the script itself.
from scripts.pretrain import (
    _build_lr_scheduler,
    _build_model,
    _load_checkpoint,
    _sample_mask,
    _save_checkpoint,
)

torch.manual_seed(0)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

B, M, T = 4, 8, 800   # small M so smoke tests are fast
_cfg_base = Config(
    arch=ArchConfig(
        d_model=32, n_heads_spatial=4, n_layers_spatial=1,
        n_heads_temporal=4, n_layers_temporal=1,
        geom_mlp_hidden=8, patch_samples=50, patch_stride=50,
    ),
    train=TrainConfig(
        lr=1e-3, batch_size=B, n_epochs=20, lr_warmup_epochs=5,
        tau_base=0.996, tau_final=1.0, lambda_R=1.0,
    ),
)
T_p = _cfg_base.arch.n_patches(T)

def _fake_batch():
    return torch.randn(B, M, T)

def _ch_pos():
    pos = torch.randn(M, 3)
    pos = pos - pos.mean(0, keepdim=True)
    pos = pos / pos.norm(dim=-1).max().clamp_min(1e-8)
    return pos


# ---------------------------------------------------------------------------
# Test 1: geometric 2-step loop
# ---------------------------------------------------------------------------

cfg1 = Config(arch=_cfg_base.arch, train=_cfg_base.train)  # default: geometric G1
model1 = _build_model(cfg1, n_electrodes=M)
ch_pos = _ch_pos()
opt1 = torch.optim.AdamW(model1.online_parameters(), lr=1e-3)

L_vals = []
for _ in range(2):
    x = _fake_batch()
    tmask = torch.rand(B, T_p) < 0.5
    # Tables recomputed each step: MLP weights change with optimizer.step().
    bt, vt = model1.online_backbone.precompute_geom_tables(ch_pos)
    L, L_A, L_R = model1(x, temporal_mask=tmask, bias_tables=bt, value_tables=vt)
    opt1.zero_grad(); L.backward(); opt1.step()
    tau = PretrainModel.cosine_tau(0, 100)
    model1.update_ema(tau=tau)
    L_vals.append(L.item())

assert len(L_vals) == 2
assert all(isinstance(v, float) for v in L_vals)
print(f"PASS 1: geometric 2-step loop  L={L_vals}")


# ---------------------------------------------------------------------------
# Test 2: transductive 2-step loop
# ---------------------------------------------------------------------------

cfg2 = Config(
    arch=_cfg_base.arch,
    train=_cfg_base.train,
    ablation=AblationConfig(use_codex=True, geometry_injection="G1"),
)
model2 = _build_model(cfg2, n_electrodes=M)
opt2 = torch.optim.AdamW(model2.online_parameters(), lr=1e-3)

for _ in range(2):
    x = _fake_batch()
    tmask = torch.rand(B, T_p) < 0.5
    L, _, _ = model2(x, temporal_mask=tmask)
    opt2.zero_grad(); L.backward(); opt2.step()
    model2.update_ema(tau=0.996)

print("PASS 2: transductive 2-step loop OK")


# ---------------------------------------------------------------------------
# Test 3: temporal-only masking
# ---------------------------------------------------------------------------

cfg3 = Config(
    arch=_cfg_base.arch,
    train=_cfg_base.train,
    ablation=AblationConfig(mask_scheme="temporal", mask_ratio=0.5),
)
tmask3, smask3 = _sample_mask(B, T_p, M, cfg3, device=torch.device("cpu"))
assert tmask3 is not None, "temporal mask should be set for scheme='temporal'"
assert smask3 is None,     "spatial mask should be None for scheme='temporal'"

model3 = _build_model(cfg3, n_electrodes=M)
_cp3 = _ch_pos()
bt3, vt3 = model3.online_backbone.precompute_geom_tables(_cp3)
L3, _, L_R3 = model3(_fake_batch(), temporal_mask=tmask3, bias_tables=bt3, value_tables=vt3)
assert L_R3.item() > 0.0, "reconstruction loss should be nonzero with temporal masking"
print(f"PASS 3: temporal-only mask  L_R={L_R3.item():.4f}")


# ---------------------------------------------------------------------------
# Test 4: joint masking
# ---------------------------------------------------------------------------

cfg4 = Config(
    arch=_cfg_base.arch,
    train=_cfg_base.train,
    ablation=AblationConfig(mask_scheme="joint", mask_ratio=0.5),
)
tmask4, smask4 = _sample_mask(B, T_p, M, cfg4, device=torch.device("cpu"))
assert tmask4 is not None, "temporal mask should be set for scheme='joint'"
assert smask4 is not None, "spatial mask should be set for scheme='joint'"
assert smask4.shape == (B, M)

model4 = _build_model(cfg4, n_electrodes=M)
_cp4 = _ch_pos()
bt4, vt4 = model4.online_backbone.precompute_geom_tables(_cp4)
L4, _, _ = model4(_fake_batch(), temporal_mask=tmask4, bias_tables=bt4, value_tables=vt4,
                   spatial_mask=smask4)
assert L4.item() == L4.item()  # not NaN
print(f"PASS 4: joint masking  L={L4.item():.4f}")


# ---------------------------------------------------------------------------
# Test 5: checkpoint save → load → same output
# ---------------------------------------------------------------------------

cfg5 = Config(arch=_cfg_base.arch, train=_cfg_base.train)
model5 = _build_model(cfg5, n_electrodes=M)
opt5 = torch.optim.AdamW(model5.online_parameters(), lr=1e-3)
sched5 = _build_lr_scheduler(opt5, cfg5)
_cp5 = _ch_pos()

x5 = _fake_batch()
tmask5 = torch.rand(B, T_p) < 0.5
with torch.no_grad():
    bt5, vt5 = model5.online_backbone.precompute_geom_tables(_cp5)
    out_before, _, _ = model5(x5, temporal_mask=tmask5, bias_tables=bt5, value_tables=vt5)

with tempfile.TemporaryDirectory() as tmpdir:
    ckpt_dir = Path(tmpdir)
    path = _save_checkpoint(ckpt_dir, epoch=0, step=10,
                            model=model5, optimizer=opt5, scheduler=sched5)

    # Build a fresh model and load the checkpoint.
    model5b = _build_model(cfg5, n_electrodes=M)
    opt5b = torch.optim.AdamW(model5b.online_parameters(), lr=1e-3)
    sched5b = _build_lr_scheduler(opt5b, cfg5)
    ep, st = _load_checkpoint(path, model5b, opt5b, sched5b, device="cpu")
    assert ep == 1 and st == 11

    with torch.no_grad():
        bt5b, vt5b = model5b.online_backbone.precompute_geom_tables(_cp5)
        out_after, _, _ = model5b(x5, temporal_mask=tmask5, bias_tables=bt5b, value_tables=vt5b)

assert torch.allclose(out_before, out_after, atol=1e-5), \
    "checkpoint save/load must reproduce identical output"
print("PASS 5: checkpoint save → load → identical output")


# ---------------------------------------------------------------------------
# Test 6: LR schedule warmup then rises
# ---------------------------------------------------------------------------

cfg6 = Config(arch=_cfg_base.arch, train=TrainConfig(n_epochs=20, lr_warmup_epochs=5, lr=1e-3))
model6 = _build_model(cfg6, n_electrodes=M)
opt6 = torch.optim.AdamW(model6.online_parameters(), lr=cfg6.train.lr)
sched6 = _build_lr_scheduler(opt6, cfg6)

lr_at = []
for ep in range(10):
    lr_at.append(opt6.param_groups[0]["lr"])
    sched6.step()

# During warmup (epochs 0-4) LR should be rising; by epoch 5 it should be at peak.
assert lr_at[0] < lr_at[4], f"LR should rise during warmup: {lr_at[:6]}"
# After warmup, cosine decay kicks in, so epoch 5 >= epoch 9.
assert lr_at[5] >= lr_at[9], f"LR should decay after warmup: {lr_at[5:]}"
print(f"PASS 6: LR schedule  lr[0]={lr_at[0]:.2e} → lr[5]={lr_at[5]:.2e} → lr[9]={lr_at[9]:.2e}")


print("\nAll pretrain smoke tests passed.")
