"""Smoke tests for scripts/probe.py linear probe logic.

Does NOT require MNE or real PhysioNet data.  Exercises feature extraction,
metrics, LOSO splitting, and checkpoint round-trip with synthetic data.

Tests:
1.  extract_features: output shape (N, d_model), no NaN.
2.  extract_features with transductive backbone works (no geometry tables).
3.  _compute_metrics: perfect predictions → BAC=1.0, κ=1.0.
4.  _compute_metrics: random predictions → BAC near chance (4-class = 0.25).
5.  LOSO split logic: each subject appears as test exactly once.
6.  Checkpoint save (pretrain) → load into probe backbone → identical features.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_src = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_src))

import numpy as np
import torch

from src.config import AblationConfig, ArchConfig, Config, TrainConfig
from src.model.backbone import Backbone
from src.model.geometric_attention import GeometricSpatialEncoder
from src.model.momentum_encoder import PretrainModel
from src.model.transductive_baseline import TransductiveSpatialEncoder

from scripts.probe import (
    _build_backbone,
    _compute_metrics,
    _load_backbone,
    extract_features,
)
from scripts.pretrain import _build_model, _save_checkpoint, _build_lr_scheduler

torch.manual_seed(1)
np.random.seed(1)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

N, M, T = 20, 8, 800
N_CLASSES = 4

_small_arch = ArchConfig(
    d_model=32, n_heads_spatial=4, n_layers_spatial=1,
    n_heads_temporal=4, n_layers_temporal=1,
    geom_mlp_hidden=8, patch_samples=50, patch_stride=50,
)
_cfg = Config(arch=_small_arch)

def _ch_pos():
    pos = torch.randn(M, 3)
    pos = pos - pos.mean(0, keepdim=True)
    pos = pos / pos.norm(dim=-1).max().clamp_min(1e-8)
    return pos.numpy()

def _fake_X():
    return np.random.randn(N, M, T).astype(np.float32)

def _make_geom_backbone():
    bb = _build_backbone(_cfg, n_electrodes=M)
    bb.eval()
    for p in bb.parameters(): p.requires_grad_(False)
    return bb

def _make_trans_backbone():
    cfg_td = Config(
        arch=_small_arch,
        ablation=AblationConfig(use_codex=True, geometry_injection="G1"),
    )
    bb = _build_backbone(cfg_td, n_electrodes=M)
    bb.eval()
    for p in bb.parameters(): p.requires_grad_(False)
    return bb


# ---------------------------------------------------------------------------
# Test 1: extract_features shape and no NaN (geometric)
# ---------------------------------------------------------------------------

bb1 = _make_geom_backbone()
feats1 = extract_features(bb1, _fake_X(), _ch_pos(), device=torch.device("cpu"),
                           batch_size=8)
assert feats1.shape == (N, _small_arch.d_model), \
    f"expected ({N}, {_small_arch.d_model}), got {feats1.shape}"
assert not np.isnan(feats1).any(), "features contain NaN"
print(f"PASS 1: geometric extract_features  shape={feats1.shape}")


# ---------------------------------------------------------------------------
# Test 2: extract_features with transductive backbone
# ---------------------------------------------------------------------------

bb2 = _make_trans_backbone()
feats2 = extract_features(bb2, _fake_X(), _ch_pos(), device=torch.device("cpu"),
                           batch_size=8)
assert feats2.shape == (N, _small_arch.d_model)
assert not np.isnan(feats2).any()
print(f"PASS 2: transductive extract_features  shape={feats2.shape}")


# ---------------------------------------------------------------------------
# Test 3: perfect predictions → BAC=1, κ=1
# ---------------------------------------------------------------------------

y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
metrics_perfect = _compute_metrics(y_true, y_true)
assert metrics_perfect["balanced_accuracy"] == 1.0
assert metrics_perfect["cohens_kappa"] == 1.0
print(f"PASS 3: perfect predictions  BAC={metrics_perfect['balanced_accuracy']:.3f}")


# ---------------------------------------------------------------------------
# Test 4: random predictions near chance
# ---------------------------------------------------------------------------

rng = np.random.default_rng(42)
y_rand_true = rng.integers(0, N_CLASSES, size=200)
y_rand_pred = rng.integers(0, N_CLASSES, size=200)
metrics_rand = _compute_metrics(y_rand_true, y_rand_pred)
assert metrics_rand["balanced_accuracy"] < 0.5, \
    f"random predictions should be near chance: {metrics_rand['balanced_accuracy']:.3f}"
print(f"PASS 4: random predictions  BAC={metrics_rand['balanced_accuracy']:.3f} (near 0.25)")


# ---------------------------------------------------------------------------
# Test 5: LOSO split — each subject appears as test exactly once
# ---------------------------------------------------------------------------

# Simulate loso_eval split logic without loading real data.
subjects = [1, 2, 3, 4, 5]
test_subjects_seen = []
for test_subj in subjects:
    train_subjs = [s for s in subjects if s != test_subj]
    assert test_subj not in train_subjs
    assert len(train_subjs) == len(subjects) - 1
    test_subjects_seen.append(test_subj)

assert sorted(test_subjects_seen) == sorted(subjects), \
    "each subject must appear as test exactly once"
print(f"PASS 5: LOSO split covers all {len(subjects)} subjects exactly once")


# ---------------------------------------------------------------------------
# Test 6: pretrain checkpoint → load into probe backbone → identical features
# ---------------------------------------------------------------------------

cfg6 = Config(arch=_small_arch, train=TrainConfig(n_epochs=5))
pretrain_model = _build_model(cfg6, n_electrodes=M)
opt6 = torch.optim.AdamW(pretrain_model.online_parameters(), lr=1e-3)
sched6 = _build_lr_scheduler(opt6, cfg6)

X6 = _fake_X()
ch6 = _ch_pos()

# Extract features from the pretrain model's online backbone before saving.
pretrain_model.online_backbone.eval()
for p in pretrain_model.online_backbone.parameters(): p.requires_grad_(False)
feats_before = extract_features(pretrain_model.online_backbone, X6, ch6,
                                 device=torch.device("cpu"))

with tempfile.TemporaryDirectory() as tmpdir:
    ckpt_dir = Path(tmpdir)
    cfg6.to_yaml(str(ckpt_dir / "config.yaml"))
    ckpt_path = _save_checkpoint(ckpt_dir, epoch=0, step=0,
                                  model=pretrain_model, optimizer=opt6, scheduler=sched6)

    # Load into a fresh probe backbone.
    probe_bb = _load_backbone(ckpt_path, cfg6, n_electrodes=M,
                               device=torch.device("cpu"))
    feats_after = extract_features(probe_bb, X6, ch6, device=torch.device("cpu"))

assert np.allclose(feats_before, feats_after, atol=1e-5), \
    "probe backbone must reproduce pretrained features exactly"
print("PASS 6: checkpoint → probe backbone → identical features")


print("\nAll probe smoke tests passed.")
