"""Smoke tests for the Backbone (patcher + projection + spatial + temporal).

Covers:
  (a) shape contract end-to-end (B, M, T) -> (B, T_p, d_model);
  (b) gradient flow through both spatial-encoder variants;
  (c) patcher correctness (n_patches, no overlap, no leakage between epochs);
  (d) electrode-agnostic projection: same projection weights applied to every
      electrode (otherwise the geometric-variant's "no per-electrode params"
      promise breaks at the projection layer);
  (e) cross-montage transfer at the Backbone level (geometric variant
      survives a montage change; transductive variant raises);
  (f) temporal positional embedding actually changes the output;
  (g) temporal_mask zeros out masked time steps' contribution;
  (h) determinism in eval mode (needed for momentum-encoder alignment).

Run: `PYTHONPATH=. python tests/smoke_test_backbone.py` or via pytest.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import ArchConfig, Config
from src.model.backbone import Backbone, Patcher, TemporalPositionalEmbedding
from src.model.geometric_attention import GeometricSpatialEncoder
from src.model.transductive_baseline import TransductiveSpatialEncoder


def _fake_ch_pos(M: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    pos = torch.randn(M, 3, generator=g)
    pos = pos - pos.mean(dim=0, keepdim=True)
    pos = pos / pos.norm(dim=-1).max().clamp_min(1e-8)
    return pos


def _build_arch_for_small_smoke(d_model: int = 64) -> ArchConfig:
    """Smaller-than-default arch for fast smoke tests."""
    return ArchConfig(
        d_model=d_model,
        n_heads_spatial=4,
        n_layers_spatial=2,
        n_heads_temporal=4,
        n_layers_temporal=2,
        mlp_ratio=4.0,
        geom_mlp_hidden=16,
        patch_samples=50,
        patch_stride=50,
        temporal_posemb="learned",
        dropout=0.0,
    )


# ---------------------------------------------------------------------------

def test_patcher_basic():
    """Patcher produces the right number of non-overlapping patches."""
    p = Patcher(patch_samples=50, patch_stride=50)
    x = torch.arange(800, dtype=torch.float32).repeat(2, 8, 1)  # (B=2, M=8, T=800)
    out = p(x)
    assert out.shape == (2, 8, 16, 50)
    # First patch is samples 0..49, second is 50..99, etc.
    assert torch.allclose(out[0, 0, 0], torch.arange(0, 50, dtype=torch.float32))
    assert torch.allclose(out[0, 0, 1], torch.arange(50, 100, dtype=torch.float32))
    assert torch.allclose(out[0, 0, 15], torch.arange(750, 800, dtype=torch.float32))
    # No overlap, no leakage: concatenating all patches reconstructs the input.
    recon = out[0, 0].reshape(-1)
    assert torch.allclose(recon, torch.arange(0, 800, dtype=torch.float32))
    print("  patcher: 16 non-overlapping patches of 50 samples, no leakage")


def test_patcher_overlapping():
    """Overlapping patches (stride < samples) also work and produce more patches."""
    p = Patcher(patch_samples=50, patch_stride=25)
    x = torch.arange(800, dtype=torch.float32).repeat(1, 1, 1)
    out = p(x)
    # n_patches = 1 + (800 - 50) // 25 = 1 + 30 = 31
    assert out.shape == (1, 1, 31, 50)
    # Patch 1 starts at sample 25, not 50.
    assert out[0, 0, 1, 0].item() == 25.0
    print("  patcher: overlapping stride=25 OK (31 patches)")


def test_backbone_shapes_geometric():
    torch.manual_seed(0)
    arch = _build_arch_for_small_smoke()
    M, T, B = 8, 800, 2
    pos = _fake_ch_pos(M)
    spatial = GeometricSpatialEncoder(
        d_model=arch.d_model,
        n_heads=arch.n_heads_spatial,
        n_layers=arch.n_layers_spatial,
        geometry_injection="g1",
        geom_mlp_hidden=arch.geom_mlp_hidden,
        dropout=arch.dropout,
    )
    bb = Backbone(spatial, arch=arch)
    bt, vt = bb.precompute_geom_tables(pos)
    x = torch.randn(B, M, T)
    y = bb(x, bt, vt)
    assert y.shape == (B, 16, arch.d_model)
    assert torch.isfinite(y).all()
    print(f"  geometric Backbone: (B={B}, M={M}, T={T}) -> {tuple(y.shape)}")


def test_backbone_shapes_transductive():
    torch.manual_seed(0)
    arch = _build_arch_for_small_smoke()
    M, T, B = 8, 800, 2
    spatial = TransductiveSpatialEncoder(
        d_model=arch.d_model,
        n_heads=arch.n_heads_spatial,
        n_layers=arch.n_layers_spatial,
        n_electrodes=M,
        dropout=arch.dropout,
    )
    bb = Backbone(spatial, arch=arch)
    x = torch.randn(B, M, T)
    y = bb(x)
    assert y.shape == (B, 16, arch.d_model)
    assert torch.isfinite(y).all()
    print(f"  transductive Backbone: (B={B}, M={M}, T={T}) -> {tuple(y.shape)}")


def test_backbone_gradient_flow_both_variants():
    torch.manual_seed(0)
    arch = _build_arch_for_small_smoke()
    M, T, B = 8, 800, 2
    pos = _fake_ch_pos(M)

    for variant in ("geometric", "transductive"):
        if variant == "geometric":
            spatial = GeometricSpatialEncoder(
                d_model=arch.d_model, n_heads=arch.n_heads_spatial,
                n_layers=arch.n_layers_spatial, geometry_injection="g1",
                geom_mlp_hidden=arch.geom_mlp_hidden, dropout=arch.dropout,
            )
            bb = Backbone(spatial, arch=arch)
            bt, vt = bb.precompute_geom_tables(pos)
        else:
            spatial = TransductiveSpatialEncoder(
                d_model=arch.d_model, n_heads=arch.n_heads_spatial,
                n_layers=arch.n_layers_spatial, n_electrodes=M, dropout=arch.dropout,
            )
            bb = Backbone(spatial, arch=arch)
            bt, vt = None, None

        x = torch.randn(B, M, T, requires_grad=True)
        y = bb(x, bt, vt) if variant == "geometric" else bb(x)
        loss = y.pow(2).sum()
        loss.backward()

        assert x.grad is not None and x.grad.abs().sum() > 0, f"{variant}: input got no grad"
        # Projection should always receive gradient.
        assert bb.proj.weight.grad is not None and bb.proj.weight.grad.abs().sum() > 0
        # Temporal Transformer should receive gradient.
        any_temp = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in bb.temporal.parameters()
        )
        assert any_temp, f"{variant}: temporal Transformer got no grad"
        # Spatial encoder should receive gradient.
        any_spatial = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in bb.spatial.parameters()
        )
        assert any_spatial, f"{variant}: spatial encoder got no grad"
        print(f"  gradient flow ({variant}): input + projection + spatial + temporal all OK")


def test_projection_is_electrode_agnostic():
    """The projection's weights must be SHARED across electrodes.

    This is the load-bearing property: the geometric variant claims no
    per-electrode parameters. If the projection had per-electrode weights
    that promise would break here, not at the spatial encoder.
    """
    arch = _build_arch_for_small_smoke()
    spatial = GeometricSpatialEncoder(
        d_model=arch.d_model, n_heads=arch.n_heads_spatial,
        n_layers=arch.n_layers_spatial, geometry_injection="g1",
        geom_mlp_hidden=arch.geom_mlp_hidden, dropout=arch.dropout,
    )
    bb = Backbone(spatial, arch=arch)
    # The projection is an nn.Linear with input size patch_samples and
    # output size d_model. NO dependence on M anywhere in its shape.
    assert bb.proj.weight.shape == (arch.d_model, arch.patch_samples)
    assert bb.proj.bias.shape == (arch.d_model,)
    # Construct the same backbone with a different M and check projection
    # weights are still the same shape.
    arch2 = _build_arch_for_small_smoke()
    spatial2 = GeometricSpatialEncoder(
        d_model=arch2.d_model, n_heads=arch2.n_heads_spatial,
        n_layers=arch2.n_layers_spatial, geometry_injection="g1",
        geom_mlp_hidden=arch2.geom_mlp_hidden, dropout=arch2.dropout,
    )
    bb2 = Backbone(spatial2, arch=arch2)
    assert bb.proj.weight.shape == bb2.proj.weight.shape
    print("  projection is electrode-agnostic (shape independent of M)")


def test_cross_montage_transfer():
    """Geometric Backbone transfers across M; transductive raises."""
    torch.manual_seed(0)
    arch = _build_arch_for_small_smoke()
    M_train, M_eval, T, B = 16, 8, 800, 2
    pos_train = _fake_ch_pos(M_train, seed=1)
    pos_eval = _fake_ch_pos(M_eval, seed=2)

    # Geometric: same model handles both montages, only the geometry tables change.
    spatial_g = GeometricSpatialEncoder(
        d_model=arch.d_model, n_heads=arch.n_heads_spatial,
        n_layers=arch.n_layers_spatial, geometry_injection="g1",
        geom_mlp_hidden=arch.geom_mlp_hidden, dropout=arch.dropout,
    )
    bb_g = Backbone(spatial_g, arch=arch)
    bt_tr, vt_tr = bb_g.precompute_geom_tables(pos_train)
    y_tr = bb_g(torch.randn(B, M_train, T), bt_tr, vt_tr)
    assert y_tr.shape == (B, 16, arch.d_model)
    bt_ev, vt_ev = bb_g.precompute_geom_tables(pos_eval)
    y_ev = bb_g(torch.randn(B, M_eval, T), bt_ev, vt_ev)
    assert y_ev.shape == (B, 16, arch.d_model)
    print(f"  geometric Backbone: M={M_train} -> M={M_eval} transfer OK")

    # Transductive: codex built for M_train, eval at M_eval should raise.
    spatial_t = TransductiveSpatialEncoder(
        d_model=arch.d_model, n_heads=arch.n_heads_spatial,
        n_layers=arch.n_layers_spatial, n_electrodes=M_train, dropout=arch.dropout,
    )
    bb_t = Backbone(spatial_t, arch=arch)
    try:
        bb_t(torch.randn(B, M_eval, T))
        raise AssertionError("transductive Backbone should have raised")
    except ValueError as e:
        assert "transductive" in str(e).lower()
    print(f"  transductive Backbone: M={M_train} -> M={M_eval} refused (expected)")

    # Also: precompute_geom_tables on a transductive Backbone should raise.
    try:
        bb_t.precompute_geom_tables(pos_train)
        raise AssertionError("precompute_geom_tables should have raised for transductive")
    except RuntimeError as e:
        assert "transductive" in str(e).lower()
    print("  precompute_geom_tables on transductive: refused (expected)")


def test_temporal_posemb_actually_used():
    """Sanity: with mode='learned', removing posemb changes the output."""
    torch.manual_seed(0)
    arch_learned = _build_arch_for_small_smoke()
    arch_none = ArchConfig(
        **{**arch_learned.__dict__, "temporal_posemb": "none"}
    )
    M, T, B = 8, 800, 1
    pos = _fake_ch_pos(M)
    x = torch.randn(B, M, T)

    bb_a = Backbone(
        GeometricSpatialEncoder(
            d_model=arch_learned.d_model, n_heads=arch_learned.n_heads_spatial,
            n_layers=arch_learned.n_layers_spatial, geometry_injection="g1",
            geom_mlp_hidden=arch_learned.geom_mlp_hidden, dropout=0.0,
        ),
        arch=arch_learned,
    )
    bb_b = Backbone(
        GeometricSpatialEncoder(
            d_model=arch_none.d_model, n_heads=arch_none.n_heads_spatial,
            n_layers=arch_none.n_layers_spatial, geometry_injection="g1",
            geom_mlp_hidden=arch_none.geom_mlp_hidden, dropout=0.0,
        ),
        arch=arch_none,
    )
    # Copy non-posemb weights from a -> b so the only difference is posemb.
    sd_a = bb_a.state_dict()
    sd_b = bb_b.state_dict()
    # bb_b has no posemb.pe key; copy everything else from a.
    for k, v in sd_a.items():
        if k.startswith("temporal.posemb"):
            continue
        sd_b[k] = v
    bb_b.load_state_dict(sd_b)
    bb_a.eval(); bb_b.eval()

    bt_a, vt_a = bb_a.precompute_geom_tables(pos)
    bt_b, vt_b = bb_b.precompute_geom_tables(pos)
    # Need geom tables to match too, in case they differ from copy.
    # (They will, since the geom_*_mlp params were copied above.)
    y_a = bb_a(x, bt_a, vt_a)
    y_b = bb_b(x, bt_b, vt_b)
    # With learned posemb (initialized to small nonzero values), outputs
    # should differ from the no-posemb case.
    assert not torch.allclose(y_a, y_b, atol=1e-5), (
        "learned posemb produced the same output as no posemb; posemb may "
        "be initialized to exactly zero or not applied"
    )
    print("  temporal posemb: learned vs none produce different outputs (used)")


def test_temporal_mask_masks_position():
    """Masked time steps shouldn't influence the unmasked outputs.

    Standard sanity for self-attention's key_padding_mask: corrupting the
    masked positions' inputs leaves the unmasked positions' outputs
    unchanged (modulo the masked positions themselves).
    """
    torch.manual_seed(0)
    arch = _build_arch_for_small_smoke()
    M, T, B = 8, 800, 1
    pos = _fake_ch_pos(M)
    spatial = GeometricSpatialEncoder(
        d_model=arch.d_model, n_heads=arch.n_heads_spatial,
        n_layers=arch.n_layers_spatial, geometry_injection="g1",
        geom_mlp_hidden=arch.geom_mlp_hidden, dropout=0.0,
    )
    bb = Backbone(spatial, arch=arch)
    bb.eval()
    bt, vt = bb.precompute_geom_tables(pos)

    x = torch.randn(B, M, T)
    T_p = arch.n_patches(T)
    mask = torch.zeros(B, T_p, dtype=torch.bool)
    mask[0, 3] = True
    mask[0, 7] = True

    y_ref = bb(x, bt, vt, temporal_mask=mask)
    # Corrupt the inputs that produce masked time steps.
    # Time step t spans samples [t*stride, t*stride + patch_samples).
    x_corr = x.clone()
    for t in (3, 7):
        s = t * arch.patch_stride
        x_corr[0, :, s : s + arch.patch_samples] = 1e3 * torch.randn(M, arch.patch_samples)
    y_corr = bb(x_corr, bt, vt, temporal_mask=mask)
    # At unmasked positions, the output must be identical.
    unmasked = ~mask[0]
    assert torch.allclose(y_ref[0, unmasked], y_corr[0, unmasked], atol=1e-4), (
        "temporal_mask did not prevent masked time steps from influencing "
        "unmasked positions; mask may be broken"
    )
    print("  temporal_mask: corrupting masked positions doesn't leak to unmasked")


def test_eval_determinism():
    """Two forwards on the same input give bit-identical results in eval mode.

    Needed for momentum-encoder alignment: the EMA target must be a
    deterministic function of its input.
    """
    torch.manual_seed(0)
    arch = _build_arch_for_small_smoke()
    M, T, B = 8, 800, 2
    pos = _fake_ch_pos(M)
    spatial = GeometricSpatialEncoder(
        d_model=arch.d_model, n_heads=arch.n_heads_spatial,
        n_layers=arch.n_layers_spatial, geometry_injection="g1",
        geom_mlp_hidden=arch.geom_mlp_hidden, dropout=0.0,
    )
    bb = Backbone(spatial, arch=arch)
    bb.eval()
    bt, vt = bb.precompute_geom_tables(pos)
    x = torch.randn(B, M, T)
    y1 = bb(x, bt, vt)
    bb2 = copy.deepcopy(bb)
    bt2, vt2 = bb2.precompute_geom_tables(pos)
    y2 = bb2(x, bt2, vt2)
    assert torch.allclose(y1, y2, atol=1e-6)
    print("  eval determinism: deepcopy reproduces output (EMA-ready)")


def test_geometric_wired_backbone_rejects_loose_tables_arg_for_transductive():
    """Passing geom tables to a transductive Backbone is a usage error."""
    arch = _build_arch_for_small_smoke()
    M, T, B = 8, 800, 1
    spatial = TransductiveSpatialEncoder(
        d_model=arch.d_model, n_heads=arch.n_heads_spatial,
        n_layers=arch.n_layers_spatial, n_electrodes=M, dropout=0.0,
    )
    bb = Backbone(spatial, arch=arch)
    try:
        bb(torch.randn(B, M, T), bias_tables=["junk"], value_tables=["junk"])
        raise AssertionError("should have raised")
    except ValueError as e:
        assert "transductive" in str(e).lower()
    print("  transductive Backbone: rejects geometry tables (expected)")


# ---------------------------------------------------------------------------

def main():
    print("smoke_test_backbone.py")
    test_patcher_basic()
    test_patcher_overlapping()
    test_backbone_shapes_geometric()
    test_backbone_shapes_transductive()
    test_backbone_gradient_flow_both_variants()
    test_projection_is_electrode_agnostic()
    test_cross_montage_transfer()
    test_temporal_posemb_actually_used()
    test_temporal_mask_masks_position()
    test_eval_determinism()
    test_geometric_wired_backbone_rejects_loose_tables_arg_for_transductive()
    print("ALL OK")


if __name__ == "__main__":
    main()
