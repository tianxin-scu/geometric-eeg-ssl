"""Smoke tests for the geometric attention layer and transductive baseline.

Specifically exercises the three things flagged in session 2's "for next
session" note:
  (a) gradient flow through G1/G2/G3 and the transductive baseline,
  (b) parameter-count comparison between geometric and codex variants,
  (c) behavior when M changes between train and test (the cross-montage
      transfer demo that motivates the whole proposal).

Plus a couple of correctness checks that are easy to get wrong:
  - key_padding_mask actually zeros out masked electrodes' contribution,
  - geometric descriptor is asymmetric in (i, j) for non-distance channels,
  - CLS row/col of geometry tables is zero (no spurious geometric signal
    leaking into the CLS via padding),
  - momentum-encoder forward path (we don't have an EMA wrapper yet, but
    a deepcopy of the model produces the same output for the same input,
    confirming the forward pass is deterministic).

Run with `python -m tests.smoke_test_geometric_attention` or directly.
"""

from __future__ import annotations

import sys
import copy
from pathlib import Path

import torch

# Allow running as a script without installing the package.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.model.geometric_attention import (
    build_geometric_descriptor,
    GeometricSpatialEncoder,
)
from src.model.transductive_baseline import (
    TransductiveSpatialEncoder,
    count_distinguishing_parameters,
)


def _fake_ch_pos(M: int, seed: int = 0) -> torch.Tensor:
    """Centroid-subtract + unit-max-norm scale, matching preprocess.py."""
    g = torch.Generator().manual_seed(seed)
    pos = torch.randn(M, 3, generator=g)
    pos = pos - pos.mean(dim=0, keepdim=True)
    max_norm = pos.norm(dim=-1).max()
    pos = pos / max_norm.clamp_min(1e-8)
    return pos


# ---------------------------------------------------------------------------

def test_descriptor_shape_and_asymmetry():
    M = 8
    pos = _fake_ch_pos(M)
    g = build_geometric_descriptor(pos)
    assert g.shape == (M, M, 4)
    # Distance is symmetric: g[i,j,0] == g[j,i,0]
    assert torch.allclose(g[..., 0], g[..., 0].T, atol=1e-6)
    # Displacement is antisymmetric: g[i,j,1:] == -g[j,i,1:]
    assert torch.allclose(g[..., 1:], -g[..., 1:].transpose(0, 1), atol=1e-6)
    # Self-pairs are zero.
    diag = torch.diagonal(g, dim1=0, dim2=1).T  # (M, 4)
    assert torch.allclose(diag, torch.zeros_like(diag), atol=1e-6)
    print("  descriptor: shape OK, symmetric distance, antisymmetric displacement")


def test_geometry_tables_cls_padding():
    """The CLS token must have zero geometric relationship to electrodes."""
    M, d, H, L = 8, 64, 4, 2
    pos = _fake_ch_pos(M)
    enc = GeometricSpatialEncoder(
        d_model=d, n_heads=H, n_layers=L, geometry_injection="g3"
    )
    bts, vts = enc.precompute_geom_tables(pos)
    for bt, vt in zip(bts, vts):
        assert bt.shape == (M + 1, M + 1, H)
        assert vt.shape == (M + 1, M + 1, H, d // H)
        assert torch.allclose(bt[0, :, :], torch.zeros_like(bt[0, :, :]))
        assert torch.allclose(bt[:, 0, :], torch.zeros_like(bt[:, 0, :]))
        assert torch.allclose(vt[0, :, :, :], torch.zeros_like(vt[0, :, :, :]))
        assert torch.allclose(vt[:, 0, :, :], torch.zeros_like(vt[:, 0, :, :]))
    print("  geom tables: CLS row/col zero (no geometry leak through CLS)")


def test_gradient_flow_all_modes():
    """G1, G2, G3, and none all produce nonzero gradients on inputs and params."""
    torch.manual_seed(0)
    B, M, d = 2, 8, 64
    pos = _fake_ch_pos(M)
    for mode in ("g1", "g2", "g3", "none"):
        enc = GeometricSpatialEncoder(
            d_model=d, n_heads=4, n_layers=2, geometry_injection=mode
        )
        if mode == "none":
            bts, vts = [None] * 2, [None] * 2
        else:
            bts, vts = enc.precompute_geom_tables(pos)
        x = torch.randn(B, M, d, requires_grad=True)
        s = enc(x, bts, vts)
        assert s.shape == (B, d)
        loss = s.pow(2).sum()
        loss.backward()
        assert x.grad is not None and x.grad.abs().sum() > 0, mode
        # At least one parameter has a nonzero gradient.
        any_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in enc.parameters()
        )
        assert any_grad, mode
        # Specifically: if mode uses geom_bias_mlp, it should receive grad.
        if mode in ("g1", "g3"):
            grads = [
                p.grad.abs().sum().item()
                for n, p in enc.named_parameters()
                if "geom_bias_mlp" in n and p.grad is not None
            ]
            assert grads and sum(grads) > 0, f"{mode}: geom_bias_mlp got no grad"
        if mode in ("g2", "g3"):
            grads = [
                p.grad.abs().sum().item()
                for n, p in enc.named_parameters()
                if "geom_value_mlp" in n and p.grad is not None
            ]
            assert grads and sum(grads) > 0, f"{mode}: geom_value_mlp got no grad"
        print(f"  gradient flow ({mode}): OK")


def test_transductive_gradient_flow():
    torch.manual_seed(0)
    B, M, d = 2, 8, 64
    enc = TransductiveSpatialEncoder(
        d_model=d, n_heads=4, n_layers=2, n_electrodes=M
    )
    x = torch.randn(B, M, d, requires_grad=True)
    s = enc(x)
    s.pow(2).sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    # Codex must receive gradient.
    codex_grads = [
        p.grad.abs().sum().item()
        for n, p in enc.named_parameters() if "codex" in n and p.grad is not None
    ]
    assert codex_grads and sum(codex_grads) > 0
    print("  gradient flow (transductive/codex): OK")


def test_cross_montage_transfer():
    """The whole point of the proposal.

    Train the geometric encoder on M_train electrodes, eval on M_eval
    electrodes (different montage). It should run without error and produce
    a finite output. The transductive baseline must FAIL when the codex
    size doesn't match M_eval — also part of the claim.
    """
    torch.manual_seed(0)
    M_train, M_eval, d = 64, 19, 64  # 64-ch pretrain, 19-ch consumer eval
    pos_train = _fake_ch_pos(M_train, seed=1)
    pos_eval = _fake_ch_pos(M_eval, seed=2)

    # --- Geometric: transfers cleanly. ---
    enc_g = GeometricSpatialEncoder(
        d_model=d, n_heads=4, n_layers=2, geometry_injection="g1"
    )
    # Train pass (just one forward to mimic having pretrained):
    bts_tr, vts_tr = enc_g.precompute_geom_tables(pos_train)
    x_tr = torch.randn(2, M_train, d)
    s_tr = enc_g(x_tr, bts_tr, vts_tr)
    assert s_tr.shape == (2, d) and torch.isfinite(s_tr).all()
    # Eval pass on a different montage — SAME enc_g, just new geometry tables:
    bts_ev, vts_ev = enc_g.precompute_geom_tables(pos_eval)
    x_ev = torch.randn(2, M_eval, d)
    s_ev = enc_g(x_ev, bts_ev, vts_ev)
    assert s_ev.shape == (2, d) and torch.isfinite(s_ev).all()
    print(f"  geometric: 64-ch -> 19-ch transfer OK (no retraining)")

    # --- Transductive: refuses to run on a different M. ---
    enc_t = TransductiveSpatialEncoder(
        d_model=d, n_heads=4, n_layers=2, n_electrodes=M_train
    )
    x_ev = torch.randn(2, M_eval, d)
    try:
        enc_t(x_ev)
        raise AssertionError("transductive should have raised on M mismatch")
    except ValueError as e:
        assert "transductive" in str(e).lower()
    print("  transductive: 64-ch -> 19-ch refused with ValueError (expected)")


def test_key_padding_mask_effect():
    """Masked electrodes must not influence the output.

    Run twice: once with mask, once with the masked electrodes' values
    replaced by junk. Outputs should match.
    """
    torch.manual_seed(0)
    B, M, d = 1, 8, 32
    pos = _fake_ch_pos(M)
    enc = GeometricSpatialEncoder(
        d_model=d, n_heads=4, n_layers=2, geometry_injection="g1"
    )
    bts, vts = enc.precompute_geom_tables(pos)
    enc.eval()
    x = torch.randn(B, M, d)
    mask = torch.zeros(B, M, dtype=torch.bool)
    mask[0, 3] = True
    mask[0, 5] = True
    s_ref = enc(x, bts, vts, key_padding_mask=mask)
    # Now corrupt masked electrodes; output should be identical.
    x_corr = x.clone()
    x_corr[0, 3] = 1e3 * torch.randn(d)
    x_corr[0, 5] = -1e3 * torch.randn(d)
    s_corr = enc(x_corr, bts, vts, key_padding_mask=mask)
    assert torch.allclose(s_ref, s_corr, atol=1e-4), (
        "masked electrodes influenced output; key_padding_mask is broken"
    )
    print("  key_padding_mask: masked electrodes have no effect on CLS")


def test_parameter_count_comparison():
    """Print a side-by-side breakdown so the student can decide on codex_rank.

    Defaults from project_direction_summary.md §3: d=256, H=8, L=2.
    For PhysioNet MI: M=64.
    """
    d, H, L, M = 256, 8, 2, 64

    enc_g = GeometricSpatialEncoder(
        d_model=d, n_heads=H, n_layers=L, geometry_injection="g1"
    )
    enc_g2 = GeometricSpatialEncoder(
        d_model=d, n_heads=H, n_layers=L, geometry_injection="g2"
    )
    enc_g3 = GeometricSpatialEncoder(
        d_model=d, n_heads=H, n_layers=L, geometry_injection="g3"
    )
    enc_t_full = TransductiveSpatialEncoder(
        d_model=d, n_heads=H, n_layers=L, n_electrodes=M
    )

    def fmt(model, name):
        b = count_distinguishing_parameters(model)
        return (
            f"  {name:20s}  total={b['total']:>8d}  "
            f"distinguishing={b['codex'] + b['geom_bias'] + b['geom_value']:>6d}  "
            f"(codex={b['codex']}, geom_bias={b['geom_bias']}, "
            f"geom_value={b['geom_value']})"
        )

    print("  parameter breakdown (d=256, H=8, L=2, M=64):")
    print(fmt(enc_g, "geometric G1"))
    print(fmt(enc_g2, "geometric G2"))
    print(fmt(enc_g3, "geometric G3"))
    print(fmt(enc_t_full, "codex (full rank)"))
    # Sanity: total counts differ ONLY by the distinguishing parameters,
    # since QKVO/MLP/Norm/CLS are identical between geometric and codex.
    b_g = count_distinguishing_parameters(enc_g)
    b_t = count_distinguishing_parameters(enc_t_full)
    shared_g = b_g["attn_qkvo"] + b_g["mlp"] + b_g["norm"] + b_g["cls"] + b_g["other"]
    shared_t = b_t["attn_qkvo"] + b_t["mlp"] + b_t["norm"] + b_t["cls"] + b_t["other"]
    assert shared_g == shared_t, (
        f"shared params differ: geom={shared_g} vs codex={shared_t}; "
        "the headline comparison won't be apples-to-apples"
    )
    print(f"  shared (QKVO + MLP + Norm + CLS) = {shared_g} params, identical OK")


def test_deterministic_forward_for_ema():
    """Cheap sanity: two forward passes on the same input give the same output.

    The momentum encoder is a deepcopy of the online encoder; if forward
    weren't deterministic, the alignment loss would be ill-defined.
    """
    torch.manual_seed(0)
    B, M, d = 2, 8, 64
    pos = _fake_ch_pos(M)
    enc = GeometricSpatialEncoder(
        d_model=d, n_heads=4, n_layers=2, geometry_injection="g1"
    )
    enc.eval()
    bts, vts = enc.precompute_geom_tables(pos)
    x = torch.randn(B, M, d)
    s1 = enc(x, bts, vts)
    enc_copy = copy.deepcopy(enc)
    bts2, vts2 = enc_copy.precompute_geom_tables(pos)
    s2 = enc_copy(x, bts2, vts2)
    assert torch.allclose(s1, s2, atol=1e-6)
    print("  deterministic forward: deepcopy reproduces output (EMA-ready)")


# ---------------------------------------------------------------------------

def main():
    print("smoke_test_geometric_attention.py")
    test_descriptor_shape_and_asymmetry()
    test_geometry_tables_cls_padding()
    test_gradient_flow_all_modes()
    test_transductive_gradient_flow()
    test_cross_montage_transfer()
    test_key_padding_mask_effect()
    test_parameter_count_comparison()
    test_deterministic_forward_for_ema()
    print("ALL OK")


if __name__ == "__main__":
    main()
