"""End-to-end v2 smoke: descriptor + spatial + backbone + losses + mixed corpus.

Run from repo root::

    PYTHONPATH=. .venv/bin/python tests/smoke_test_v2.py

Tests cover the v2 stack independently of v1 (no shared module state).
The per-module smoke blocks under src/v2/ verify each piece in isolation;
this file exercises the full forward + backward through a mixed-corpus
training step to make sure the parts compose.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.config import Config
from src.v2.datasets.mixed_corpus import MixedCorpus
from src.v2.losses import PretrainModelV2
from src.v2.model.backbone_v2 import BackboneV2
from src.v2.model.geometric_attention_v2 import (
    GeometricSpatialEncoderV2,
    build_geometric_descriptor_v2,
)
from src.v2.preprocess import ch_pos_from_names


def main() -> None:
    torch.manual_seed(0)
    cfg = Config()
    arch = cfg.arch
    T = cfg.preprocess.samples_per_epoch

    # --- Real ch_pos from the three datasets via the v2 helper ---
    names = {
        "physionet_mi": [
            "Fc5", "C3", "Cz", "C4", "Fc6", "Pz", "Oz", "Fpz",
        ],  # 8-channel subset is enough for the smoke
        "bcic_2b": ["C3", "Cz", "C4"],
        "sleep_edfx": ["FpzCz", "PzOz"],
    }
    ch_pos = {n: torch.from_numpy(ch_pos_from_names(ns, dataset=n))
              for n, ns in names.items()}

    # --- Per-dataset signal tensors (random, just for shapes) ---
    Xs = {
        "physionet_mi": torch.randn(40, len(names["physionet_mi"]), T),
        "bcic_2b":      torch.randn(20, 3, T),
        "sleep_edfx":   torch.randn(80, 2, T),
    }

    # --- One backbone per dataset (different M means different geometry
    # tables; the model itself is shared by deepcopy in PretrainModelV2). ---
    spatial = GeometricSpatialEncoderV2(
        d_model=arch.d_model,
        n_heads=arch.n_heads_spatial,
        n_layers=arch.n_layers_spatial,
        geometry_injection="g3",
        geom_mlp_hidden=arch.geom_mlp_hidden,
        dropout=arch.dropout,
    )
    backbone = BackboneV2(spatial, arch=arch)
    model = PretrainModelV2(backbone, lambda_R=1.0)

    # --- Mixed corpus, leave-one-out style (drop sleep for this smoke) ---
    corpus = MixedCorpus(
        datasets={
            "physionet_mi": (Xs["physionet_mi"], ch_pos["physionet_mi"]),
            "bcic_2b":      (Xs["bcic_2b"], ch_pos["bcic_2b"]),
        },
        batch_size=4,
        epochs_per_dataset=8,
        seed=1,
    )

    opt = torch.optim.AdamW(model.online_parameters(), lr=1e-4)

    T_p = arch.n_patches(T)
    n_steps = 0
    losses = []
    for name, batch in corpus:
        # Recompute geometry tables each step: the MLP weights change with
        # every optimizer step, so cached tables would reference stale
        # views (autograd will complain). Same pattern as v1's pretrain.
        bts, vts = backbone.precompute_geom_tables(ch_pos[name])
        tmask = torch.rand(batch.shape[0], T_p) < 0.5
        L, L_A, L_R = model(batch, tmask, bts, vts)
        opt.zero_grad()
        L.backward()
        opt.step()
        model.update_ema(tau=0.99)
        losses.append((name, L.item(), L_A.item(), L_R.item()))
        n_steps += 1

    assert n_steps == corpus.batches_per_pass, (n_steps, corpus.batches_per_pass)

    # Quick sanity: a third dataset (sleep) plugs into the same trained
    # backbone via its own geometry tables -- the zero-shot path.
    bts_s, vts_s = backbone.precompute_geom_tables(ch_pos["sleep_edfx"])
    with torch.no_grad():
        y = model.online_backbone(
            Xs["sleep_edfx"][:2], bias_tables=bts_s, value_tables=vts_s
        )
    assert y.shape == (2, 2, T_p, arch.d_model)

    # 10-D descriptor self-pair invariant.
    g = build_geometric_descriptor_v2(ch_pos["bcic_2b"])
    assert g.shape == (3, 3, 10)
    for i in range(3):
        assert g[i, i, 9].item() < 1e-6
        assert torch.allclose(g[i, i, 6:9], torch.zeros(3), atol=1e-6)

    print(f"smoke OK; {n_steps} mixed-corpus steps, losses[-1]={losses[-1]}")
    print(
        f"zero-shot to held-out sleep_edfx (M=2): backbone output shape "
        f"{tuple(y.shape)}"
    )


if __name__ == "__main__":
    main()
