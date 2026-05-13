"""Smoke test for the PhysioNet MI loader.

Run: python smoke_test_physionet_mi.py

Loads subjects 1 and 2, asserts shapes are consistent with the config,
prints the resolved config, and checks both pretrain and eval modes.
"""

from __future__ import annotations

import json

import numpy as np

from config import Config
from datasets.physionet_mi import PhysioNetMI


def main() -> None:
    cfg = Config()
    print("Resolved config:")
    print(json.dumps(cfg.resolved_dict(), indent=2, default=str))
    print()

    expected_T = cfg.preprocess.samples_per_epoch
    expected_M = 64  # PhysioNet MI channel count
    print(f"Expecting M={expected_M}, T={expected_T}")
    print()

    # ----- eval mode (default) -----
    print("Loading subjects [1, 2] in eval mode...")
    loader = PhysioNetMI(subjects=[1, 2], cfg=cfg, mode="eval", verbose=True)
    X, y, ch_pos, ch_names = loader.load()

    print(f"\nEval-mode shapes:")
    print(f"  X:       {X.shape}        dtype={X.dtype}")
    print(f"  y:       {y.shape}      dtype={y.dtype}")
    print(f"  ch_pos:  {ch_pos.shape}   dtype={ch_pos.dtype}")
    print(f"  ch_names[:5]: {ch_names[:5]}")
    print(f"  label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"  ch_pos range: x=[{ch_pos[:,0].min():.3f}, {ch_pos[:,0].max():.3f}], "
          f"y=[{ch_pos[:,1].min():.3f}, {ch_pos[:,1].max():.3f}], "
          f"z=[{ch_pos[:,2].min():.3f}, {ch_pos[:,2].max():.3f}]")

    # Assertions.
    assert X.ndim == 3, f"X should be (N, M, T), got {X.shape}"
    assert X.shape[1] == expected_M, f"M={X.shape[1]}, expected {expected_M}"
    assert X.shape[2] == expected_T, f"T={X.shape[2]}, expected {expected_T}"
    assert X.shape[0] == y.shape[0], "X and y first dim mismatch"
    assert X.dtype == np.float32
    assert ch_pos.shape == (expected_M, 3)
    assert len(ch_names) == expected_M
    assert set(np.unique(y)).issubset({0, 1, 2, 3}), f"labels: {np.unique(y)}"

    # z-score sanity: per-epoch, per-channel mean ~0, std ~1.
    means = X.mean(axis=-1)
    stds = X.std(axis=-1)
    assert np.abs(means).max() < 1e-3, f"z-score mean off: max |mean|={np.abs(means).max()}"
    assert np.abs(stds - 1.0).max() < 0.2, f"z-score std off: max |std-1|={np.abs(stds-1).max()}"

    # Coordinate normalization sanity: centered around origin, max norm ~1.
    if cfg.preprocess.coord_normalize:
        centroid = ch_pos.mean(axis=0)
        assert np.abs(centroid).max() < 1e-5, f"ch_pos not centered: {centroid}"
        max_norm = np.linalg.norm(ch_pos, axis=1).max()
        assert abs(max_norm - 1.0) < 1e-5, f"ch_pos not unit-scaled: max={max_norm}"

    print("\n[OK] eval-mode shape and normalization checks passed.")

    # ----- pretrain mode (no amplitude reject, sigma-clip applied) -----
    print("\nLoading subjects [1] in pretrain mode...")
    loader_pre = PhysioNetMI(subjects=[1], cfg=cfg, mode="pretrain", verbose=False)
    Xp, yp, _, _ = loader_pre.load()
    print(f"Pretrain-mode X: {Xp.shape}, |X|.max() = {np.abs(Xp).max():.3f}")
    assert np.abs(Xp).max() <= cfg.artifact.pretrain.clip_sigma + 1e-5

    print("\n[OK] pretrain-mode clip check passed.")
    print("\nAll smoke checks passed.")


if __name__ == "__main__":
    main()