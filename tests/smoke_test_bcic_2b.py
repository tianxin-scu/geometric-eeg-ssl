"""Smoke test for the BCIC IV-2b loader.

Run: PYTHONPATH=src python tests/smoke_test_bcic_2b.py
Loads subject 1 (one of 9) and asserts shape/label/position invariants.
First invocation downloads ~10 MB via MOABB; cached in ~/mne_data afterward.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from config import Config
from datasets.bcic_2b import BCIC2B, EEG_CHANNELS, LABEL_LEFT_HAND, LABEL_RIGHT_HAND


def main() -> None:
    cfg = Config()
    expected_T = cfg.preprocess.samples_per_epoch
    expected_M = 3

    print(f"Loading BCIC-2B subject 1; expecting M={expected_M}, T={expected_T}.")
    loader = BCIC2B(subjects=[1], cfg=cfg, mode="eval", verbose=True)
    X, y, ch_pos, ch_names = loader.load()
    print(f"X: {X.shape}  y: {y.shape}  ch_pos: {ch_pos.shape}  ch_names: {ch_names}")

    # Shape contracts.
    assert X.ndim == 3 and X.shape[1] == expected_M and X.shape[2] == expected_T, (
        f"Bad X shape {X.shape}; expected (N, {expected_M}, {expected_T})."
    )
    assert y.shape == (X.shape[0],), f"Label/epoch count mismatch: {y.shape} vs {X.shape[0]}"

    # Labels are binary.
    label_set = set(int(v) for v in np.unique(y))
    assert label_set <= {LABEL_LEFT_HAND, LABEL_RIGHT_HAND}, f"Unexpected labels: {label_set}"

    # Channels are exactly C3/Cz/C4 in standard order.
    assert tuple(ch_names) == EEG_CHANNELS, f"ch_names {ch_names} != {EEG_CHANNELS}"
    assert ch_pos.shape == (expected_M, 3), f"ch_pos shape {ch_pos.shape}"
    assert not np.isnan(ch_pos).any(), "ch_pos has NaN — montage lookup failed"

    # Coordinate normalization: centroid ~0, unit max-norm.
    if cfg.preprocess.coord_normalize:
        max_norm = float(np.linalg.norm(ch_pos, axis=1).max())
        assert abs(max_norm - 1.0) < 1e-5, f"Max-norm not 1: {max_norm}"

    # Per-epoch mean must be ~0 (z-score makes this exact). Don't assert
    # std==1: normalize_epochs's eps=1e-6 isn't volts-aware, so std is
    # systematically <1 across the pipeline (same behavior on PhysioNet MI).
    mean = X.mean(axis=-1)
    assert np.abs(mean).max() < 1e-3, f"Z-score mean drift: {np.abs(mean).max()}"
    assert np.isfinite(X).all(), "Non-finite values in X"

    # Class balance: both classes present.
    counts = np.bincount(y, minlength=2)
    assert counts.min() > 0, f"One class missing: counts={counts}"

    print(f"OK — {X.shape[0]} epochs, class counts {counts.tolist()}")


if __name__ == "__main__":
    main()
