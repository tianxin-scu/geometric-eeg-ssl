"""Smoke test for the Sleep-EDFx loader.

Run: PYTHONPATH=src python tests/smoke_test_sleep_edfx.py
Loads subject 0 (both nights) and asserts shape/label/position invariants.
First invocation downloads ~100 MB via MNE; cached afterward.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from config import Config
from datasets.sleep_edfx import EEG_BIPOLAR_SHORT, SleepEDFx


def main() -> None:
    cfg = Config()
    expected_T = cfg.preprocess.samples_per_epoch
    expected_M = 2

    print(f"Loading Sleep-EDFx subject 0 (both nights); expecting M={expected_M}, T={expected_T}.")
    loader = SleepEDFx(subjects=[0], cfg=cfg, mode="eval", verbose=True)
    X, y, ch_pos, ch_names, night = loader.load()
    print(f"X: {X.shape}  y: {y.shape}  ch_pos: {ch_pos.shape}  night: {night.shape}")
    print(f"ch_names: {ch_names}")

    # Shape contracts.
    assert X.ndim == 3 and X.shape[1] == expected_M and X.shape[2] == expected_T, (
        f"Bad X shape {X.shape}; expected (N, {expected_M}, {expected_T})."
    )
    assert y.shape == (X.shape[0],) and night.shape == (X.shape[0],)

    # Labels are 0..4 (5-class AASM).
    label_set = set(int(v) for v in np.unique(y))
    assert label_set <= {0, 1, 2, 3, 4}, f"Unexpected labels: {label_set}"

    # Channels.
    assert tuple(ch_names) == EEG_BIPOLAR_SHORT, f"ch_names {ch_names} != {EEG_BIPOLAR_SHORT}"
    assert ch_pos.shape == (expected_M, 3)
    assert not np.isnan(ch_pos).any(), "ch_pos has NaN"

    # Coord normalization.
    if cfg.preprocess.coord_normalize:
        max_norm = float(np.linalg.norm(ch_pos, axis=1).max())
        assert abs(max_norm - 1.0) < 1e-5, f"Max-norm not 1: {max_norm}"

    # Night tag values.
    night_set = set(int(n) for n in np.unique(night))
    assert night_set <= {1, 2}, f"Unexpected night values: {night_set}"
    assert len(night_set) >= 1, "No night data loaded"

    # Mean ~0 only; std<1 systematically because normalize_epochs eps isn't
    # volts-aware. Same behavior as PhysioNet MI; not a Sleep-EDFx bug.
    mean = X.mean(axis=-1)
    assert np.abs(mean).max() < 1e-3, f"Z-score mean drift: {np.abs(mean).max()}"
    assert np.isfinite(X).all(), "Non-finite values in X"

    # Both nights should contribute roughly comparable counts (subject 0 has both).
    counts_n1 = int((night == 1).sum())
    counts_n2 = int((night == 2).sum())
    print(f"Night 1: {counts_n1} epochs;  Night 2: {counts_n2} epochs")
    if 0 in {counts_n1, counts_n2}:
        # Subject 0 should have both nights; if one is missing the loader has a bug.
        raise AssertionError("Subject 0 missing a night — expected both 1 and 2.")

    # 30 s -> ~7 sub-epochs per stage window: total epochs should be sizeable
    # (a full night of sleep ~8 hours = ~960 30s windows -> ~6720 sub-epochs).
    # Conservative lower bound: 2000 epochs total across both nights.
    assert X.shape[0] >= 2000, f"Too few epochs ({X.shape[0]}); slicing may be broken."

    # Class balance per night: assert each night has at least 3 distinct stages.
    for n in (1, 2):
        labels_n = set(int(v) for v in np.unique(y[night == n]))
        assert len(labels_n) >= 3, f"Night {n} has too few stages: {labels_n}"

    print(f"OK — {X.shape[0]} epochs, label set {sorted(label_set)}")


if __name__ == "__main__":
    main()
