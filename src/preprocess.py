"""Shared preprocessing: applied identically to every dataset.

The contract: in -> raw MNE Raw with a standard-named channel set.
              out -> the same Raw, bandpassed, resampled, montaged.

Epoching and per-epoch operations (detrend, z-score, clip, amplitude reject)
live in `epoch_and_normalize`, which operates on numpy arrays so it works
identically regardless of where epochs came from (fixed-length windows for
SSL pretrain vs event-locked epochs for downstream eval).
"""

from __future__ import annotations

from typing import Optional

import mne
import numpy as np

from config import PreprocessConfig


def preprocess_raw(raw: mne.io.BaseRaw, cfg: PreprocessConfig) -> mne.io.BaseRaw:
    """Apply Tier 1 preprocessing to a Raw object in place (copy first if needed).

    Order matters:
      1. Set montage (so channel positions exist before any spatial ops).
      2. Bandpass (zero-phase FIR, MNE default).
      3. Resample (after bandpass, before any epoching).

    No re-referencing, no ICA -- intentional per design discussion.
    """
    raw.load_data()

    # 1. Montage -- attach standard 10-05 positions by channel name.
    montage = mne.channels.make_standard_montage(cfg.montage)
    raw.set_montage(montage, match_case=False, on_missing="warn")

    # 2. Bandpass. Zero-phase FIR; MNE picks a sensible filter length.
    raw.filter(
        l_freq=cfg.bandpass_hz[0],
        h_freq=cfg.bandpass_hz[1],
        method="fir",
        phase="zero",
        fir_design="firwin",
        verbose="WARNING",
    )

    # 3. Resample. Done after filtering to avoid aliasing.
    if raw.info["sfreq"] != cfg.resample_hz:
        raw.resample(cfg.resample_hz, verbose="WARNING")

    return raw


def get_channel_positions(
    raw: mne.io.BaseRaw,
    cfg: PreprocessConfig,
) -> np.ndarray:
    """Return (M, 3) array of channel positions in the order of raw.ch_names.

    Coordinates are in MNE's head frame (meters). If cfg.coord_normalize,
    the centroid is subtracted and positions are scaled to unit max-norm,
    yielding values roughly in [-1, 1]. Use these for g_ij.
    """
    pos_dict = raw.get_montage().get_positions()["ch_pos"]
    pos = np.array([pos_dict[ch] for ch in raw.ch_names], dtype=np.float32)

    if np.isnan(pos).any():
        bad = [ch for ch, p in zip(raw.ch_names, pos) if np.isnan(p).any()]
        raise ValueError(f"Channels missing 3-D positions: {bad}")

    if cfg.coord_normalize:
        pos = pos - pos.mean(axis=0, keepdims=True)
        scale = np.linalg.norm(pos, axis=1).max()
        if scale > 0:
            pos = pos / scale

    return pos


def _detrend_epoch(x: np.ndarray, mode: str) -> np.ndarray:
    """Detrend along the last axis. x: (..., T)."""
    if mode == "none":
        return x
    if mode == "constant":
        return x - x.mean(axis=-1, keepdims=True)
    if mode == "linear":
        # Subtract least-squares line per (..., T) slice.
        T = x.shape[-1]
        t = np.arange(T, dtype=x.dtype)
        t_centered = t - t.mean()
        denom = (t_centered ** 2).sum()
        # x: (..., T); compute slope per slice.
        slope = (x * t_centered).sum(axis=-1, keepdims=True) / denom
        intercept = x.mean(axis=-1, keepdims=True) - slope * t.mean()
        return x - (slope * t + intercept)
    raise ValueError(f"unknown detrend mode: {mode}")


def _zscore_epoch(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-epoch, per-channel z-score. x: (N, C, T) or (C, T)."""
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return (x - mean) / (std + eps)


def normalize_epochs(
    epochs: np.ndarray,
    cfg: PreprocessConfig,
    clip_sigma: Optional[float] = None,
) -> np.ndarray:
    """Detrend + z-score + optional sigma-clip.

    epochs: (N, C, T) float array.
    clip_sigma: if not None, clip to +-clip_sigma after z-score (Tier 3 pretrain).
                For eval, pass None and apply amplitude rejection upstream.
    """
    if epochs.ndim != 3:
        raise ValueError(f"expected (N, C, T), got shape {epochs.shape}")

    x = _detrend_epoch(epochs.astype(np.float32), cfg.detrend)
    x = _zscore_epoch(x)
    if clip_sigma is not None:
        x = np.clip(x, -clip_sigma, clip_sigma)
    return x


def amplitude_reject_mask(
    epochs_uv: np.ndarray,
    threshold_uv: float,
) -> np.ndarray:
    """Return boolean mask (N,) of epochs whose |amplitude| stays below threshold.

    Operate on the *unnormalized* epochs in microvolts so the threshold has
    physical meaning. Caller is responsible for converting from MNE's volts.
    """
    if epochs_uv.ndim != 3:
        raise ValueError(f"expected (N, C, T), got shape {epochs_uv.shape}")
    peak = np.abs(epochs_uv).max(axis=(1, 2))
    return peak < threshold_uv