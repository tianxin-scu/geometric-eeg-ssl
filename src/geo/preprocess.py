"""Fixed-scale coordinate normalization.

Normalizing each montage by its own centroid and max-norm makes the same
physical electrode land at different `(x,y,z)` across datasets (e.g. Cz lives
at different coordinates in PhysioNet vs BCIC-2B vs Sleep-EDFx). That is fine
when the geometric MLP consumes only translation-invariant features like
distance + displacement, but feeding absolute `p_i, p_j` into `b` only carries
signal across montages if they share a frame.

This module provides that shared frame. The reference centroid and scale are
computed once from the full standard_1005 layout and then applied to any
subset of electrodes:

    pos_norm = (pos_raw - REF_CENTROID) / REF_SCALE

Sleep-EDFx's bipolar derivations (Fpz-Cz, Pz-Oz) get the midpoint of the
two referenced electrodes, matching the loader's convention for raw coords
in `src/datasets/sleep_edfx.py`.

This is a Tier-1 preprocessing change. It does NOT touch the signal pipeline
-- bandpass, resample, epoching, amplitude reject, normalize_epochs are
unchanged. Only coordinates change, so cached signal data is reusable; `ch_pos`
is recomputed here from `ch_names`.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import mne
import numpy as np


# Reference centroid and scale derived from the full standard_1005 layout
# (343 electrodes). Computed once via the diagnostic in the scoping
# session; pinning them here keeps results reproducible without re-instantiating
# the montage at every call. Units: meters (head frame).
REF_CENTROID = np.array(
    [0.000678092714, -0.016800400987, 0.021227380633], dtype=np.float32
)
REF_SCALE = np.float32(0.127586246)  # ~12.76 cm


# Sleep-EDFx bipolar channel names (short form, as used by the loader)
# and their reference-electrode pairs in standard_1005.
_SLEEP_BIPOLAR_REFS = {
    "FpzCz": ("Fpz", "Cz"),
    "PzOz": ("Pz", "Oz"),
}


def _maybe_get_template_pos(name: str, pos_dict: dict) -> Optional[np.ndarray]:
    """Look up a channel name in standard_1005 with light case handling.

    standard_1005 stores keys like 'FC5', 'CPz', 'Fp1' -- uppercase prefix
    with lowercase 'z' for midline. eegbci's `standardize` leaves PhysioNet
    names as 'Fc5', 'Fcz', so a literal lookup fails. We retry with an
    uppercased prefix (keeping 'z' lowercase).
    """
    if name in pos_dict:
        return pos_dict[name]
    if name.endswith("z"):
        candidate = name[:-1].upper() + "z"
    else:
        candidate = name.upper()
    return pos_dict.get(candidate)


def ch_pos_from_names(
    ch_names: Sequence[str],
    dataset: Optional[str] = None,
) -> np.ndarray:
    """Return fixed-scale `(M, 3)` coordinates from channel names.

    Args:
        ch_names: channel names in the order they appear in the cached
            signal tensor. PhysioNet and BCIC-2B use template names from
            standard_1005 (with case quirks; see `_maybe_get_template_pos`).
            Sleep-EDFx uses the bipolar short names 'FpzCz', 'PzOz'.
        dataset: optional dataset hint. Only needed for sleep_edfx, where
            the bipolar names don't exist in standard_1005 and must be
            resolved to electrode-pair midpoints. For other datasets, pass
            None or the actual name; both work.

    Returns:
        pos: float32 (M, 3) fixed-scale coordinates in the shared frame
            defined by REF_CENTROID and REF_SCALE.

    Raises:
        KeyError if a channel cannot be resolved to either a template
        position or a known bipolar pair.
    """
    montage = mne.channels.make_standard_montage("standard_1005")
    pos_dict = montage.get_positions()["ch_pos"]

    raw = np.empty((len(ch_names), 3), dtype=np.float32)
    for i, name in enumerate(ch_names):
        # Sleep-EDFx bipolar derivations: midpoint of the reference pair.
        if name in _SLEEP_BIPOLAR_REFS:
            a, b = _SLEEP_BIPOLAR_REFS[name]
            raw[i] = (pos_dict[a] + pos_dict[b]) / 2
            continue
        p = _maybe_get_template_pos(name, pos_dict)
        if p is None:
            raise KeyError(
                f"channel {name!r} not in standard_1005 and not a known "
                f"bipolar derivation. dataset hint = {dataset!r}"
            )
        raw[i] = p

    return (raw - REF_CENTROID) / REF_SCALE


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Spot-check against the diagnostic numbers from the scoping session.
    phys_names = [
        "Fc5", "Fc3", "Fc1", "Fcz", "Fc2", "Fc4", "Fc6",
        "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
        "Cp5", "Cp3", "Cp1", "Cpz", "Cp2", "Cp4", "Cp6",
        "Fp1", "Fpz", "Fp2", "Af7", "Af3", "Afz", "Af4", "Af8",
        "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
        "Ft7", "Ft8", "T7", "T8", "T9", "T10", "Tp7", "Tp8",
        "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
        "Po7", "Po3", "Poz", "Po4", "Po8", "O1", "Oz", "O2", "Iz",
    ]
    bcic_names = ["C3", "Cz", "C4"]
    sleep_names = ["FpzCz", "PzOz"]

    for nm, names in [("PhysioNet", phys_names), ("BCIC-2B", bcic_names),
                      ("Sleep-EDFx", sleep_names)]:
        pos = ch_pos_from_names(names, dataset=nm.lower())
        d = np.linalg.norm(pos[:, None] - pos[None, :], axis=-1)
        d_pairs = d[np.triu_indices(len(pos), k=1)]
        if len(d_pairs) > 0:
            print(
                f"{nm:12s}  M={len(pos)}  "
                f"d_min={d_pairs.min():.3f}  d_max={d_pairs.max():.3f}  "
                f"d_mean={d_pairs.mean():.3f}"
            )
        else:
            print(f"{nm:12s}  M={len(pos)}  (no pairs)")
    print("preprocess smoke OK")
