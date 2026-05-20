"""BCI Competition IV-2b dataset loader (3-channel motor imagery).

Three EEG electrodes (C3, Cz, C4), 9 subjects, binary motor imagery
(left vs. right hand). Used for cross-montage zero-shot transfer (E2c):
a 64-channel encoder pretrained on PhysioNet MI is evaluated on this
3-channel layout without retraining.

Access via MOABB:
    from moabb.datasets import BNCI2014_004
    ds = BNCI2014_004()
    sessions = ds.get_data(subjects=[1])
    # nested dict[subject][session][run] -> mne.io.Raw

Usage:
    cfg = Config()
    loader = BCIC2B(subjects=[1, 2], cfg=cfg)
    X, y, ch_pos, ch_names = loader.load()
    # X: (N, 3, T)  with T = cfg.preprocess.samples_per_epoch
    # y: (N,) in {0=left_hand, 1=right_hand}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import mne
import numpy as np

from config import Config
from preprocess import (
    amplitude_reject_mask,
    get_channel_positions,
    normalize_epochs,
    preprocess_raw,
)


# MOABB event ids for BNCI2014_004 are {"left_hand": 1, "right_hand": 2}.
LABEL_LEFT_HAND = 0
LABEL_RIGHT_HAND = 1
EVENT_ID_MAP = {"left_hand": LABEL_LEFT_HAND, "right_hand": LABEL_RIGHT_HAND}

# BCIC-2B trial schedule: cue at t=3 s, motor imagery from 3-7 s post-trial-start.
# Take the imagery window. cfg.preprocess.epoch_length_s = 4.0 aligns.
TRIAL_TMIN = 3.0
TRIAL_TMAX = 7.0

# All 9 subjects are valid; no exclusions per community consensus.
ALL_SUBJECTS = tuple(range(1, 10))

# EEG channel names in BCIC-2B (3 electrodes). The raw files may include
# EOG channels too; we drop those.
EEG_CHANNELS = ("C3", "Cz", "C4")


@dataclass
class BCIC2B:
    """Loader for BCI Competition IV-2b (BNCI2014_004 in MOABB)."""

    subjects: List[int]
    cfg: Config
    mode: str = "eval"  # "eval" applies amplitude rejection; "pretrain" does not
    verbose: bool = False

    def __post_init__(self) -> None:
        bad = [s for s in self.subjects if s not in ALL_SUBJECTS]
        if bad:
            raise ValueError(f"BCIC-2B subjects must be in 1..9; got bad {bad}")
        if self.mode not in {"pretrain", "eval"}:
            raise ValueError(f"mode must be 'pretrain' or 'eval', got {self.mode!r}")

    def _moabb_dataset(self):
        # Lazy import: MOABB pulls in a lot of stuff. Construct once per loader.
        from moabb.datasets import BNCI2014_004
        return BNCI2014_004()

    def _epoch_raw(self, raw: mne.io.BaseRaw) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract (N, 3, T) epochs and (N,) labels from one preprocessed raw."""
        events, event_id = mne.events_from_annotations(raw, verbose="WARNING")
        # MOABB's BNCI2014_004 annotations have keys "left_hand", "right_hand";
        # other annotations (resting, cue) may appear and are ignored.
        keep_codes = {event_id[k] for k in EVENT_ID_MAP if k in event_id}
        if not keep_codes:
            return None, None

        mask = np.isin(events[:, 2], list(keep_codes))
        events = events[mask]
        if len(events) == 0:
            return None, None

        # Build the label vector before slicing — Epochs may reorder.
        code_to_label = {event_id[k]: v for k, v in EVENT_ID_MAP.items() if k in event_id}

        kept_event_id = {k: event_id[k] for k in EVENT_ID_MAP if k in event_id}
        epochs_mne = mne.Epochs(
            raw,
            events,
            event_id=kept_event_id,
            tmin=TRIAL_TMIN,
            tmax=TRIAL_TMAX,
            baseline=None,
            preload=True,
            verbose="WARNING",
        )

        data = epochs_mne.get_data()  # (N, C, T) in volts
        target_T = self.cfg.preprocess.samples_per_epoch
        if data.shape[-1] < target_T:
            raise RuntimeError(
                f"BCIC-2B epoch has {data.shape[-1]} samples, need {target_T}. "
                f"Check resample_hz and TRIAL_TMAX-TRIAL_TMIN."
            )
        data = data[..., :target_T]

        labels = np.array(
            [code_to_label[c] for c in epochs_mne.events[:, 2]],
            dtype=np.int64,
        )
        return data, labels

    def load(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        ds = self._moabb_dataset()
        sessions = ds.get_data(subjects=list(self.subjects))

        all_X, all_y = [], []
        ch_pos_ref: Optional[np.ndarray] = None
        ch_names_ref: Optional[List[str]] = None

        for subject in self.subjects:
            sess_dict = sessions[subject]
            for sess_id, run_dict in sess_dict.items():
                for run_id, raw in run_dict.items():
                    # Keep only the 3 EEG channels; drop EOG and any stim.
                    picks = [ch for ch in EEG_CHANNELS if ch in raw.ch_names]
                    if len(picks) != len(EEG_CHANNELS):
                        raise RuntimeError(
                            f"Subject {subject} session {sess_id} run {run_id}: "
                            f"missing EEG channels. Have {raw.ch_names}; need {EEG_CHANNELS}."
                        )
                    raw = raw.copy().pick_channels(list(picks))
                    raw = preprocess_raw(raw, self.cfg.preprocess)

                    if ch_pos_ref is None:
                        ch_pos_ref = get_channel_positions(raw, self.cfg.preprocess)
                        ch_names_ref = list(raw.ch_names)
                    elif list(raw.ch_names) != ch_names_ref:
                        raise RuntimeError(
                            f"Subject {subject} session {sess_id}: channel mismatch."
                        )

                    data_v, labels = self._epoch_raw(raw)
                    if data_v is None:
                        continue

                    threshold = self.cfg.artifact.eval.amplitude_reject_uv
                    if self.mode == "eval" and threshold > 0:
                        data_uv = data_v * 1e6
                        keep_mask = amplitude_reject_mask(data_uv, threshold)
                        data_v = data_v[keep_mask]
                        labels = labels[keep_mask]
                        if len(data_v) == 0:
                            continue

                    clip = (
                        self.cfg.artifact.pretrain.clip_sigma
                        if self.mode == "pretrain"
                        else None
                    )
                    X = normalize_epochs(data_v, self.cfg.preprocess, clip_sigma=clip)
                    all_X.append(X)
                    all_y.append(labels)

                    if self.verbose:
                        print(
                            f"  subject {subject} session {sess_id} run {run_id}: "
                            f"{X.shape[0]} epochs"
                        )

        if not all_X:
            raise RuntimeError("No BCIC-2B epochs loaded — check subjects.")

        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)
        assert ch_pos_ref is not None and ch_names_ref is not None
        return X, y, ch_pos_ref, ch_names_ref
