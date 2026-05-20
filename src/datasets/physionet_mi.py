"""PhysioNet Motor Imagery dataset loader.

Wraps MNE's `eegbci.load_data` for download/caching and applies our shared
preprocessing. Returns epoched data, labels, channel positions, and channel
names.

Label scheme (locked, not a flag):
  - 4-class motor imagery: left hand / right hand / both hands / both feet.
  - Built from imagined-movement runs only: 4, 6, 8, 10, 12, 14.
  - Runs 4, 8, 12: T1 = left fist, T2 = right fist.
  - Runs 6, 10, 14: T1 = both fists, T2 = both feet.
  - T0 (rest) is dropped for the MI task.
  - Excluded subjects: 88, 92, 100, 104 (known data quality issues).

Usage:
    cfg = Config()
    loader = PhysioNetMI(subjects=[1, 2], cfg=cfg)
    X, y, ch_pos, ch_names = loader.load()
    # X: (N, M, T) float32, M=64, T=cfg.preprocess.samples_per_epoch
    # y: (N,) int in {0,1,2,3}
    # ch_pos: (M, 3) float32
    # ch_names: list[str] of length M
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import mne
import numpy as np
from mne.datasets import eegbci

from config import Config
from preprocess import (
    amplitude_reject_mask,
    get_channel_positions,
    normalize_epochs,
    preprocess_raw,
)


# Runs by label scheme. T0 is always rest (dropped).
RUNS_LEFT_VS_RIGHT_HAND = (4, 8, 12)   # T1 = left fist, T2 = right fist
RUNS_HANDS_VS_FEET = (6, 10, 14)        # T1 = both fists, T2 = both feet

# 4-class label mapping. Indices chosen for stability across runs.
LABEL_LEFT_HAND = 0
LABEL_RIGHT_HAND = 1
LABEL_BOTH_HANDS = 2
LABEL_BOTH_FEET = 3

# Subjects to exclude per community consensus.
EXCLUDED_SUBJECTS = frozenset({88, 92, 100, 104})

# Trial duration: cue at t=0, imagery from ~0 to 4s. We take the full imagery
# window. Aligns with cfg.preprocess.epoch_length_s = 4.0.
TRIAL_TMIN = 0.0
TRIAL_TMAX = 4.0  # MNE uses inclusive endpoint; samples = (tmax-tmin)*sfreq


@dataclass
class PhysioNetMI:
    """Loader for PhysioNet EEG Motor Movement/Imagery Database (eegmmidb)."""

    subjects: List[int]
    cfg: Config
    mode: str = "eval"  # "eval" applies amplitude rejection; "pretrain" does not
    verbose: bool = False

    def __post_init__(self) -> None:
        bad = [s for s in self.subjects if s in EXCLUDED_SUBJECTS]
        if bad:
            raise ValueError(f"Subjects {bad} are excluded (data quality).")
        if any(s < 1 or s > 109 for s in self.subjects):
            raise ValueError("PhysioNet MI subjects are numbered 1..109.")
        if self.mode not in {"pretrain", "eval"}:
            raise ValueError(f"mode must be 'pretrain' or 'eval', got {self.mode!r}")

    def _load_subject_raws(self, subject: int) -> List[mne.io.BaseRaw]:
        """Download (if needed) and load all MI runs for one subject."""
        all_runs = RUNS_LEFT_VS_RIGHT_HAND + RUNS_HANDS_VS_FEET
        fnames = eegbci.load_data(subject, runs=list(all_runs), verbose="WARNING")
        raws = []
        for fname in fnames:
            raw = mne.io.read_raw_edf(fname, preload=True, verbose="WARNING")
            # PhysioNet channel names have trailing dots (e.g. "Fc5.."); strip
            # and normalize case so the standard montage matches.
            eegbci.standardize(raw)
            raws.append(raw)
        return raws

    def _epoch_run(
        self,
        raw: mne.io.BaseRaw,
        run_number: int,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract epochs and labels from one preprocessed run.

        Returns (epochs_volts, labels) or (None, None) if no events.
        """
        events, event_id = mne.events_from_annotations(
            raw, event_id=dict(T0=0, T1=1, T2=2), verbose="WARNING"
        )
        if len(events) == 0:
            return None, None

        # Map (run, annotation) -> 4-class label. Drop T0 (rest).
        if run_number in RUNS_LEFT_VS_RIGHT_HAND:
            ann_to_label = {1: LABEL_LEFT_HAND, 2: LABEL_RIGHT_HAND}
        elif run_number in RUNS_HANDS_VS_FEET:
            ann_to_label = {1: LABEL_BOTH_HANDS, 2: LABEL_BOTH_FEET}
        else:
            raise ValueError(f"unexpected run {run_number}")

        keep = np.isin(events[:, 2], list(ann_to_label.keys()))
        events = events[keep]
        if len(events) == 0:
            return None, None

        # Epoch. tmax inclusive: we'll trim to exactly samples_per_epoch below.
        epochs_mne = mne.Epochs(
            raw,
            events,
            event_id={k: v for k, v in event_id.items() if v in ann_to_label},
            tmin=TRIAL_TMIN,
            tmax=TRIAL_TMAX,
            baseline=None,  # no baseline correction, per design
            preload=True,
            verbose="WARNING",
        )

        # Data in volts; convert to microvolts for amplitude-based ops.
        data = epochs_mne.get_data()  # (N, C, T)
        target_T = self.cfg.preprocess.samples_per_epoch
        if data.shape[-1] >= target_T:
            data = data[..., :target_T]
        else:
            raise RuntimeError(
                f"run {run_number}: epoch has {data.shape[-1]} samples, "
                f"need {target_T}. Check sfreq after resample."
            )

        labels = np.array(
            [ann_to_label[e] for e in epochs_mne.events[:, 2]],
            dtype=np.int64,
        )
        return data, labels

    def _run_number_from_fname(self, fname) -> int:
        """eegbci filenames look like S001R04.edf -> run 4."""
        fname_str = str(fname)  # Handle Path objects
        stem = fname_str.split("/")[-1]
        return int(stem.split("R")[-1].split(".")[0])

    def load(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Load all configured subjects. See module docstring for return shape."""
        from ._cache import load_or_build
        return load_or_build(
            dataset="physionet_mi",
            mode=self.mode,
            subjects=self.subjects,
            cfg=self.cfg,
            field_names=("X", "y", "ch_pos", "ch_names"),
            build_fn=self._build,
        )

    def _build(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        all_X, all_y = [], []
        ch_pos_ref: Optional[np.ndarray] = None
        ch_names_ref: Optional[List[str]] = None

        for subject in self.subjects:
            fnames = eegbci.load_data(
                subject,
                runs=list(RUNS_LEFT_VS_RIGHT_HAND + RUNS_HANDS_VS_FEET),
                verbose="WARNING",
            )
            for fname in fnames:
                run_number = self._run_number_from_fname(fname)
                raw = mne.io.read_raw_edf(fname, preload=True, verbose="WARNING")
                eegbci.standardize(raw)
                raw = preprocess_raw(raw, self.cfg.preprocess)

                if ch_pos_ref is None:
                    ch_pos_ref = get_channel_positions(raw, self.cfg.preprocess)
                    ch_names_ref = list(raw.ch_names)
                else:
                    # Sanity: all subjects share the same channel set in PhysioNet MI.
                    if list(raw.ch_names) != ch_names_ref:
                        raise RuntimeError(
                            f"Subject {subject} run {run_number}: channel mismatch."
                        )

                data_v, labels = self._epoch_run(raw, run_number)
                if data_v is None:
                    continue

                # Amplitude rejection in microvolts, eval mode only.
                # Skipped when threshold == 0 (disabled). Default is disabled
                # because PhysioNet MI amplitudes regularly exceed 100 µV after
                # preprocessing; a finite threshold discards most epochs.
                # Same stance as EEGPT — rely on z-score normalization instead.
                threshold = self.cfg.artifact.eval.amplitude_reject_uv
                if self.mode == "eval" and threshold > 0:
                    data_uv = data_v * 1e6
                    keep_mask = amplitude_reject_mask(data_uv, threshold)
                    data_v = data_v[keep_mask]
                    labels = labels[keep_mask]
                    if len(data_v) == 0:
                        continue

                # Per-epoch normalization. Clip only for pretrain.
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
                        f"  subject {subject:3d} run {run_number:2d}: "
                        f"{X.shape[0]} epochs"
                    )

        if not all_X:
            raise RuntimeError("No epochs loaded -- check subjects and runs.")

        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)
        assert ch_pos_ref is not None and ch_names_ref is not None
        return X, y, ch_pos_ref, ch_names_ref