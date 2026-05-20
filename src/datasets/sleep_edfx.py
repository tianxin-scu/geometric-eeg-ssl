"""Sleep-EDFx dataset loader (5-class sleep staging, multi-night).

PhysioNet Sleep-EDFx (SC subset): two-channel EEG sleep recordings on ~78
healthy subjects, mostly two nights per subject. Used for the cross-session
within-subject test (E2a): train a linear probe on night 1, evaluate on
night 2 of the same subject, with the encoder frozen.

Two practical wrinkles unique to this dataset:

  1) Bipolar derivations. The two EEG signals are "EEG Fpz-Cz" and
     "EEG Pz-Oz". They are not single-electrode recordings, so their 3-D
     position is taken as the midpoint of the two reference electrodes from
     the standard_1005 montage. This is the standard physical interpretation
     of a bipolar derivation -- the signal samples activity from the line
     segment between the two electrodes.

  2) 30 s -> 7 x 4 s slicing. Sleep stages are annotated per 30 s window,
     but the backbone consumes 4 s epochs. Each 30 s window is sliced into
     seven non-overlapping 4 s sub-epochs, each inheriting the parent
     window's sleep stage. The trailing 2 s of each window are discarded.

Label scheme (5-class, AASM):
  W=0, N1=1, N2=2, N3=3 (merging Sleep stage 3 + Sleep stage 4), REM=4.

Access via MNE native:
    mne.datasets.sleep_physionet.age.fetch_data(subjects=[s], recording=[1, 2])

Usage:
    cfg = Config()
    loader = SleepEDFx(subjects=[0, 1], cfg=cfg)
    X, y, ch_pos, ch_names, night = loader.load()
    # X: (N, 2, T)  with T = cfg.preprocess.samples_per_epoch
    # y: (N,) in {0..4}
    # night: (N,) int64 in {1, 2}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import mne
import numpy as np

from config import Config, PreprocessConfig
from preprocess import amplitude_reject_mask, normalize_epochs


# All Sleep-EDFx age-study subjects.
_UNAVAILABLE = {39, 68, 69, 78, 79}
ALL_SUBJECTS = tuple(s for s in range(0, 83) if s not in _UNAVAILABLE)

# Subject-night pairs that are missing per MNE's documentation.
_MISSING_RECORDINGS = {(36, 1), (52, 1), (13, 2)}

# Hypnogram annotation -> AASM 5-class label. N3 + N4 merged.
ANNOT_TO_LABEL = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
}
# Annotations like "Sleep stage ?" or "Movement time" are dropped.

# The two bipolar EEG derivations we use. These are the raw EDF channel names.
EEG_BIPOLAR_RAW = ("EEG Fpz-Cz", "EEG Pz-Oz")
# Short names attached to the loaded raw (no spaces; unambiguous).
EEG_BIPOLAR_SHORT = ("FpzCz", "PzOz")
# The reference pairs that define each bipolar derivation -- used to compute
# the midpoint position from standard_1005.
_BIPOLAR_REFS = {
    "FpzCz": ("Fpz", "Cz"),
    "PzOz": ("Pz", "Oz"),
}

# Annotated sleep-stage windows are 30 s. We slice into non-overlapping 4 s
# sub-epochs that inherit the parent label.
ANNOT_WINDOW_S = 30.0


@dataclass
class SleepEDFx:
    """Loader for the PhysioNet Sleep-EDFx Sleep-Cassette dataset."""

    subjects: List[int]
    cfg: Config
    mode: str = "eval"
    verbose: bool = False
    recordings: Tuple[int, ...] = (1, 2)  # nights to load; subset for testing

    def __post_init__(self) -> None:
        bad = [s for s in self.subjects if s not in ALL_SUBJECTS]
        if bad:
            raise ValueError(f"Sleep-EDFx subjects must be in 0..82 minus {_UNAVAILABLE}; bad {bad}")
        if self.mode not in {"pretrain", "eval"}:
            raise ValueError(f"mode must be 'pretrain' or 'eval', got {self.mode!r}")
        if not set(self.recordings).issubset({1, 2}):
            raise ValueError(f"recordings must be subset of (1, 2); got {self.recordings}")

    def _compute_bipolar_positions(self) -> np.ndarray:
        """Midpoint positions of the bipolar references, in meters (head frame).

        Normalized identically to get_channel_positions: centroid-subtract +
        unit max-norm scaling when cfg.preprocess.coord_normalize is True.
        """
        montage = mne.channels.make_standard_montage(self.cfg.preprocess.montage)
        ref_pos = montage.get_positions()["ch_pos"]
        rows: List[np.ndarray] = []
        for short in EEG_BIPOLAR_SHORT:
            a, b = _BIPOLAR_REFS[short]
            rows.append(0.5 * (np.asarray(ref_pos[a]) + np.asarray(ref_pos[b])))
        pos = np.stack(rows, axis=0).astype(np.float32)

        if self.cfg.preprocess.coord_normalize:
            pos = pos - pos.mean(axis=0, keepdims=True)
            scale = float(np.linalg.norm(pos, axis=1).max())
            if scale > 0:
                pos = pos / scale
        return pos

    def _filter_resample(
        self, raw: mne.io.BaseRaw, cfg: PreprocessConfig
    ) -> mne.io.BaseRaw:
        """Bandpass + resample only (no montage; bipolar channels are not 10-05)."""
        raw.load_data()
        raw.filter(
            l_freq=cfg.bandpass_hz[0],
            h_freq=cfg.bandpass_hz[1],
            method="fir",
            phase="zero",
            fir_design="firwin",
            verbose="WARNING",
        )
        if raw.info["sfreq"] != cfg.resample_hz:
            raw.resample(cfg.resample_hz, verbose="WARNING")
        return raw

    def _load_recording(
        self, psg_path: str, hyp_path: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Load one (PSG, hypnogram) recording -> (N, 2, T) epochs + labels."""
        raw = mne.io.read_raw_edf(psg_path, preload=False, verbose="WARNING")
        present = [ch for ch in EEG_BIPOLAR_RAW if ch in raw.ch_names]
        if len(present) != len(EEG_BIPOLAR_RAW):
            return None
        raw = raw.copy().pick_channels(list(present))
        rename_map = dict(zip(EEG_BIPOLAR_RAW, EEG_BIPOLAR_SHORT))
        raw.rename_channels(rename_map)
        raw = self._filter_resample(raw, self.cfg.preprocess)

        # Attach hypnogram annotations and slice into 4 s sub-epochs.
        ann = mne.read_annotations(hyp_path)
        raw.set_annotations(ann, emit_warning=False, verbose="WARNING")

        sfreq = raw.info["sfreq"]
        samples_per_subepoch = self.cfg.preprocess.samples_per_epoch  # e.g. 800
        subepoch_s = self.cfg.preprocess.epoch_length_s
        n_sub_per_window = int(ANNOT_WINDOW_S // subepoch_s)  # 7 at 4 s
        if n_sub_per_window < 1:
            raise RuntimeError(
                f"epoch_length_s={subepoch_s} too long to fit into {ANNOT_WINDOW_S}s window."
            )

        data_all = raw.get_data()  # (2, T_total)
        n_total = data_all.shape[1]

        epochs_chunks: List[np.ndarray] = []
        labels_chunks: List[int] = []
        for descr, onset_s, dur_s in zip(ann.description, ann.onset, ann.duration):
            if descr not in ANNOT_TO_LABEL:
                continue
            label = ANNOT_TO_LABEL[descr]
            # Each annotation spans `dur_s` seconds. Sleep-EDFx annotates in
            # blocks longer than 30 s sometimes (one annotation per stage run).
            # Slice the entire annotated span into 4 s sub-epochs.
            n_windows = int(np.floor(dur_s / subepoch_s))
            for i in range(n_windows):
                start = int(round((onset_s + i * subepoch_s) * sfreq))
                end = start + samples_per_subepoch
                if end > n_total:
                    break
                sub = data_all[:, start:end]
                if sub.shape[1] != samples_per_subepoch:
                    break
                epochs_chunks.append(sub)
                labels_chunks.append(label)

        if not epochs_chunks:
            return None
        X = np.stack(epochs_chunks, axis=0)  # (N, 2, T) volts
        y = np.asarray(labels_chunks, dtype=np.int64)
        return X, y

    def load(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray]:
        """Returns (X, y, ch_pos, ch_names, night)."""
        files = mne.datasets.sleep_physionet.age.fetch_data(
            subjects=list(self.subjects),
            recording=list(self.recordings),
            verbose="WARNING",
        )
        # `files` is a list of (psg_path, hyp_path) tuples, in the order
        # of (subject, recording) iteration that MNE uses internally.
        # We need to know which night each pair belongs to. Reconstruct by
        # parsing the filename: SCNN[N][N1|N2]xx-PSG.edf where the 4th char
        # before -PSG indicates night (1 or 2).
        per_recording: List[Tuple[int, int, str, str]] = []
        for psg_path, hyp_path in files:
            stem = str(psg_path).split("/")[-1]
            # Sleep cassette filename: SC4ssNE0-PSG.edf, ss=subj, N=night.
            # Robust parse: find "SC4" then characters 5-6 are subj, char 7 is night.
            try:
                idx = stem.index("SC4")
                subj = int(stem[idx + 3 : idx + 5])
                night = int(stem[idx + 5])
            except (ValueError, IndexError) as e:
                raise RuntimeError(f"Cannot parse subj/night from {stem}: {e}")
            per_recording.append((subj, night, psg_path, hyp_path))

        all_X, all_y, all_night = [], [], []
        ch_names_ref = list(EEG_BIPOLAR_SHORT)
        ch_pos = self._compute_bipolar_positions()

        for subj, night, psg, hyp in per_recording:
            if (subj, night) in _MISSING_RECORDINGS:
                continue
            result = self._load_recording(psg, hyp)
            if result is None:
                continue
            data_v, labels = result

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
            all_night.append(np.full(len(labels), night, dtype=np.int64))

            if self.verbose:
                print(f"  subject {subj:2d} night {night}: {X.shape[0]} epochs")

        if not all_X:
            raise RuntimeError("No Sleep-EDFx epochs loaded -- check subjects/recordings.")

        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)
        night = np.concatenate(all_night, axis=0)
        return X, y, ch_pos, ch_names_ref, night
