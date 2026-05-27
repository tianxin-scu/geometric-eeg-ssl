"""Mixed-corpus iterator for v2.2 cross-montage pretraining.

The three project datasets have very different epoch counts:

    PhysioNet MI : ~9,450 epochs   (64 channels)
    BCIC-2B      : ~6,500 epochs   (3 channels)
    Sleep-EDFx   : ~1,000,000 sub-epochs (2 bipolar derivations)

A naive concat would let Sleep-EDFx dominate ~100x. The v2.2 claim is
"distribution of g_ij matters, count doesn't" -- so we cap each dataset
at the same per-run budget. Small datasets are oversampled with
replacement; large ones are subsampled without replacement.

Each yielded batch is drawn from a *single* dataset (different montages
have different M, so cross-dataset batching would require padding +
masking that we explicitly want to avoid -- the geometry tables are
precomputed per-montage). Batches are interleaved round-robin across
datasets so gradient steps alternate.

A "step" advances the round-robin pointer by one batch. A "pass" through
the corpus is `epochs_per_dataset / batch_size` batches per dataset.

Usage::

    corpus = MixedCorpus(
        datasets={
            "physionet_mi": (X_phys, ch_pos_phys),
            "bcic_2b":      (X_bcic, ch_pos_bcic),
        },
        batch_size=64,
        epochs_per_dataset=8000,
        seed=0,
    )
    for name, batch in corpus:           # batch: (B, M_name, T)
        bias_tabs, val_tabs = geom_tables[name]
        ... pretrain step ...
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Tuple

import numpy as np
import torch


@dataclass
class MixedCorpus:
    """Round-robin batch iterator with per-dataset budget capping.

    Args:
        datasets: dict name -> (X, ch_pos) where X is float tensor
            ``(N, M, T)`` and ch_pos is float tensor ``(M, 3)``. Both
            tensors must be on the same device or convertible later.
            ch_pos is carried for the user's convenience -- the iterator
            itself only yields (name, X_batch).
        batch_size: number of epochs per yielded batch.
        epochs_per_dataset: per-dataset budget per pass. Must be a
            multiple of batch_size.
        seed: RNG seed for reproducible sampling.
        shuffle_each_pass: if True, regenerate the per-dataset sample
            indices at the start of each iteration. Default True.

    Yields:
        (name, X_batch) tuples. X_batch is ``(batch_size, M_name, T)``.
    """

    datasets: Dict[str, Tuple[torch.Tensor, torch.Tensor]]
    batch_size: int
    epochs_per_dataset: int
    seed: int = 0
    shuffle_each_pass: bool = True

    def __post_init__(self):
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive: {self.batch_size}")
        if self.epochs_per_dataset <= 0:
            raise ValueError(
                f"epochs_per_dataset must be positive: {self.epochs_per_dataset}"
            )
        if self.epochs_per_dataset % self.batch_size != 0:
            raise ValueError(
                f"epochs_per_dataset={self.epochs_per_dataset} must be a "
                f"multiple of batch_size={self.batch_size}"
            )
        # Cache shapes for sanity reporting.
        self._shapes = {
            name: tuple(X.shape) for name, (X, _) in self.datasets.items()
        }
        # One epoch length T must agree across datasets (the model assumes
        # a fixed temporal patching).
        Ts = {sh[2] for sh in self._shapes.values()}
        if len(Ts) != 1:
            raise ValueError(
                f"datasets have mismatched T: { {n: sh[2] for n, sh in self._shapes.items()} }"
            )
        self._rng = np.random.default_rng(self.seed)
        self._pass_idx = 0

    @property
    def batches_per_pass(self) -> int:
        """Total batches yielded per call to __iter__."""
        return (self.epochs_per_dataset // self.batch_size) * len(self.datasets)

    def _build_indices(self) -> Dict[str, np.ndarray]:
        """Build per-dataset index arrays of length epochs_per_dataset."""
        idx = {}
        for name, (X, _) in self.datasets.items():
            N = X.shape[0]
            if N >= self.epochs_per_dataset:
                # Subsample without replacement.
                idx[name] = self._rng.choice(
                    N, size=self.epochs_per_dataset, replace=False
                )
            else:
                # Oversample with replacement.
                idx[name] = self._rng.choice(
                    N, size=self.epochs_per_dataset, replace=True
                )
            if self.shuffle_each_pass:
                self._rng.shuffle(idx[name])
        return idx

    def __iter__(self) -> Iterator[Tuple[str, torch.Tensor]]:
        idx_per_ds = self._build_indices()
        names = list(self.datasets.keys())
        n_batches_per_ds = self.epochs_per_dataset // self.batch_size

        for b in range(n_batches_per_ds):
            for name in names:
                X, _ = self.datasets[name]
                start = b * self.batch_size
                end = start + self.batch_size
                sel = idx_per_ds[name][start:end]
                batch = X[sel]
                yield name, batch

        self._pass_idx += 1


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    T = 800
    # PhysioNet-ish: many epochs, 64 channels.
    X_p = torch.randn(150, 64, T)
    cp_p = torch.randn(64, 3)
    # BCIC-ish: fewer epochs, 3 channels.
    X_b = torch.randn(40, 3, T)
    cp_b = torch.randn(3, 3)

    corpus = MixedCorpus(
        datasets={"physionet_mi": (X_p, cp_p), "bcic_2b": (X_b, cp_b)},
        batch_size=8,
        epochs_per_dataset=64,
        seed=42,
    )
    counts = {"physionet_mi": 0, "bcic_2b": 0}
    Ms = {"physionet_mi": 0, "bcic_2b": 0}
    for name, batch in corpus:
        counts[name] += batch.shape[0]
        Ms[name] = batch.shape[1]
    assert counts == {"physionet_mi": 64, "bcic_2b": 64}, counts
    assert Ms == {"physionet_mi": 64, "bcic_2b": 3}, Ms

    # Second pass should re-sample (different indices for the
    # oversampled-with-replacement small dataset).
    first_pass_batch = None
    for name, batch in corpus:
        first_pass_batch = batch
        break

    # Pass-2 first BCIC batch should usually differ (probabilistic);
    # verify only that iteration restarts cleanly.
    total_2nd = sum(1 for _ in corpus)
    assert total_2nd == 16, total_2nd  # 8 batches per ds * 2 datasets
    print(
        f"mixed_corpus smoke OK; per-pass batches={corpus.batches_per_pass}, "
        f"M_phys={Ms['physionet_mi']}, M_bcic={Ms['bcic_2b']}"
    )
