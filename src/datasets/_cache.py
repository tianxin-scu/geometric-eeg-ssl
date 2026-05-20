"""Preprocessing cache for dataset loaders.

Each loader's `load()` does multi-minute work: read EDF/GDF, filter, resample,
epoch, normalize. Identical work happens on every pretrain run and every
eval. This module caches the post-preprocessing arrays to a single .npz per
(dataset, mode, subjects, config) tuple.

Activation: set the `EEG_CACHE_DIR` environment variable. With it unset, the
loaders run uncached (legacy behavior).

Invalidation: the cache key is a SHA-256 hash of every preprocessing knob the
loaders read (`PreprocessConfig`, artifact thresholds, mode, dataset name, and
the sorted subject list). Any change to those produces a different file; old
files remain on disk and can be deleted manually.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, Tuple

import numpy as np


def _cache_root() -> Optional[Path]:
    p = os.environ.get("EEG_CACHE_DIR")
    return Path(p) if p else None


def _fingerprint(dataset: str, mode: str, subjects: Iterable[int], cfg) -> str:
    pp = asdict(cfg.preprocess) if is_dataclass(cfg.preprocess) else dict(cfg.preprocess.__dict__)
    art = {
        "eval_amplitude_reject_uv": cfg.artifact.eval.amplitude_reject_uv,
        "pretrain_clip_sigma": cfg.artifact.pretrain.clip_sigma,
    }
    key = {
        "dataset": dataset,
        "mode": mode,
        "subjects": sorted(int(s) for s in subjects),
        "preprocess": pp,
        "artifact": art,
    }
    s = json.dumps(key, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def load_or_build(
    dataset: str,
    mode: str,
    subjects: Iterable[int],
    cfg,
    field_names: Tuple[str, ...],
    build_fn: Callable[[], Tuple],
) -> Tuple:
    """Return cached arrays if available, else build, save, return.

    `field_names` names the positional return values of `build_fn` in order.
    String/list fields are stored as object arrays and restored to `list`.
    """
    root = _cache_root()
    if root is None:
        return build_fn()

    fp = _fingerprint(dataset, mode, subjects, cfg)
    path = root / f"{dataset}_{mode}_{fp}.npz"

    if path.exists():
        print(f"[cache] loading {dataset}/{mode} from {path.name}", flush=True)
        d = np.load(path, allow_pickle=True)
        out = []
        for name in field_names:
            v = d[name]
            if name == "ch_names":
                v = list(v)
            out.append(v)
        return tuple(out)

    print(f"[cache] building {dataset}/{mode}; will save to {path.name}", flush=True)
    arrays = build_fn()
    root.mkdir(parents=True, exist_ok=True)
    save_kwargs = {}
    for name, arr in zip(field_names, arrays):
        if name == "ch_names":
            save_kwargs[name] = np.array(list(arr), dtype=object)
        else:
            save_kwargs[name] = arr
    np.savez_compressed(path, **save_kwargs)
    print(f"[cache] saved {path.name} ({path.stat().st_size / 1e6:.1f} MB)", flush=True)
    return arrays
