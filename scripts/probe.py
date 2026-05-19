"""Linear probing script for evaluating pretrained EEG SSL representations.

Loads a pretrained online backbone, freezes it, extracts mean-pooled
temporal representations (B, T_p, d_model) → (B, d_model), then fits a
scikit-learn LogisticRegression probe.  Evaluation follows the §6 plan:
leave-one-subject-out (LOSO) on PhysioNet MI by default.

Usage:
    python scripts/probe.py \\
        --checkpoint runs/pretrain/geometric_g1/epoch_0099.pt \\
        --config     runs/pretrain/geometric_g1/config.yaml \\
        --subjects   1-87,89-91,93-99,101-103,105-109 \\
        --output     results/e1_geometric_g1.json

Representation:
    online_backbone(x, ...) → (B, T_p, d_model)
    mean-pool over T_p      → (B, d_model)   [probe input]

Metrics reported (§6):
    Balanced Accuracy (primary), Cohen's κ, Weighted F1.
    Per-subject values + mean ± std.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_src = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_src))

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score
from sklearn.preprocessing import StandardScaler

from src.config import Config
from src.datasets.physionet_mi import EXCLUDED_SUBJECTS, PhysioNetMI
from src.model.backbone import Backbone
from src.model.geometric_attention import GeometricSpatialEncoder
from src.model.transductive_baseline import TransductiveSpatialEncoder


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Linear probe evaluation.")
    p.add_argument("--checkpoint", required=True,
                   help="Path to pretrained checkpoint (.pt).")
    p.add_argument("--config", default=None,
                   help="Path to config YAML. Default: config.yaml in checkpoint's directory.")
    p.add_argument("--subjects", default=None,
                   help="Comma/hyphen subject IDs. Default: all valid PhysioNet MI subjects.")
    p.add_argument("--device", default=None)
    p.add_argument("--output", default=None,
                   help="Path for JSON results. Default: <checkpoint_dir>/probe_results.json.")
    p.add_argument("--max-iter", type=int, default=1000,
                   help="Max iterations for LogisticRegression. Default: 1000.")
    return p.parse_args(argv)


def _all_valid_subjects():
    return [s for s in range(1, 110) if s not in EXCLUDED_SUBJECTS]


def _parse_subjects(spec: str):
    ids = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            ids.extend(range(int(lo), int(hi) + 1))
        else:
            ids.append(int(part))
    return ids


# ---------------------------------------------------------------------------
# Backbone construction (mirrors pretrain.py; no PretrainModel wrapper)
# ---------------------------------------------------------------------------

def _build_backbone(cfg: Config, n_electrodes: int) -> Backbone:
    arch = cfg.arch
    abl = cfg.ablation
    if abl.use_codex:
        spatial = TransductiveSpatialEncoder(
            n_electrodes=n_electrodes,
            d_model=arch.d_model,
            n_heads=arch.n_heads_spatial,
            n_layers=arch.n_layers_spatial,
            dropout=arch.dropout,
        )
    else:
        spatial = GeometricSpatialEncoder(
            d_model=arch.d_model,
            n_heads=arch.n_heads_spatial,
            n_layers=arch.n_layers_spatial,
            geometry_injection=abl.geometry_injection.lower(),
            geom_mlp_hidden=arch.geom_mlp_hidden,
            dropout=arch.dropout,
        )
    return Backbone(spatial, arch=arch)


def _load_backbone(ckpt_path: Path, cfg: Config, n_electrodes: int,
                   device: torch.device) -> Backbone:
    backbone = _build_backbone(cfg, n_electrodes).to(device)
    state = torch.load(ckpt_path, map_location=device)
    backbone.load_state_dict(state["online_backbone"])
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    return backbone


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features(
    backbone: Backbone,
    X: np.ndarray,
    ch_pos: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Extract mean-pooled temporal representations.

    Args:
        X:       (N, M, T) float32.
        ch_pos:  (M, 3) float32.
    Returns:
        features: (N, d_model) float32.
    """
    ch_pos_t = torch.from_numpy(ch_pos).to(device)

    # Geometry tables: precompute once (no backward in probe — safe to reuse).
    bias_tables = val_tables = None
    if backbone.uses_geometry:
        bias_tables, val_tables = backbone.precompute_geom_tables(ch_pos_t)

    N = X.shape[0]
    feats = []
    for i in range(0, N, batch_size):
        x_batch = torch.from_numpy(X[i : i + batch_size]).to(device)
        z = backbone(x_batch, bias_tables=bias_tables, value_tables=val_tables)
        # z: (B, T_p, d_model) → mean-pool → (B, d_model)
        feats.append(z.mean(dim=1).cpu().numpy())

    return np.concatenate(feats, axis=0)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "cohens_kappa":      float(cohen_kappa_score(y_true, y_pred)),
        "weighted_f1":       float(f1_score(y_true, y_pred, average="weighted",
                                            zero_division=0)),
    }


# ---------------------------------------------------------------------------
# LOSO evaluation
# ---------------------------------------------------------------------------

def loso_eval(
    subjects: list[int],
    cfg: Config,
    backbone: Backbone,
    device: torch.device,
    max_iter: int,
) -> dict:
    """Leave-one-subject-out linear probe evaluation on PhysioNet MI.

    For each test subject:
      - Train probe on remaining subjects' features.
      - Evaluate on held-out subject.

    Returns a dict with per-subject metrics and aggregate mean ± std.
    """
    print(f"LOSO over {len(subjects)} subjects …")

    # Load all subjects once; store per-subject arrays.
    per_subject: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    ch_pos_ref = None

    for subj in subjects:
        print(f"  loading subject {subj} …", end=" ", flush=True)
        loader = PhysioNetMI(subjects=[subj], cfg=cfg, mode="eval", verbose=False)
        X_s, y_s, ch_pos_s, _ = loader.load()
        print(f"{X_s.shape[0]} epochs")
        if ch_pos_ref is None:
            ch_pos_ref = ch_pos_s
        per_subject[subj] = (X_s, y_s)

    assert ch_pos_ref is not None

    per_subject_results = {}
    bac_list = []

    for test_subj in subjects:
        # Train split: all other subjects.
        train_X = np.concatenate(
            [per_subject[s][0] for s in subjects if s != test_subj], axis=0
        )
        train_y = np.concatenate(
            [per_subject[s][1] for s in subjects if s != test_subj], axis=0
        )
        test_X, test_y = per_subject[test_subj]

        # Extract features.
        train_feats = extract_features(backbone, train_X, ch_pos_ref, device)
        test_feats  = extract_features(backbone, test_X,  ch_pos_ref, device)

        # Standard-scale features (fit on train only).
        scaler = StandardScaler()
        train_feats = scaler.fit_transform(train_feats)
        test_feats  = scaler.transform(test_feats)

        # Fit linear probe.
        clf = LogisticRegression(
            max_iter=max_iter, C=1.0, solver="lbfgs", n_jobs=-1,
        )
        clf.fit(train_feats, train_y)
        y_pred = clf.predict(test_feats)

        metrics = _compute_metrics(test_y, y_pred)
        per_subject_results[test_subj] = metrics
        bac_list.append(metrics["balanced_accuracy"])
        print(
            f"  subj {test_subj:3d} | "
            f"BAC={metrics['balanced_accuracy']:.3f}  "
            f"κ={metrics['cohens_kappa']:.3f}"
        )

    bac_arr = np.array(bac_list)
    aggregate = {
        "bac_mean": float(bac_arr.mean()),
        "bac_std":  float(bac_arr.std()),
        "n_subjects": len(subjects),
    }
    print(f"\nLOSO BAC: {aggregate['bac_mean']:.3f} ± {aggregate['bac_std']:.3f}")
    return {"per_subject": per_subject_results, "aggregate": aggregate}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    args = _parse_args(argv)
    ckpt_path = Path(args.checkpoint)

    # Config: default to config.yaml sitting next to the checkpoint.
    if args.config is None:
        args.config = str(ckpt_path.parent / "config.yaml")
    cfg = Config.from_yaml(args.config)

    # Device.
    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"device: {device}")

    # Subject list.
    subjects = _parse_subjects(args.subjects) if args.subjects else _all_valid_subjects()

    # Output path.
    if args.output is None:
        args.output = str(ckpt_path.parent / "probe_results.json")

    # Build + load backbone.
    # We don't know M until we load one subject; use PhysioNet MI's fixed M=64.
    PHYSIONET_M = 64
    backbone = _load_backbone(ckpt_path, cfg, n_electrodes=PHYSIONET_M, device=device)
    n_params = sum(p.numel() for p in backbone.parameters())
    print(f"backbone parameters: {n_params:,}  (frozen)")

    # Evaluate.
    results = loso_eval(subjects, cfg, backbone, device, max_iter=args.max_iter)
    results["checkpoint"] = str(ckpt_path)
    results["config"] = cfg.resolved_dict()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"results → {out_path}")


if __name__ == "__main__":
    main()
