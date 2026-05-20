"""Linear probing script for evaluating pretrained EEG SSL representations.

Loads a pretrained online backbone, freezes it, extracts mean-pooled
temporal representations (B, T_p, d_model) -> (B, d_model), then fits a
scikit-learn LogisticRegression probe.

Evaluation protocols:

  - loso:                  leave-one-subject-out (default). Works on any
                           dataset with multiple subjects.
  - within_subject_night:  train probe on night==1, test on night==2,
                           per subject. Sleep-EDFx only (uses the loader's
                           5th return value).

Datasets (dispatch table in `_DATASETS`):

  - physionet_mi:  64-channel, 4-class motor imagery (default)
  - bcic_2b:       3-channel, 2-class motor imagery — used for cross-montage
                   transfer (E2c). Set `--dataset bcic_2b` when the checkpoint
                   was pretrained on PhysioNet MI and being probed on BCIC-2B.
  - sleep_edfx:    2-channel, 5-class sleep staging — used for E2a.

Cross-montage transductive caveat:
  When a 64-channel codex checkpoint is probed on a smaller montage, the
  codex table has no entries for the new layout. Two fallbacks per the
  proposal §5/E2c:
    --codex-fallback random:  re-init the codex for the new M
    --codex-fallback nn:      nearest-neighbor copy from source positions
                              (currently raises NotImplementedError; falls
                              back to random with a warning if requested).

Usage:
    python scripts/probe.py \
        --checkpoint runs/pretrain/g1_pilot/epoch_0099.pt \
        --dataset physionet_mi \
        --eval-protocol loso \
        --output runs/pretrain/g1_pilot/probe_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_src = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_src))

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score
from sklearn.preprocessing import StandardScaler

from src.config import Config
from src.datasets.bcic_2b import ALL_SUBJECTS as BCIC2B_SUBJECTS
from src.datasets.bcic_2b import BCIC2B
from src.datasets.physionet_mi import EXCLUDED_SUBJECTS, PhysioNetMI
from src.datasets.sleep_edfx import ALL_SUBJECTS as SLEEPEDFX_SUBJECTS
from src.datasets.sleep_edfx import SleepEDFx
from src.model.backbone import Backbone
from src.model.geometric_attention import GeometricSpatialEncoder
from src.model.transductive_baseline import TransductiveSpatialEncoder


# ---------------------------------------------------------------------------
# Dataset dispatch
# ---------------------------------------------------------------------------

_DATASETS = {
    "physionet_mi": {
        "loader_cls": PhysioNetMI,
        "all_subjects": lambda: [s for s in range(1, 110) if s not in EXCLUDED_SUBJECTS],
        "has_night": False,
    },
    "bcic_2b": {
        "loader_cls": BCIC2B,
        "all_subjects": lambda: list(BCIC2B_SUBJECTS),
        "has_night": False,
    },
    "sleep_edfx": {
        "loader_cls": SleepEDFx,
        "all_subjects": lambda: list(SLEEPEDFX_SUBJECTS),
        "has_night": True,
    },
}


def _load_one_subject(dataset: str, subject: int, cfg: Config):
    """Returns (X, y, ch_pos, ch_names, night_or_None)."""
    spec = _DATASETS[dataset]
    loader = spec["loader_cls"](subjects=[subject], cfg=cfg, mode="eval", verbose=False)
    out = loader.load()
    if spec["has_night"]:
        X, y, ch_pos, ch_names, night = out
        return X, y, ch_pos, ch_names, night
    X, y, ch_pos, ch_names = out
    return X, y, ch_pos, ch_names, None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Linear probe evaluation.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--dataset", default="physionet_mi", choices=sorted(_DATASETS))
    p.add_argument("--eval-protocol", default="loso",
                   choices=["loso", "within_subject_night"])
    p.add_argument("--subjects", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--max-iter", type=int, default=1000)
    p.add_argument("--codex-fallback", default="random", choices=["random", "nn"],
                   help="When the checkpoint codex size != target n_electrodes, "
                        "how to handle the mismatch. Only relevant for use_codex=True.")
    return p.parse_args(argv)


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
# Backbone construction
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


def _load_backbone(
    ckpt_path: Path,
    cfg: Config,
    n_electrodes: int,
    device: torch.device,
    codex_fallback: str = "random",
) -> Backbone:
    """Load a pretrained backbone; handle codex-size mismatch for cross-montage."""
    state = torch.load(ckpt_path, map_location=device)
    online_state = state["online_backbone"]

    backbone = _build_backbone(cfg, n_electrodes).to(device)
    use_codex = cfg.ablation.use_codex

    # Cross-montage codex handling: source M (from ckpt) may differ from target M.
    if use_codex:
        ckpt_codex_keys = [k for k in online_state if k.endswith(".codex.weight")
                           or k.endswith(".codex.embedding.weight")]
        # Try common naming patterns.
        if not ckpt_codex_keys:
            ckpt_codex_keys = [k for k in online_state if "codex" in k and k.endswith(".weight")]

        if ckpt_codex_keys:
            src_M = None
            for k in ckpt_codex_keys:
                shape = online_state[k].shape
                if shape[0] not in (None,):
                    src_M = shape[0]
                    break
            if src_M is not None and src_M != n_electrodes:
                print(f"Cross-montage codex: ckpt M={src_M}, target M={n_electrodes}; "
                      f"fallback={codex_fallback}")
                if codex_fallback == "nn":
                    warnings.warn(
                        "codex_fallback='nn' is not implemented; falling back to "
                        "random init of target codex (proposal §5 caveat).",
                        RuntimeWarning,
                    )
                # Drop checkpoint codex entries; target codex is fresh random init.
                for k in ckpt_codex_keys:
                    online_state.pop(k, None)

    missing, unexpected = backbone.load_state_dict(online_state, strict=False)
    if unexpected:
        print(f"  warning: unexpected keys ignored: {unexpected[:3]}{'...' if len(unexpected) > 3 else ''}")
    if missing:
        # Only allow missing keys when we intentionally dropped a codex.
        dropped_ok = all("codex" in k for k in missing)
        if not dropped_ok:
            raise RuntimeError(f"Unexpected missing keys: {missing}")
        print(f"  codex re-initialized at target M={n_electrodes}: {len(missing)} keys")

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
    """Mean-pool the temporal output of a frozen backbone."""
    ch_pos_t = torch.from_numpy(ch_pos).to(device)
    bias_tables = val_tables = None
    if backbone.uses_geometry:
        bias_tables, val_tables = backbone.precompute_geom_tables(ch_pos_t)
    N = X.shape[0]
    feats = []
    for i in range(0, N, batch_size):
        x_batch = torch.from_numpy(X[i : i + batch_size]).to(device)
        z = backbone(x_batch, bias_tables=bias_tables, value_tables=val_tables)
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


def _fit_and_score(train_feats, train_y, test_feats, test_y, max_iter):
    scaler = StandardScaler()
    train_feats = scaler.fit_transform(train_feats)
    test_feats = scaler.transform(test_feats)
    clf = LogisticRegression(max_iter=max_iter, C=1.0, solver="lbfgs", n_jobs=-1)
    clf.fit(train_feats, train_y)
    y_pred = clf.predict(test_feats)
    return _compute_metrics(test_y, y_pred)


# ---------------------------------------------------------------------------
# LOSO evaluation (generic over dataset)
# ---------------------------------------------------------------------------

def loso_eval(
    subjects: list[int],
    dataset: str,
    cfg: Config,
    backbone: Backbone,
    device: torch.device,
    max_iter: int,
) -> dict:
    """Leave-one-subject-out linear probe."""
    print(f"LOSO over {len(subjects)} subjects on {dataset} ...")

    # Load all subjects once.
    per_subject: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    ch_pos_ref = None
    for subj in subjects:
        X_s, y_s, ch_pos_s, _, _ = _load_one_subject(dataset, subj, cfg)
        if ch_pos_ref is None:
            ch_pos_ref = ch_pos_s
        per_subject[subj] = (X_s, y_s)
        print(f"  subject {subj}: {X_s.shape[0]} epochs")

    assert ch_pos_ref is not None
    per_subject_results = {}
    bac_list = []

    for test_subj in subjects:
        train_X = np.concatenate(
            [per_subject[s][0] for s in subjects if s != test_subj], axis=0
        )
        train_y = np.concatenate(
            [per_subject[s][1] for s in subjects if s != test_subj], axis=0
        )
        test_X, test_y = per_subject[test_subj]
        train_feats = extract_features(backbone, train_X, ch_pos_ref, device)
        test_feats = extract_features(backbone, test_X, ch_pos_ref, device)
        metrics = _fit_and_score(train_feats, train_y, test_feats, test_y, max_iter)
        per_subject_results[test_subj] = metrics
        bac_list.append(metrics["balanced_accuracy"])
        print(f"  subj {test_subj:3d} | BAC={metrics['balanced_accuracy']:.3f}  "
              f"κ={metrics['cohens_kappa']:.3f}")

    bac_arr = np.array(bac_list)
    aggregate = {
        "bac_mean": float(bac_arr.mean()),
        "bac_std": float(bac_arr.std()),
        "n_subjects": len(subjects),
    }
    print(f"\nLOSO BAC: {aggregate['bac_mean']:.3f} ± {aggregate['bac_std']:.3f}")
    return {"per_subject": per_subject_results, "aggregate": aggregate}


# ---------------------------------------------------------------------------
# Within-subject night split (E2a)
# ---------------------------------------------------------------------------

def within_subject_eval(
    subjects: list[int],
    dataset: str,
    cfg: Config,
    backbone: Backbone,
    device: torch.device,
    max_iter: int,
) -> dict:
    """Per subject, train probe on night==1, test on night==2.

    Only meaningful for datasets where the loader returns a `night` array
    (sleep_edfx). Skips subjects missing either night.
    """
    if not _DATASETS[dataset]["has_night"]:
        raise ValueError(f"dataset {dataset!r} has no night labels; use --eval-protocol loso")

    print(f"Within-subject night split over {len(subjects)} subjects on {dataset} ...")
    per_subject_results = {}
    bac_list = []
    ch_pos_ref = None

    for subj in subjects:
        X_s, y_s, ch_pos_s, _, night_s = _load_one_subject(dataset, subj, cfg)
        if ch_pos_ref is None:
            ch_pos_ref = ch_pos_s
        n1 = night_s == 1
        n2 = night_s == 2
        if not n1.any() or not n2.any():
            print(f"  subj {subj:3d} | SKIP (missing night data: n1={n1.sum()}, n2={n2.sum()})")
            continue
        feats = extract_features(backbone, X_s, ch_pos_ref, device)
        metrics = _fit_and_score(feats[n1], y_s[n1], feats[n2], y_s[n2], max_iter)
        per_subject_results[subj] = metrics
        bac_list.append(metrics["balanced_accuracy"])
        print(f"  subj {subj:3d} | n1={n1.sum()} -> n2={n2.sum()} | "
              f"BAC={metrics['balanced_accuracy']:.3f}")

    if not bac_list:
        raise RuntimeError("No valid subjects with both nights.")

    bac_arr = np.array(bac_list)
    aggregate = {
        "bac_mean": float(bac_arr.mean()),
        "bac_std": float(bac_arr.std()),
        "n_subjects": len(bac_list),
    }
    print(f"\nWithin-subject BAC: {aggregate['bac_mean']:.3f} ± {aggregate['bac_std']:.3f}")
    return {"per_subject": per_subject_results, "aggregate": aggregate}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    args = _parse_args(argv)
    ckpt_path = Path(args.checkpoint)

    if args.config is None:
        args.config = str(ckpt_path.parent / "config.yaml")
    cfg = Config.from_yaml(args.config)

    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"device: {device}")

    spec = _DATASETS[args.dataset]
    subjects = _parse_subjects(args.subjects) if args.subjects else spec["all_subjects"]()

    if args.output is None:
        args.output = str(ckpt_path.parent / f"probe_results_{args.dataset}_{args.eval_protocol}.json")

    # Discover n_electrodes by loading the first subject. Cheap if cached;
    # avoids hardcoding M per dataset.
    print(f"Discovering n_electrodes from {args.dataset} subject {subjects[0]} ...")
    _, _, ch_pos_probe, _, _ = _load_one_subject(args.dataset, subjects[0], cfg)
    n_electrodes = int(ch_pos_probe.shape[0])
    print(f"  n_electrodes = {n_electrodes}")

    backbone = _load_backbone(
        ckpt_path, cfg, n_electrodes=n_electrodes, device=device,
        codex_fallback=args.codex_fallback,
    )
    n_params = sum(p.numel() for p in backbone.parameters())
    print(f"backbone parameters: {n_params:,}  (frozen)")

    if args.eval_protocol == "loso":
        results = loso_eval(subjects, args.dataset, cfg, backbone, device, args.max_iter)
    elif args.eval_protocol == "within_subject_night":
        results = within_subject_eval(subjects, args.dataset, cfg, backbone, device, args.max_iter)
    else:
        raise ValueError(f"unknown eval_protocol {args.eval_protocol!r}")

    results["checkpoint"] = str(ckpt_path)
    results["dataset"] = args.dataset
    results["eval_protocol"] = args.eval_protocol
    results["config"] = cfg.resolved_dict()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"results -> {out_path}")


if __name__ == "__main__":
    main()
