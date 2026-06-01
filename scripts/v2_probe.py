"""Linear probe evaluation for v2 cross-montage checkpoints.

Mirrors v1's `scripts/probe.py` but loads the v2 stack (`BackboneV2` +
`GeometricSpatialEncoderV2`) and recomputes channel positions with the v2
fixed-scale normalizer (`src/v2/preprocess.ch_pos_from_names`). v1 modules
are not touched -- this keeps the v2-isolation discipline of `src/v2/`.

Self-configuring: given a checkpoint dir containing `v2_run.txt`, the
variant (geometric/chind) and held-out dataset are read from that sidecar,
so the headline runs need no extra flags:

    python scripts/v2_probe.py --checkpoint runs/pretrain/v2_geometric_no_bcic/epoch_0099.pt

The held-out dataset is the one to probe (zero-shot): the encoder never
saw its montage during pretraining. Per `docs/v2/experiment_protocol.md`:

  - physionet_mi : LOSO
  - bcic_2b      : LOSO
  - sleep_edfx   : within-subject night split (train night 1, test night 2)

Feature pooling: the v2 backbone returns per-electrode per-time-step
tokens (B, M, T_p, d). The probe mean-pools over BOTH electrodes and time
to a fixed (B, d) summary -- montage-size-invariant, which is what lets a
probe train on a held-out montage the encoder never saw.

NOTE on eval scale: by default this probes ALL subjects of the held-out
dataset (stable BAC). The local 9-subject caches built for pretraining are
NOT enough for phys/sleep eval -- the loader will download/preprocess the
full population on first run. Use --subjects to probe a subset for a quick
pipeline check.

Output JSON schema matches v1's probe so v2 results aggregation can reuse it.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score
from sklearn.preprocessing import StandardScaler

from src.config import Config
from src.datasets.bcic_2b import ALL_SUBJECTS as BCIC2B_SUBJECTS
from src.datasets.bcic_2b import BCIC2B
from src.datasets.physionet_mi import EXCLUDED_SUBJECTS, PhysioNetMI
from src.datasets.sleep_edfx import ALL_SUBJECTS as SLEEPEDFX_SUBJECTS
from src.datasets.sleep_edfx import SleepEDFx
from src.v2.model.backbone_v2 import BackboneV2
from src.v2.model.geometric_attention_v2 import GeometricSpatialEncoderV2
from src.v2.preprocess import ch_pos_from_names


# ---------------------------------------------------------------------------
# Dataset dispatch
# ---------------------------------------------------------------------------

_DATASETS = {
    "physionet_mi": {
        "loader_cls": PhysioNetMI,
        "all_subjects": lambda: [s for s in range(1, 110) if s not in EXCLUDED_SUBJECTS],
        "has_night": False,
        "default_protocol": "loso",
    },
    "bcic_2b": {
        "loader_cls": BCIC2B,
        "all_subjects": lambda: list(BCIC2B_SUBJECTS),
        "has_night": False,
        "default_protocol": "loso",
    },
    "sleep_edfx": {
        "loader_cls": SleepEDFx,
        "all_subjects": lambda: list(SLEEPEDFX_SUBJECTS),
        "has_night": True,
        "default_protocol": "within_subject_night",
    },
}


def _load_one_subject(dataset: str, subject: int, cfg: Config):
    """Load one subject; recompute ch_pos with the v2 fixed-scale normalizer.

    Returns (X, y, ch_pos_v2, night_or_None). The loader's own ch_pos is
    discarded -- v2 must use the shared-frame coordinates derived from
    ch_names, not the per-montage coords the v1 cache stored.
    """
    spec = _DATASETS[dataset]
    loader = spec["loader_cls"](subjects=[subject], cfg=cfg, mode="eval", verbose=False)
    out = loader.load()
    if spec["has_night"]:
        X, y, _cp_v1, ch_names, night = out
    else:
        X, y, _cp_v1, ch_names = out
        night = None
    ch_pos = ch_pos_from_names(ch_names, dataset=dataset)
    return X, y, ch_pos, night


# ---------------------------------------------------------------------------
# Run-spec sidecar
# ---------------------------------------------------------------------------

def _read_run_spec(ckpt_dir: Path) -> dict:
    """Parse v2_run.txt (variant, held_out, ...) if present."""
    spec_path = ckpt_dir / "v2_run.txt"
    spec: dict[str, str] = {}
    if spec_path.exists():
        for line in spec_path.read_text().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                spec[k.strip()] = v.strip()
    return spec


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="v2 linear probe evaluation.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default=None,
                   help="Default: <ckpt dir>/config.yaml.")
    p.add_argument("--dataset", default=None, choices=sorted(_DATASETS),
                   help="Held-out dataset to probe. Default: read 'held_out' "
                        "from <ckpt dir>/v2_run.txt.")
    p.add_argument("--variant", default=None, choices=["geometric", "chind"],
                   help="Spatial-encoder variant. Default: read 'variant' "
                        "from v2_run.txt.")
    p.add_argument("--eval-protocol", default=None,
                   choices=["loso", "within_subject_night"],
                   help="Default: per-dataset (loso for phys/bcic, "
                        "within_subject_night for sleep).")
    p.add_argument("--subjects", default=None,
                   help="Subset like '1,2,5-9'. Default: all subjects of the "
                        "held-out dataset.")
    p.add_argument("--device", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--max-iter", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=64)
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

def _geom_mode(cfg: Config, variant: str) -> str:
    if variant == "geometric":
        return cfg.ablation.geometry_injection.lower()
    if variant == "chind":
        return "none"
    raise ValueError(f"unknown variant: {variant!r}")


def _build_backbone(cfg: Config, variant: str) -> BackboneV2:
    arch = cfg.arch
    spatial = GeometricSpatialEncoderV2(
        d_model=arch.d_model,
        n_heads=arch.n_heads_spatial,
        n_layers=arch.n_layers_spatial,
        geometry_injection=_geom_mode(cfg, variant),
        geom_mlp_hidden=arch.geom_mlp_hidden,
        dropout=arch.dropout,
        descriptor=cfg.ablation.geom_descriptor,
    )
    return BackboneV2(spatial, arch=arch)


def _load_backbone(ckpt_path: Path, cfg: Config, variant: str,
                   device: torch.device) -> BackboneV2:
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    backbone = _build_backbone(cfg, variant).to(device)
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
    backbone: BackboneV2,
    X: np.ndarray,
    ch_pos: np.ndarray,
    device: torch.device,
    uses_geometry: bool,
    batch_size: int = 64,
) -> np.ndarray:
    """Frozen backbone -> (B, M, T_p, d) -> mean over (M, T_p) -> (B, d)."""
    bias_tables = val_tables = None
    if uses_geometry:
        ch_pos_t = torch.from_numpy(ch_pos).to(device)
        bias_tables, val_tables = backbone.precompute_geom_tables(ch_pos_t)
    N = X.shape[0]
    feats = []
    for i in range(0, N, batch_size):
        x_batch = torch.from_numpy(X[i : i + batch_size]).to(device)
        z = backbone(x_batch, bias_tables=bias_tables, value_tables=val_tables)
        feats.append(z.mean(dim=(1, 2)).cpu().numpy())  # (b, M, T_p, d) -> (b, d)
    return np.concatenate(feats, axis=0)


# ---------------------------------------------------------------------------
# Metrics + probe fit
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
    clf = LogisticRegression(max_iter=max_iter, C=1.0, solver="lbfgs")
    clf.fit(train_feats, train_y)
    y_pred = clf.predict(test_feats)
    return _compute_metrics(test_y, y_pred)


# ---------------------------------------------------------------------------
# LOSO evaluation
# ---------------------------------------------------------------------------

def loso_eval(subjects, dataset, cfg, backbone, device, uses_geometry,
              max_iter, batch_size) -> dict:
    print(f"LOSO over {len(subjects)} subjects on {dataset} ...")

    per_subject_feats: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for subj in subjects:
        X_s, y_s, ch_pos_s, _ = _load_one_subject(dataset, subj, cfg)
        feats = extract_features(backbone, X_s, ch_pos_s, device,
                                 uses_geometry, batch_size)
        per_subject_feats[subj] = (feats, y_s)
        print(f"  subject {subj}: {X_s.shape[0]} epochs")

    per_subject_results = {}
    bac_list = []
    for test_subj in subjects:
        train_feats = np.concatenate(
            [per_subject_feats[s][0] for s in subjects if s != test_subj], axis=0)
        train_y = np.concatenate(
            [per_subject_feats[s][1] for s in subjects if s != test_subj], axis=0)
        test_feats, test_y = per_subject_feats[test_subj]
        metrics = _fit_and_score(train_feats, train_y, test_feats, test_y, max_iter)
        per_subject_results[test_subj] = metrics
        bac_list.append(metrics["balanced_accuracy"])
        print(f"  subj {test_subj:3d} | BAC={metrics['balanced_accuracy']:.3f}  "
              f"kappa={metrics['cohens_kappa']:.3f}")

    bac_arr = np.array(bac_list)
    aggregate = {
        "bac_mean": float(bac_arr.mean()),
        "bac_std": float(bac_arr.std()),
        "n_subjects": len(subjects),
    }
    print(f"\nLOSO BAC: {aggregate['bac_mean']:.3f} +/- {aggregate['bac_std']:.3f}")
    return {"per_subject": per_subject_results, "aggregate": aggregate}


# ---------------------------------------------------------------------------
# Within-subject night split (sleep_edfx)
# ---------------------------------------------------------------------------

def within_subject_eval(subjects, dataset, cfg, backbone, device,
                        uses_geometry, max_iter, batch_size) -> dict:
    if not _DATASETS[dataset]["has_night"]:
        raise ValueError(f"dataset {dataset!r} has no night labels; use loso")

    print(f"Within-subject night split over {len(subjects)} subjects on {dataset} ...")
    per_subject_results = {}
    bac_list = []
    for subj in subjects:
        X_s, y_s, ch_pos_s, night_s = _load_one_subject(dataset, subj, cfg)
        n1 = night_s == 1
        n2 = night_s == 2
        if not n1.any() or not n2.any():
            print(f"  subj {subj:3d} | SKIP (n1={n1.sum()}, n2={n2.sum()})")
            continue
        feats = extract_features(backbone, X_s, ch_pos_s, device,
                                 uses_geometry, batch_size)
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
    print(f"\nWithin-subject BAC: {aggregate['bac_mean']:.3f} +/- {aggregate['bac_std']:.3f}")
    return {"per_subject": per_subject_results, "aggregate": aggregate}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    args = _parse_args(argv)
    ckpt_path = Path(args.checkpoint)
    ckpt_dir = ckpt_path.parent
    run_spec = _read_run_spec(ckpt_dir)

    # Resolve variant + dataset from v2_run.txt unless overridden.
    variant = args.variant or run_spec.get("variant")
    if variant is None:
        raise ValueError(
            "variant not given and not found in v2_run.txt; pass --variant.")
    dataset = args.dataset or run_spec.get("held_out")
    if dataset is None:
        raise ValueError(
            "dataset not given and 'held_out' not in v2_run.txt; pass --dataset.")
    if dataset not in _DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}")

    if args.config is None:
        args.config = str(ckpt_dir / "config.yaml")
    cfg = Config.from_yaml(args.config)

    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    protocol = args.eval_protocol or _DATASETS[dataset]["default_protocol"]
    subjects = (_parse_subjects(args.subjects) if args.subjects
                else _DATASETS[dataset]["all_subjects"]())
    uses_geometry = _geom_mode(cfg, variant) != "none"

    print(f"checkpoint : {ckpt_path}")
    print(f"variant    : {variant}  (geometry={'on' if uses_geometry else 'off'})")
    print(f"held-out   : {dataset}  (zero-shot probe)")
    print(f"protocol   : {protocol}")
    print(f"subjects   : {len(subjects)}")
    print(f"device     : {device}")

    if args.output is None:
        args.output = str(ckpt_dir / f"probe_{dataset}_{protocol}.json")

    backbone = _load_backbone(ckpt_path, cfg, variant, device)
    n_params = sum(p.numel() for p in backbone.parameters())
    print(f"backbone parameters: {n_params:,}  (frozen)\n")

    if protocol == "loso":
        results = loso_eval(subjects, dataset, cfg, backbone, device,
                            uses_geometry, args.max_iter, args.batch_size)
    elif protocol == "within_subject_night":
        results = within_subject_eval(subjects, dataset, cfg, backbone, device,
                                      uses_geometry, args.max_iter, args.batch_size)
    else:
        raise ValueError(f"unknown protocol {protocol!r}")

    descriptor = cfg.ablation.geom_descriptor
    # variant_label distinguishes descriptor ablations of the same variant
    # (e.g. geometric[full] vs geometric[pos_only]) for results aggregation.
    variant_label = variant if descriptor == "full" else f"{variant}_{descriptor}"
    results["checkpoint"] = str(ckpt_path)
    results["dataset"] = dataset
    results["variant"] = variant
    results["descriptor"] = descriptor
    results["variant_label"] = variant_label
    results["held_out"] = dataset
    results["eval_protocol"] = protocol
    results["config"] = cfg.resolved_dict()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"results -> {out_path}")


if __name__ == "__main__":
    main()
