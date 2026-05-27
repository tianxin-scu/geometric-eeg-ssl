"""v2 cross-montage pretraining (leave-one-dataset-out).

Mixed-corpus pretraining for the v2 backbone. Specify two of three datasets
on the CLI; the third is held out for downstream zero-shot eval.

Usage (no-sleep leave-one-out):
    python scripts/v2_pretrain.py \\
        --config configs/v2/pretrain/v2_default.yaml \\
        --pretrain-datasets physionet_mi,bcic_2b \\
        --epochs-per-dataset 8000 \\
        --ckpt-dir runs/pretrain/v2_no_sleep

Usage (resume):
    python scripts/v2_pretrain.py ... --resume latest

Differences vs v1 `scripts/pretrain.py`:
- Loads two datasets and mixes them via MixedCorpus.
- Recomputes geometry tables PER DATASET PER STEP (different montage).
- Uses BackboneV2 + PretrainModelV2 (per-electrode tokens, per-channel
  alignment + per-electrode recon).
- Uses src/v2/preprocess.ch_pos_from_names for fixed-scale coordinates.

The v1 signal cache is reused (preprocessing knobs unchanged). Set
`EEG_CACHE_DIR` to skip re-preprocessing.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(_repo_root / "src"))   # loaders use bare `from config import`

import torch
import torch.nn as nn

from src.config import Config
from src.datasets.bcic_2b import ALL_SUBJECTS as BCIC_ALL
from src.datasets.bcic_2b import BCIC2B
from src.datasets.physionet_mi import EXCLUDED_SUBJECTS as PHYS_EXCLUDED
from src.datasets.physionet_mi import PhysioNetMI
from src.datasets.sleep_edfx import ALL_SUBJECTS as SLEEP_ALL
from src.datasets.sleep_edfx import SleepEDFx
from src.v2.datasets.mixed_corpus import MixedCorpus
from src.v2.losses import PretrainModelV2
from src.v2.model.backbone_v2 import BackboneV2
from src.v2.model.geometric_attention_v2 import GeometricSpatialEncoderV2
from src.v2.preprocess import ch_pos_from_names


VALID_DATASETS = {"physionet_mi", "bcic_2b", "sleep_edfx"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="v2 cross-montage pretraining.")
    p.add_argument("--config", required=True)
    p.add_argument(
        "--pretrain-datasets", required=True,
        help="Comma-separated names from {physionet_mi, bcic_2b, sleep_edfx}. "
             "The remaining one is held out for downstream eval.",
    )
    p.add_argument(
        "--epochs-per-dataset", type=int, required=True,
        help="Per-dataset budget per pass; small datasets are oversampled, "
             "large ones are subsampled to hit this exact count.",
    )
    p.add_argument(
        "--n-subjects-per-dataset", type=int, default=None,
        help="Cap subjects per dataset (subject-level balancing; Option A "
             "in docs/v2/experiment_protocol.md). Default: use all subjects.",
    )
    p.add_argument(
        "--variant", choices=("geometric", "chind"), default="geometric",
        help="Spatial encoder variant. 'geometric' = g3 + 10-D descriptor "
             "(the headline). 'chind' = geometry_injection=none "
             "(channel-independent baseline; no spatial structure). "
             "'codex' variant is deferred -- see docs/v2/experiment_protocol.md.",
    )
    p.add_argument(
        "--ckpt-dir", default=None,
        help="Default: runs/pretrain/v2_<timestamp>.",
    )
    p.add_argument("--device", default=None,
                   help="Torch device ('cuda', 'mps', 'cpu'). Auto-detected by default.")
    p.add_argument("--resume", default=None,
                   help="'latest' or 'epoch_NNNN.pt' inside --ckpt-dir.")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _load_dataset(
    name: str, cfg: Config, n_subjects: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """Run v1 loader (cached signals), recompute ch_pos with v2 normalizer.

    Returns (X, ch_pos_v2, ch_names) -- ch_pos is the fixed-scale variant
    derived from ch_names, NOT what the v1 loader put in its cache.

    If n_subjects is given, the first n subjects of each dataset's natural
    list are used (Option A subject-level balancing). BCIC-2B's hard
    ceiling is 9, so n_subjects > 9 still gets only 9 for BCIC.
    """
    if name == "physionet_mi":
        subjects = [s for s in range(1, 110) if s not in PHYS_EXCLUDED]
        if n_subjects is not None:
            subjects = subjects[:n_subjects]
        X, _y, _cp_v1, ch_names = PhysioNetMI(
            subjects=subjects, cfg=cfg, mode="pretrain", verbose=True
        ).load()
    elif name == "bcic_2b":
        subjects = list(BCIC_ALL)
        if n_subjects is not None:
            subjects = subjects[:n_subjects]
        X, _y, _cp_v1, ch_names = BCIC2B(
            subjects=subjects, cfg=cfg, mode="pretrain", verbose=True
        ).load()
    elif name == "sleep_edfx":
        subjects = list(SLEEP_ALL)
        if n_subjects is not None:
            subjects = subjects[:n_subjects]
        X, _y, _cp_v1, ch_names, _night = SleepEDFx(
            subjects=subjects, cfg=cfg, mode="pretrain", verbose=True
        ).load()
    else:
        raise ValueError(f"unknown dataset: {name}")

    pos = ch_pos_from_names(ch_names, dataset=name)
    return torch.from_numpy(X), torch.from_numpy(pos), list(ch_names)


# ---------------------------------------------------------------------------
# Model + scheduler
# ---------------------------------------------------------------------------

def _build_model(cfg: Config, variant: str) -> PretrainModelV2:
    """Build the v2 PretrainModel for one of the variants.

    variant = 'geometric': uses cfg.ablation.geometry_injection (G3 default).
    variant = 'chind':     forces geometry_injection='none' (channel-independent).
    """
    abl = cfg.ablation
    arch = cfg.arch
    if abl.use_codex:
        raise ValueError(
            "v2 has no codex baseline yet (deferred); use_codex must be False."
        )
    if variant == "geometric":
        geom = abl.geometry_injection.lower()
    elif variant == "chind":
        geom = "none"
    else:
        raise ValueError(f"unknown variant: {variant!r}")
    spatial = GeometricSpatialEncoderV2(
        d_model=arch.d_model,
        n_heads=arch.n_heads_spatial,
        n_layers=arch.n_layers_spatial,
        geometry_injection=geom,
        geom_mlp_hidden=arch.geom_mlp_hidden,
        dropout=arch.dropout,
    )
    backbone = BackboneV2(spatial, arch=arch)
    return PretrainModelV2(backbone, lambda_R=cfg.train.lambda_R)


def _build_lr_scheduler(optimizer, cfg: Config):
    warmup = cfg.train.lr_warmup_epochs
    total = cfg.train.n_epochs
    if warmup > 0:
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-4, end_factor=1.0, total_iters=warmup,
        )
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(total - warmup, 1), eta_min=0.0,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup],
        )
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total, eta_min=0.0,
    )


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(ckpt_dir: Path, epoch: int, step: int, model: PretrainModelV2,
                     optimizer, scheduler) -> Path:
    state = {
        "epoch": epoch,
        "step": step,
        "online_backbone": model.online_backbone.state_dict(),
        "momentum_backbone": model.momentum_backbone.state_dict(),
        "pred_mlp": model.pred_mlp.state_dict(),
        "recon_head": model.recon_head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "version": "v2",
    }
    path = ckpt_dir / f"epoch_{epoch:04d}.pt"
    torch.save(state, path)
    return path


def _load_checkpoint(path: Path, model: PretrainModelV2, optimizer, scheduler, device):
    state = torch.load(path, map_location=device, weights_only=False)
    model.online_backbone.load_state_dict(state["online_backbone"])
    model.momentum_backbone.load_state_dict(state["momentum_backbone"])
    model.pred_mlp.load_state_dict(state["pred_mlp"])
    model.recon_head.load_state_dict(state["recon_head"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    return state["epoch"] + 1, state["step"] + 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    args = _parse_args(argv)
    cfg = Config.from_yaml(args.config)

    pretrain_names = [n.strip() for n in args.pretrain_datasets.split(",")]
    if len(pretrain_names) < 2 or len(pretrain_names) > 3:
        raise ValueError(
            f"--pretrain-datasets needs 2 or 3 names; got {pretrain_names}"
        )
    bad = [n for n in pretrain_names if n not in VALID_DATASETS]
    if bad:
        raise ValueError(f"unknown dataset(s): {bad}; valid = {sorted(VALID_DATASETS)}")
    held_out = sorted(VALID_DATASETS - set(pretrain_names))
    print(f"pretrain on: {pretrain_names}")
    print(f"held out (for downstream zero-shot eval): {held_out}")

    if args.ckpt_dir is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.ckpt_dir = f"runs/pretrain/v2_{args.variant}_{ts}"
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_yaml(str(ckpt_dir / "config.yaml"))
    # Drop a tiny run-spec sidecar so the held-out dataset + variant
    # are recoverable from the checkpoint dir alone.
    (ckpt_dir / "v2_run.txt").write_text(
        f"variant: {args.variant}\n"
        f"pretrain_datasets: {','.join(pretrain_names)}\n"
        f"held_out: {','.join(held_out)}\n"
        f"epochs_per_dataset: {args.epochs_per_dataset}\n"
        f"n_subjects_per_dataset: {args.n_subjects_per_dataset}\n"
    )

    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"device: {device}")

    # Load datasets.
    datasets: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for name in pretrain_names:
        print(f"\n--- loading {name} ---")
        X, pos, ch_names = _load_dataset(name, cfg, n_subjects=args.n_subjects_per_dataset)
        pos = pos.to(device)
        print(f"  X={tuple(X.shape)}  M={pos.shape[0]}  ch_names[0..5]={ch_names[:5]}")
        datasets[name] = (X, pos)

    # Build mixed corpus.
    corpus = MixedCorpus(
        datasets={n: (X, pos) for n, (X, pos) in datasets.items()},
        batch_size=cfg.train.batch_size,
        epochs_per_dataset=args.epochs_per_dataset,
        seed=0,
    )
    batches_per_epoch = corpus.batches_per_pass
    total_steps = cfg.train.n_epochs * batches_per_epoch
    print(
        f"\nmixed corpus: {len(datasets)} datasets x "
        f"{args.epochs_per_dataset} epochs/ds = "
        f"{batches_per_epoch} batches/epoch ({total_steps} total steps)"
    )

    # Model + optimizer.
    model = _build_model(cfg, variant=args.variant).to(device)
    n_params = sum(p.numel() for p in model.online_parameters())
    print(f"variant: {args.variant}  |  online parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.online_parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = _build_lr_scheduler(optimizer, cfg)

    start_epoch, global_step = 0, 0
    if args.resume is not None:
        if args.resume == "latest":
            ckpts = sorted(ckpt_dir.glob("epoch_*.pt"))
            if not ckpts:
                print("no checkpoint to resume; starting fresh")
            else:
                start_epoch, global_step = _load_checkpoint(
                    ckpts[-1], model, optimizer, scheduler, device
                )
                print(f"resumed from {ckpts[-1].name} "
                      f"(epoch {start_epoch}, step {global_step})")
        else:
            path = ckpt_dir / args.resume
            start_epoch, global_step = _load_checkpoint(
                path, model, optimizer, scheduler, device
            )
            print(f"resumed from {path.name}")

    arch = cfg.arch
    T_p = arch.n_patches(cfg.preprocess.samples_per_epoch)

    # Training loop.
    for epoch in range(start_epoch, cfg.train.n_epochs):
        model.train()
        sum_L = sum_LA = sum_LR = 0.0
        per_ds_counts = {n: 0 for n in datasets}

        for name, x_batch in corpus:
            x_batch = x_batch.to(device, non_blocking=True)

            # Geometry tables for THIS dataset; recomputed every step (MLP
            # weights change with each opt.step).
            bias_tables, val_tables = model.online_backbone.precompute_geom_tables(
                datasets[name][1]
            )
            temporal_mask = torch.rand(
                x_batch.shape[0], T_p, device=device
            ) < cfg.ablation.mask_ratio

            L, L_A, L_R = model(
                x_batch,
                temporal_mask=temporal_mask,
                bias_tables=bias_tables,
                value_tables=val_tables,
            )
            optimizer.zero_grad()
            L.backward()
            nn.utils.clip_grad_norm_(model.online_parameters(), max_norm=1.0)
            optimizer.step()

            tau = PretrainModelV2.cosine_tau(
                global_step, total_steps,
                cfg.train.tau_base, cfg.train.tau_final,
            )
            model.update_ema(tau=tau)

            sum_L += L.item()
            sum_LA += L_A.item()
            sum_LR += L_R.item()
            per_ds_counts[name] += 1
            global_step += 1

        scheduler.step()
        avg = lambda v: v / max(batches_per_epoch, 1)
        per_ds = " ".join(f"{n}={c}" for n, c in per_ds_counts.items())
        print(
            f"epoch {epoch:04d} | L={avg(sum_L):.4f}  "
            f"L_A={avg(sum_LA):.4f}  L_R={avg(sum_LR):.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}  tau={tau:.5f}  ({per_ds})",
            flush=True,
        )

        if (epoch + 1) % 10 == 0 or epoch == cfg.train.n_epochs - 1:
            path = _save_checkpoint(
                ckpt_dir, epoch, global_step, model, optimizer, scheduler
            )
            print(f"  -> {path}", flush=True)


if __name__ == "__main__":
    main()
