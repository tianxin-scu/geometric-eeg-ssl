"""Pretraining script for the geometric EEG SSL model.

Usage (geometric G1, all PhysioNet MI subjects):
    python scripts/pretrain.py --config configs/pretrain/geometric_g1.yaml

Usage (resume):
    python scripts/pretrain.py --config configs/pretrain/geometric_g1.yaml \\
        --ckpt-dir runs/pretrain/geometric_g1_20260515 --resume epoch_0009.pt

All training hyperparameters are read from cfg.train (TrainConfig).
Ablation settings (geometry variant, mask scheme, lambda_R) come from cfg.ablation
and cfg.train.  Architecture shape comes from cfg.arch.

Checkpoints are written to --ckpt-dir every 10 epochs and at the end.
The resolved config is written to <ckpt-dir>/config.yaml once at startup.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(_repo_root / "src"))  # physionet_mi uses bare 'from config import'

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import Config
from src.datasets.physionet_mi import EXCLUDED_SUBJECTS, PhysioNetMI
from src.model.backbone import Backbone
from src.model.geometric_attention import GeometricSpatialEncoder
from src.model.momentum_encoder import PretrainModel
from src.model.transductive_baseline import TransductiveSpatialEncoder


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Pretrain geometric EEG SSL model.")
    p.add_argument("--config", required=True, help="Path to config YAML.")
    p.add_argument(
        "--ckpt-dir", default=None,
        help="Checkpoint output directory. Default: runs/pretrain/<timestamp>.",
    )
    p.add_argument(
        "--subjects", default=None,
        help="Comma/hyphen-separated subject IDs, e.g. '1-10,15,20'. "
             "Default: all valid PhysioNet MI subjects (1-109 minus excluded).",
    )
    p.add_argument(
        "--device", default=None,
        help="Torch device string ('cuda', 'mps', 'cpu'). Default: auto-detect.",
    )
    p.add_argument(
        "--resume", default=None,
        help="Filename of a checkpoint inside --ckpt-dir to resume from, "
             "e.g. 'epoch_0009.pt'.",
    )
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
# Model construction
# ---------------------------------------------------------------------------

def _build_model(cfg: Config, n_electrodes: int) -> PretrainModel:
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
    backbone = Backbone(spatial, arch=arch)
    return PretrainModel(backbone, lambda_R=cfg.train.lambda_R)


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

def _sample_mask(B, T_p, M, cfg: Config, device):
    """Return (temporal_mask, spatial_mask) per ablation mask scheme."""
    scheme = cfg.ablation.mask_scheme
    ratio = cfg.ablation.mask_ratio
    temporal_mask = spatial_mask = None
    if scheme in ("temporal", "joint"):
        temporal_mask = torch.rand(B, T_p, device=device) < ratio
    if scheme in ("spatial", "joint"):
        spatial_mask = torch.rand(B, M, device=device) < ratio
    return temporal_mask, spatial_mask


# ---------------------------------------------------------------------------
# LR schedule: linear warmup then cosine decay
# ---------------------------------------------------------------------------

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

def _save_checkpoint(ckpt_dir: Path, epoch: int, step: int, model: PretrainModel,
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
    }
    path = ckpt_dir / f"epoch_{epoch:04d}.pt"
    torch.save(state, path)
    return path


def _load_checkpoint(path: Path, model: PretrainModel, optimizer, scheduler, device):
    state = torch.load(path, map_location=device)
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

    # Checkpoint directory.
    if args.ckpt_dir is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.ckpt_dir = f"runs/pretrain/{ts}"
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_yaml(str(ckpt_dir / "config.yaml"))

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

    # Dataset.
    subjects = _parse_subjects(args.subjects) if args.subjects else _all_valid_subjects()
    print(f"loading {len(subjects)} subjects …")
    ds_loader = PhysioNetMI(subjects=subjects, cfg=cfg, mode="pretrain", verbose=True)
    X_np, _y, ch_pos_np, ch_names = ds_loader.load()
    print(f"dataset shape: {X_np.shape}  ch_pos: {ch_pos_np.shape}")

    X_t = torch.from_numpy(X_np)       # (N, M, T) float32
    ch_pos = torch.from_numpy(ch_pos_np).to(device)

    N, M, T = X_t.shape
    T_p = cfg.arch.n_patches(T)
    B = cfg.train.batch_size

    dataset = TensorDataset(X_t)
    dataloader = DataLoader(
        dataset,
        batch_size=B,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )

    # Model.
    model = _build_model(cfg, n_electrodes=M).to(device)

    # Geometry is fixed per montage, but the geometry MLP weights change each
    # step, so tables must be recomputed each forward pass (not once at startup).
    # `_geom_ch_pos` is stored here and passed into the loop below.
    _geom_ch_pos = ch_pos if model.online_backbone.uses_geometry else None

    n_online = sum(p.numel() for p in model.online_parameters())
    print(f"online parameters: {n_online:,}")

    # Optimizer + LR schedule.
    optimizer = torch.optim.AdamW(
        model.online_parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = _build_lr_scheduler(optimizer, cfg)

    # Resume.
    start_epoch = 0
    global_step = 0
    if args.resume is not None:
        if args.resume == "latest":
            ckpts = sorted(ckpt_dir.glob("epoch_*.pt"))
            if not ckpts:
                raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
            resume_path = ckpts[-1]
        else:
            resume_path = ckpt_dir / args.resume
        start_epoch, global_step = _load_checkpoint(
            resume_path, model, optimizer, scheduler, device
        )
        print(f"resumed from {resume_path}  (epoch {start_epoch}, step {global_step})")

    total_steps = cfg.train.n_epochs * len(dataloader)
    print(f"training: {cfg.train.n_epochs} epochs × {len(dataloader)} batches = {total_steps} steps")

    # Training loop.
    for epoch in range(start_epoch, cfg.train.n_epochs):
        model.train()
        sum_L = sum_LA = sum_LR = 0.0

        for (x_batch,) in dataloader:
            x_batch = x_batch.to(device, non_blocking=True)
            temporal_mask, spatial_mask = _sample_mask(B, T_p, M, cfg, device)

            # Recompute geometry tables each step: the MLP weights (part of
            # online_parameters) change with each optimizer step, so the
            # tables must be rebuilt to keep the computation graph fresh.
            bias_tables = val_tables = None
            if _geom_ch_pos is not None:
                bias_tables, val_tables = model.online_backbone.precompute_geom_tables(
                    _geom_ch_pos
                )

            L, L_A, L_R = model(
                x_batch,
                temporal_mask=temporal_mask,
                bias_tables=bias_tables,
                value_tables=val_tables,
                spatial_mask=spatial_mask,
            )

            optimizer.zero_grad()
            L.backward()
            nn.utils.clip_grad_norm_(model.online_parameters(), max_norm=1.0)
            optimizer.step()

            tau = PretrainModel.cosine_tau(
                global_step, total_steps,
                cfg.train.tau_base, cfg.train.tau_final,
            )
            model.update_ema(tau=tau)

            sum_L += L.item()
            sum_LA += L_A.item()
            sum_LR += L_R.item()
            global_step += 1

        n_batches = len(dataloader)
        scheduler.step()
        print(
            f"epoch {epoch:04d} | "
            f"L={sum_L/n_batches:.4f}  "
            f"L_A={sum_LA/n_batches:.4f}  "
            f"L_R={sum_LR/n_batches:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}  "
            f"tau={tau:.5f}"
        )

        if (epoch + 1) % 10 == 0 or epoch == cfg.train.n_epochs - 1:
            ckpt_path = _save_checkpoint(
                ckpt_dir, epoch, global_step, model, optimizer, scheduler
            )
            print(f"  → {ckpt_path}")


if __name__ == "__main__":
    main()
