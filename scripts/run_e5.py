"""E5 — G1 vs. G2 vs. G3 ablation (where g_ij enters attention).

Proposal §5/E5: G1 (score-bias only) should be competitive with G2/G3
at lower complexity. Reports BAC on PhysioNet MI (LOSO) and on BCIC-2B
(cross-montage), plus parameter count.

Usage:
    python scripts/run_e5.py [--device mps]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _eval_utils import (
    REPO_ROOT, Variant, evaluate_variants, print_bac_table,
    resolve_checkpoint,
)

VARIANTS = [
    Variant(label="G1 (score-bias only)", run_basename="g1"),
    Variant(label="G2 (value-modulation only)", run_basename="g2"),
    Variant(label="G3 (score + value)", run_basename="g3"),
]


def _report_parameter_counts() -> None:
    """Print distinguishing-parameter counts using the existing helper.

    The geometric variants differ only in their attention-side MLPs; the
    shared backbone is identical.
    """
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from src.config import Config
    from src.model.geometric_attention import GeometricSpatialEncoder

    arch = Config().arch
    print("\nDistinguishing-parameter counts (spatial encoder only):")
    print(f"{'Variant':<25} {'params':>12}")
    for label, inject in [("G1", "g1"), ("G2", "g2"), ("G3", "g3")]:
        enc = GeometricSpatialEncoder(
            d_model=arch.d_model, n_heads=arch.n_heads_spatial,
            n_layers=arch.n_layers_spatial,
            geometry_injection=inject,
            geom_mlp_hidden=arch.geom_mlp_hidden,
            dropout=arch.dropout,
        )
        n = sum(p.numel() for p in enc.parameters())
        print(f"  {label:<23} {n:>12,}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="E5 G1/G2/G3 ablation.")
    p.add_argument("--subjects", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--max-iter", type=int, default=None)
    p.add_argument("--skip-bcic2b", action="store_true",
                   help="Skip the cross-montage half of the ablation.")
    p.add_argument("--out-tag", default=None,
                   help="Subdirectory under runs/eval/ to group results (e.g. pilot_10subj).")
    args = p.parse_args(argv)

    eval_root = REPO_ROOT / "runs" / "eval"
    if args.out_tag:
        eval_root = eval_root / args.out_tag
    output_root = eval_root / "e5"

    # Half A: in-distribution PhysioNet MI LOSO.
    indist = evaluate_variants(
        VARIANTS, dataset="physionet_mi", eval_protocol="loso",
        output_root=output_root / "physionet_mi",
        subjects=args.subjects, device=args.device, max_iter=args.max_iter,
    )
    print_bac_table(indist, title="E5 — PhysioNet MI LOSO (in-distribution)")

    # Half B: cross-montage on BCIC-2B (zero-shot from 64-ch pretrain).
    if not args.skip_bcic2b:
        cross = evaluate_variants(
            VARIANTS, dataset="bcic_2b", eval_protocol="loso",
            output_root=output_root / "bcic_2b",
            subjects=None, device=args.device, max_iter=args.max_iter,
        )
        print_bac_table(cross, title="E5 — BCIC-2B LOSO (cross-montage)")

    _report_parameter_counts()


if __name__ == "__main__":
    main()
