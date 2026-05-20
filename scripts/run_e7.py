"""E7 — Channel-Independent Sanity Check.

Proposal §5/E7: train a BIOT-style channel-independent baseline (no codex,
no geometric attention) with matched budget and compare against geometric
G1 and the transductive codex on E1 (in-distribution) and E2c (cross-
montage). Verifies that explicit spatial structure helps at all.

The channel-independent variant in this codebase is the geometric model
with `ablation.geometry_injection = none` and `ablation.use_codex = false`
(see configs/pretrain/channel_independent.yaml). No separate model class.

Usage:
    python scripts/run_e7.py [--device mps] [--skip-bcic2b]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _eval_utils import REPO_ROOT, Variant, evaluate_variants, print_bac_table


VARIANTS = [
    Variant(label="Channel-independent (no codex, no geom)",
            run_basename="chind"),
    Variant(label="G1 (geometric, score-bias)", run_basename="g1"),
    Variant(label="Transductive codex", run_basename="codex"),
]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="E7 channel-independent sanity check.")
    p.add_argument("--subjects", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--max-iter", type=int, default=None)
    p.add_argument("--skip-bcic2b", action="store_true",
                   help="Skip the cross-montage half.")
    p.add_argument("--out-tag", default=None,
                   help="Subdirectory under runs/eval/ to group results (e.g. pilot_10subj).")
    args = p.parse_args(argv)

    eval_root = REPO_ROOT / "runs" / "eval"
    if args.out_tag:
        eval_root = eval_root / args.out_tag
    output_root = eval_root / "e7"

    indist = evaluate_variants(
        VARIANTS, dataset="physionet_mi", eval_protocol="loso",
        output_root=output_root / "physionet_mi",
        subjects=args.subjects, device=args.device, max_iter=args.max_iter,
    )
    print_bac_table(indist, title="E7 — PhysioNet MI LOSO (in-distribution)")

    if not args.skip_bcic2b:
        # For channel-independent, no codex fallback needed (no codex).
        # For codex baseline, default --codex-fallback random is applied.
        cross_variants = [
            Variant(label=v.label, run_basename=v.run_basename,
                    extra_args=(("--codex-fallback", "random")
                                if v.run_basename == "codex" else ()))
            for v in VARIANTS
        ]
        cross = evaluate_variants(
            cross_variants, dataset="bcic_2b", eval_protocol="loso",
            output_root=output_root / "bcic_2b",
            subjects=None, device=args.device, max_iter=args.max_iter,
        )
        print_bac_table(cross, title="E7 — BCIC-2B LOSO (cross-montage)")


if __name__ == "__main__":
    main()
