"""E2 — Robustness Ladder (three subcommands).

Proposal §5/E2: the headline test of the central claim across the three
practical non-stationarity axes.

  E2(a): Cross-session within-subject on Sleep-EDFx (night 1 -> night 2).
  E2(b): Cross-subject LOSO on PhysioNet MI + paired Wilcoxon test
         between geometric and codex per-subject BAC vectors.
  E2(c): Cross-montage zero-shot transfer — pretrained on 64-ch
         PhysioNet MI, evaluated on 3-ch BCIC-2B. The transductive codex
         baseline is reported with --codex-fallback random (and nn,
         currently a random-fallback alias with a warning).

Usage:
    python scripts/run_e2.py a            # Sleep-EDFx within-subject night split
    python scripts/run_e2.py b            # PhysioNet MI LOSO + Wilcoxon
    python scripts/run_e2.py c            # Cross-montage to BCIC-2B
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

# Variants used by E2a (Sleep-EDFx) and E2b (PhysioNet MI LOSO).
# Channel-independent omitted from E2a/b by default — proposal scopes E7
# as the place where channel-independent is exercised.
ROBUSTNESS_VARIANTS = [
    Variant(label="G1 (geometric, score-bias)", run_basename="g1"),
    Variant(label="Transductive codex", run_basename="codex"),
]

# E2c additionally splits the codex baseline into two fallback flavors.
CROSS_MONTAGE_VARIANTS = [
    Variant(label="G1 (geometric, score-bias)", run_basename="g1"),
    Variant(label="Transductive codex (random fallback)",
            run_basename="codex",
            extra_args=("--codex-fallback", "random")),
    Variant(label="Transductive codex (nn fallback)",
            run_basename="codex",
            extra_args=("--codex-fallback", "nn")),
]


def _eval_root(args) -> Path:
    root = REPO_ROOT / "runs" / "eval"
    if getattr(args, "out_tag", None):
        root = root / args.out_tag
    return root


def _e2a(args) -> None:
    output_root = _eval_root(args) / "e2a"
    results = evaluate_variants(
        ROBUSTNESS_VARIANTS, dataset="sleep_edfx",
        eval_protocol="within_subject_night",
        output_root=output_root,
        subjects=args.subjects, device=args.device, max_iter=args.max_iter,
    )
    print_bac_table(results, title="E2(a) — Sleep-EDFx within-subject (n1 -> n2)")


def _e2b(args) -> None:
    output_root = _eval_root(args) / "e2b"
    results = evaluate_variants(
        ROBUSTNESS_VARIANTS, dataset="physionet_mi", eval_protocol="loso",
        output_root=output_root,
        subjects=args.subjects, device=args.device, max_iter=args.max_iter,
    )
    print_bac_table(results, title="E2(b) — PhysioNet MI LOSO")

    # Paired Wilcoxon signed-rank test between geometric and codex per-subject BAC.
    geo = results.get("G1 (geometric, score-bias)")
    cdx = results.get("Transductive codex")
    if geo and cdx:
        try:
            from scipy.stats import wilcoxon
        except ImportError:
            print("scipy not available; skipping Wilcoxon.")
            return
        geo_subs = geo["per_subject"]
        cdx_subs = cdx["per_subject"]
        common = sorted(set(geo_subs) & set(cdx_subs), key=int)
        if not common:
            print("\nWilcoxon: no overlapping subjects.")
            return
        geo_bac = [geo_subs[s]["balanced_accuracy"] for s in common]
        cdx_bac = [cdx_subs[s]["balanced_accuracy"] for s in common]
        try:
            stat, pval = wilcoxon(geo_bac, cdx_bac)
            print(f"\nWilcoxon signed-rank (geometric vs codex, n={len(common)}): "
                  f"W={stat:.3f}  p={pval:.4f}  "
                  f"({'significant' if pval < 0.05 else 'not significant'} at α=0.05)")
        except ValueError as e:
            print(f"\nWilcoxon failed: {e} (probably all-equal BAC pairs)")

        wilcoxon_out = output_root / "wilcoxon.json"
        wilcoxon_out.write_text(json.dumps({
            "subjects": common,
            "geometric_bac": geo_bac,
            "codex_bac": cdx_bac,
        }, indent=2))


def _e2c(args) -> None:
    output_root = _eval_root(args) / "e2c"
    results = evaluate_variants(
        CROSS_MONTAGE_VARIANTS, dataset="bcic_2b", eval_protocol="loso",
        output_root=output_root,
        subjects=args.subjects, device=args.device, max_iter=args.max_iter,
    )
    print_bac_table(results, title="E2(c) — Cross-montage: 64-ch pretrain -> 3-ch BCIC-2B")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="E2 robustness ladder.")
    p.add_argument("subcmd", choices=["a", "b", "c"],
                   help="E2(a) cross-session, E2(b) LOSO+Wilcoxon, E2(c) cross-montage.")
    p.add_argument("--subjects", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--max-iter", type=int, default=None)
    p.add_argument("--out-tag", default=None,
                   help="Subdirectory under runs/eval/ to group results (e.g. pilot_10subj).")
    args = p.parse_args(argv)
    {"a": _e2a, "b": _e2b, "c": _e2c}[args.subcmd](args)


if __name__ == "__main__":
    main()
