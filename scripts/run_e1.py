"""E1 — In-Distribution Linear Probe (Geometric vs. Transductive on PhysioNet MI).

Proposal §5/E1: fairness anchor for all robustness claims. If geometric and
transductive are not competitive in-distribution under matched montages,
E2 isn't interpretable.

Probes each available variant (geometric G1, transductive codex, optional
channel-independent) on the standard PhysioNet MI LOSO protocol, then
prints a BAC table. Uses full-scale checkpoint if present, otherwise the
10-subject pilot.

Usage:
    python scripts/run_e1.py [--subjects 1-10] [--device mps]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _eval_utils import REPO_ROOT, Variant, evaluate_variants, print_bac_table


VARIANTS = [
    Variant(label="G1 (geometric, score-bias)", run_basename="g1"),
    Variant(label="Transductive codex", run_basename="codex"),
    Variant(label="Channel-independent (no codex, no geom)", run_basename="chind"),
]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="E1 in-distribution linear probe.")
    p.add_argument("--subjects", default=None,
                   help="Subject ID spec passed to probe.py (default: all).")
    p.add_argument("--device", default=None)
    p.add_argument("--max-iter", type=int, default=None)
    p.add_argument("--out-tag", default=None,
                   help="Subdirectory under runs/eval/ to group results (e.g. pilot_10subj).")
    args = p.parse_args(argv)

    eval_root = REPO_ROOT / "runs" / "eval"
    if args.out_tag:
        eval_root = eval_root / args.out_tag
    output_root = eval_root / "e1"
    results = evaluate_variants(
        VARIANTS, dataset="physionet_mi", eval_protocol="loso",
        output_root=output_root,
        subjects=args.subjects, device=args.device, max_iter=args.max_iter,
    )
    print_bac_table(results, title="E1 — PhysioNet MI LOSO BAC")


if __name__ == "__main__":
    main()
