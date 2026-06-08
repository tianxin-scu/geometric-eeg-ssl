"""Significance test: single-montage zero-shot, geom/pos_only vs chind.

Reads the three probe JSONs for one held-out eval dataset (geometric,
geometric_pos_only, chind), aligns per-subject balanced accuracy on the
common subjects, and runs the same paired comparison the headline used:
paired t-test + Wilcoxon signed-rank, mean diff, win count, Cohen's d_z,
and a 95% CI on the mean difference.

Usage:
    python scripts/stats.py \\
        --geom      runs/pretrain/geometric/probe_sleep_edfx_within_subject_night.json \\
        --pos-only  runs/pretrain/pos_only/probe_sleep_edfx_within_subject_night.json \\
        --chind     runs/pretrain/chind/probe_sleep_edfx_within_subject_night.json

Any of --pos-only / --chind may be omitted; comparisons that lack a baseline
are skipped.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from scipy import stats


def _load_per_subject_bac(path: str) -> Dict[str, float]:
    """Return {subject_id(str): balanced_accuracy} from a probe JSON."""
    with open(path) as f:
        d = json.load(f)
    return {
        str(subj): float(m["balanced_accuracy"])
        for subj, m in d["per_subject"].items()
    }


def _paired_report(name_a: str, a: Dict[str, float],
                   name_b: str, b: Dict[str, float]) -> None:
    common = sorted(set(a) & set(b), key=lambda s: int(s))
    if len(common) < 2:
        print(f"  {name_a} vs {name_b}: <2 common subjects, skipping.")
        return
    va = np.array([a[s] for s in common])
    vb = np.array([b[s] for s in common])
    diff = va - vb
    mean_diff = diff.mean()
    wins = int((diff > 0).sum())
    n = len(common)

    # Paired t-test + Wilcoxon (two-sided).
    t_p = stats.ttest_rel(va, vb).pvalue
    try:
        w_p = stats.wilcoxon(va, vb).pvalue
    except ValueError:  # all-zero differences
        w_p = float("nan")

    # Cohen's d_z (paired) and 95% CI on the mean difference.
    sd = diff.std(ddof=1)
    dz = mean_diff / sd if sd > 0 else float("nan")
    sem = sd / math.sqrt(n) if n > 0 else float("nan")
    tcrit = stats.t.ppf(0.975, n - 1)
    ci_lo, ci_hi = mean_diff - tcrit * sem, mean_diff + tcrit * sem

    print(f"  {name_a:18s} mean BAC = {va.mean():.4f}")
    print(f"  {name_b:18s} mean BAC = {vb.mean():.4f}")
    print(f"    mean diff = {mean_diff:+.4f}   wins = {wins}/{n}   "
          f"d_z = {dz:.2f}")
    print(f"    paired t p = {t_p:.4f}   Wilcoxon p = {w_p:.4f}   "
          f"95% CI = [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    sig = "SIGNIFICANT" if (t_p < 0.05) else "not significant"
    print(f"    -> {sig} at alpha=0.05\n")


def main(argv=None):
    p = argparse.ArgumentParser(description="Paired significance test.")
    p.add_argument("--geom", required=True, help="geometric (full) probe JSON")
    p.add_argument("--pos-only", default=None, help="pos_only probe JSON")
    p.add_argument("--chind", default=None, help="chind probe JSON")
    args = p.parse_args(argv)

    geom = _load_per_subject_bac(args.geom)
    print(f"\n=== paired comparison ({Path(args.geom).name}) ===\n")

    if args.chind:
        chind = _load_per_subject_bac(args.chind)
        _paired_report("geometric (full)", geom, "chind", chind)
        if args.pos_only:
            pos = _load_per_subject_bac(args.pos_only)
            _paired_report("geometric pos_only", pos, "chind", chind)
    else:
        print("  no --chind baseline given; nothing to compare against.")


if __name__ == "__main__":
    main()
