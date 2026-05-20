"""Shared helpers for run_e1, run_e2, run_e5, run_e7.

Resolves which checkpoint to use for each named variant (preferring full-scale
over pilot), shells out to probe.py per variant, collects the JSON results,
and prints a uniform BAC table.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_PRETRAIN = REPO_ROOT / "runs" / "pretrain"
PROBE_SCRIPT = REPO_ROOT / "scripts" / "probe.py"


@dataclass
class Variant:
    """A single pretrained model variant to evaluate."""

    label: str           # display name, e.g. "G1 (geometric, score-bias)"
    run_basename: str    # base directory name; resolved to ${name}_full or ${name}_pilot
    extra_args: tuple = ()  # extra args appended to probe.py invocation


def resolve_checkpoint(run_basename: str) -> Optional[tuple[Path, str]]:
    """Find the latest epoch checkpoint, preferring _full over _pilot.

    Returns (checkpoint_path, scale_tag) or None if neither exists.
    scale_tag is "full" or "pilot" so the eval table can surface it.
    """
    for scale in ("full", "pilot"):
        run_dir = RUNS_PRETRAIN / f"{run_basename}_{scale}"
        if not run_dir.is_dir():
            continue
        ckpts = sorted(run_dir.glob("epoch_*.pt"))
        if ckpts:
            return ckpts[-1], scale
    return None


def run_probe(
    checkpoint: Path,
    dataset: str,
    eval_protocol: str,
    output: Path,
    extra_args: tuple = (),
    subjects: Optional[str] = None,
    device: Optional[str] = None,
    max_iter: Optional[int] = None,
) -> dict:
    """Invoke scripts/probe.py and return the parsed JSON results."""
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd: list[str] = [
        sys.executable, str(PROBE_SCRIPT),
        "--checkpoint", str(checkpoint),
        "--dataset", dataset,
        "--eval-protocol", eval_protocol,
        "--output", str(output),
    ]
    if subjects is not None:
        cmd += ["--subjects", subjects]
    if device is not None:
        cmd += ["--device", device]
    if max_iter is not None:
        cmd += ["--max-iter", str(max_iter)]
    cmd += list(extra_args)

    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"probe.py failed (exit {result.returncode}): {checkpoint}")
    with open(output) as f:
        return json.load(f)


def evaluate_variants(
    variants: list[Variant],
    dataset: str,
    eval_protocol: str,
    output_root: Path,
    subjects: Optional[str] = None,
    device: Optional[str] = None,
    max_iter: Optional[int] = None,
) -> dict[str, Optional[dict]]:
    """Resolve and evaluate each variant; missing checkpoints are skipped.

    Returns dict[variant.label, result_dict_or_None].
    """
    results: dict[str, Optional[dict]] = {}
    for v in variants:
        resolved = resolve_checkpoint(v.run_basename)
        if resolved is None:
            print(f"\n--- {v.label}: SKIPPED (no checkpoint at {v.run_basename}_full or _pilot)")
            results[v.label] = None
            continue
        ckpt, scale = resolved
        out_path = output_root / f"{v.run_basename}_{scale}_{dataset}_{eval_protocol}.json"
        print(f"\n=== {v.label}  ({scale}-scale, {ckpt.name}) ===")
        try:
            r = run_probe(
                ckpt, dataset, eval_protocol, out_path,
                extra_args=v.extra_args,
                subjects=subjects, device=device, max_iter=max_iter,
            )
            r["scale"] = scale
            results[v.label] = r
        except Exception as e:
            print(f"  ERROR: {e}")
            results[v.label] = None
    return results


def print_bac_table(results: dict[str, Optional[dict]], title: str = "") -> None:
    """Pretty-print a BAC table from evaluate_variants output."""
    if title:
        print(f"\n{title}")
        print("-" * len(title))
    print(f"{'Variant':<40} {'Scale':<7} {'BAC':>10} {'std':>10} {'n_subj':>8}")
    print("-" * 80)
    for label, r in results.items():
        if r is None:
            print(f"{label:<40} {'-':<7} {'(no ckpt)':>10}")
            continue
        agg = r.get("aggregate", {})
        bac = agg.get("bac_mean")
        std = agg.get("bac_std")
        n = agg.get("n_subjects")
        scale = r.get("scale", "?")
        print(f"{label:<40} {scale:<7} {bac:>10.3f} {std:>10.3f} {n:>8}")
