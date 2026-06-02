"""Summarize the E5 ablation: G1/G2/G3 x {full, pos_only} vs chind, n=156.

Reads the zero-shot BCIC-2B LOSO probe JSONs for every cell of the 3x2
geometry-entry x descriptor grid (plus the chind baseline), prints a table
of per-cell BAC, and runs paired tests (t + Wilcoxon) of each geometric
cell against chind over the 9 BCIC-2B LOSO subjects.

Writes results/v2_n78/e5_ablation.{txt,md}. Pure read/aggregate -- no model
or data is touched.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent

# cell label -> probe JSON
CELLS = {
    "chind":        "runs/pretrain/v2_chind_no_bcic_n78/probe_bcic_2b_loso.json",
    "G1 + full":    "runs/pretrain/v2_g1full_no_bcic_n78/probe_bcic_2b_loso.json",
    "G1 + pos_only":"runs/pretrain/v2_g1posonly_no_bcic_n78/probe_bcic_2b_loso.json",
    "G2 + full":    "runs/pretrain/v2_g2full_no_bcic_n78/probe_bcic_2b_loso.json",
    "G2 + pos_only":"runs/pretrain/v2_g2posonly_no_bcic_n78/probe_bcic_2b_loso.json",
    "G3 + full":    "runs/pretrain/v2_geometric_no_bcic_n78/probe_bcic_2b_loso.json",
    "G3 + pos_only":"runs/pretrain/v2_geomposonly_no_bcic_n78/probe_bcic_2b_loso.json",
}


def per_subject_bac(path: Path) -> dict[str, float]:
    d = json.loads(path.read_text())
    ps = d.get("per_subject", {})
    return {k: float(v["balanced_accuracy"]) for k, v in ps.items()}


def load_all() -> dict[str, dict[str, float]]:
    out = {}
    for label, rel in CELLS.items():
        p = ROOT / rel
        if p.exists():
            out[label] = per_subject_bac(p)
        else:
            out[label] = {}  # not run yet
    return out


def paired_vs_chind(cell: dict[str, float], chind: dict[str, float]):
    keys = sorted(set(cell) & set(chind))
    if len(keys) < 2:
        return None
    a = np.array([cell[k] for k in keys])
    b = np.array([chind[k] for k in keys])
    diff = a - b
    delta = float(diff.mean())
    wins = int((diff > 0).sum())
    n = len(keys)
    t_p = float(stats.ttest_rel(a, b).pvalue)
    try:
        w_p = float(stats.wilcoxon(a, b).pvalue)
    except ValueError:
        w_p = float("nan")
    dz = float(delta / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else float("nan")
    return {"delta": delta, "t_p": t_p, "w_p": w_p, "wins": wins, "n": n, "dz": dz}


def main():
    data = load_all()
    chind = data["chind"]
    chind_mean = np.mean(list(chind.values())) if chind else float("nan")

    lines = []
    lines.append("E5 ablation -- where g_ij enters x descriptor (zero-shot BCIC-2B, n=156)")
    lines.append("=" * 78)
    lines.append(f"{'cell':<16}{'BAC':>8}{'d vs chind':>12}{'wins':>7}{'t p':>9}{'Wilcoxon':>10}{'dz':>7}")
    lines.append("-" * 78)
    lines.append(f"{'chind (base)':<16}{chind_mean:>8.3f}{'':>12}{'':>7}{'':>9}{'':>10}{'':>7}")
    for label in CELLS:
        if label == "chind":
            continue
        cell = data[label]
        if not cell:
            lines.append(f"{label:<16}{'(pending)':>8}")
            continue
        mean = float(np.mean(list(cell.values())))
        st = paired_vs_chind(cell, chind)
        if st is None:
            lines.append(f"{label:<16}{mean:>8.3f}")
        else:
            lines.append(
                f"{label:<16}{mean:>8.3f}{st['delta']:>+12.3f}"
                f"{st['wins']:>4}/{st['n']:<2}{st['t_p']:>9.3f}"
                f"{st['w_p']:>10.3f}{st['dz']:>7.2f}"
            )
    table = "\n".join(lines)
    print(table)

    outdir = ROOT / "results" / "v2_n78"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "e5_ablation.txt").write_text(table + "\n")

    # markdown version
    md = ["# E5 ablation -- G1/G2/G3 x {full, pos_only}, zero-shot BCIC-2B (n=156)", "",
          "| cell | BAC | Δ vs chind | wins | t p | Wilcoxon p | d_z |",
          "|---|---|---|---|---|---|---|",
          f"| chind (baseline) | {chind_mean:.3f} | — | — | — | — | — |"]
    for label in CELLS:
        if label == "chind":
            continue
        cell = data[label]
        if not cell:
            md.append(f"| {label} | (pending) | | | | | |")
            continue
        mean = float(np.mean(list(cell.values())))
        st = paired_vs_chind(cell, chind)
        if st is None:
            md.append(f"| {label} | {mean:.3f} | | | | | |")
        else:
            md.append(
                f"| {label} | {mean:.3f} | {st['delta']:+.3f} | "
                f"{st['wins']}/{st['n']} | {st['t_p']:.3f} | {st['w_p']:.3f} | {st['dz']:.2f} |"
            )
    (outdir / "e5_ablation.md").write_text("\n".join(md) + "\n")
    print(f"\nwrote {outdir/'e5_ablation.txt'} and .md")


if __name__ == "__main__":
    main()
