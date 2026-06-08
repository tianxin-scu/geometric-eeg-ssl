"""Aggregate probe JSONs into the headline cross-montage BAC table.

Reads the per-(variant, held-out) probe outputs written by
`scripts/probe.py` and emits the 3-row x 2-column (geometric, chind)
table of zero-shot BAC. Codex column is left out until that variant lands.

Usage:
    python scripts/results.py
    python scripts/results.py --eval-dir runs/eval --out-dir results/n9
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Row order = held-out dataset; column order = variant.
_ROWS = ["sleep_edfx", "bcic_2b", "physionet_mi"]
_ROW_LABEL = {
    "sleep_edfx": "Sleep-EDFx",
    "bcic_2b": "BCIC-2B",
    "physionet_mi": "PhysioNet MI",
}

# Preferred column order; any column found in the JSONs but not listed here
# is appended alphabetically. Columns are discovered dynamically so
# descriptor ablations (e.g. geometric_pos_only) appear automatically.
_COL_ORDER = ["geometric", "geometric_pos_only", "chind", "codex"]


def _col_label(col: str) -> str:
    return col


def _order_columns(cols: set[str]) -> list[str]:
    ordered = [c for c in _COL_ORDER if c in cols]
    extras = sorted(c for c in cols if c not in _COL_ORDER)
    return ordered + extras


def _load_cells(eval_dir: Path) -> dict:
    """Map (held_out, column) -> aggregate dict from each probe JSON.

    Column = variant_label (distinguishes descriptor ablations) falling back
    to variant for JSONs written before variant_label existed.
    """
    cells: dict[tuple[str, str], dict] = {}
    for jf in sorted(eval_dir.glob("*.json")):
        with open(jf) as f:
            d = json.load(f)
        held = d.get("held_out") or d.get("dataset")
        col = d.get("variant_label") or d.get("variant")
        if held is None or col is None:
            print(f"  skip {jf.name}: missing held_out/variant")
            continue
        cells[(held, col)] = {
            "bac_mean": d["aggregate"]["bac_mean"],
            "bac_std": d["aggregate"]["bac_std"],
            "n_subjects": d["aggregate"]["n_subjects"],
            "protocol": d.get("eval_protocol", "?"),
            "source": jf.name,
        }
    return cells


def _fmt(cell: dict | None) -> str:
    if cell is None:
        return "  --  "
    return f"{cell['bac_mean']:.3f}+/-{cell['bac_std']:.3f}"


def _render_markdown(cells: dict, cols: list[str]) -> str:
    header = "| Held-out (eval) | " + " | ".join(_col_label(c) for c in cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    lines = [header, sep]
    for row in _ROWS:
        cellstrs = [_fmt(cells.get((row, c))) for c in cols]
        lines.append(f"| {_ROW_LABEL[row]} | " + " | ".join(cellstrs) + " |")
    return "\n".join(lines)


def _render_text(cells: dict, cols: list[str]) -> str:
    lines = []
    lines.append("cross-montage zero-shot BAC (mean +/- std over LOSO/night folds)")
    lines.append("=" * 70)
    w = max(18, max(len(_col_label(c)) for c in cols) + 2)
    head = f"{'held-out':<14}" + "".join(f"{_col_label(c):>{w}}" for c in cols)
    lines.append(head)
    lines.append("-" * len(head))
    for row in _ROWS:
        cellstrs = "".join(f"{_fmt(cells.get((row, c))):>{w}}" for c in cols)
        lines.append(f"{_ROW_LABEL[row]:<14}{cellstrs}")
    lines.append("")
    # Per-cell provenance.
    lines.append("cells:")
    for row in _ROWS:
        for c in cols:
            cell = cells.get((row, c))
            if cell is None:
                lines.append(f"  {row:<13} {c:<20} MISSING")
            else:
                lines.append(
                    f"  {row:<13} {c:<20} BAC={cell['bac_mean']:.4f} "
                    f"+/-{cell['bac_std']:.4f}  n={cell['n_subjects']}  "
                    f"({cell['protocol']}, {cell['source']})")
    # Verdict: each geometric-family column vs chind, per row.
    lines.append("")
    chind_present = any(c == "chind" for c in cols)
    if chind_present:
        for c in cols:
            if not c.startswith("geometric"):
                continue
            wins = compared = 0
            for row in _ROWS:
                g = cells.get((row, c))
                ch = cells.get((row, "chind"))
                if g and ch:
                    compared += 1
                    wins += int(g["bac_mean"] > ch["bac_mean"])
            lines.append(f"{c} beats chind on {wins}/{compared} rows.")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", default="runs/eval")
    p.add_argument("--out-dir", default="results/n9")
    args = p.parse_args(argv)

    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cells = _load_cells(eval_dir)
    if not cells:
        raise SystemExit(f"no probe JSONs found in {eval_dir}")

    cols = _order_columns({c for (_, c) in cells})
    text = _render_text(cells, cols)
    md = _render_markdown(cells, cols)

    (out_dir / "headline.txt").write_text(text + "\n")
    (out_dir / "headline.md").write_text(md + "\n")

    print(text)
    print(f"\n-> {out_dir / 'headline.txt'}")
    print(f"-> {out_dir / 'headline.md'}")


if __name__ == "__main__":
    main()
