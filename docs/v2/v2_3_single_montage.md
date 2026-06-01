# v2.3 — Single-montage pretraining (no mixing)

**Question.** The v2 headline pretrains on a *mix* of two montages and transfers
zero-shot to a third. v2.3 asks two sharper questions with **single-montage**
pretraining (no mixing):

1. Does **dense single-montage coverage** alone transfer zero-shot? (PhysioNet source)
2. Does pretraining transfer to **genuinely unseen positions**? (Sleep source)

## Two source experiments

| | Pretrain on | Zero-shot eval (both held out) |
|---|---|---|
| **A** | PhysioNet MI (full) | Sleep-EDFx, BCIC-2B |
| **B** | Sleep-EDFx (full) | BCIC-2B, PhysioNet MI |

Each runs 3 variants (geometric R^10 / pos_only R^6 / chind), identical training,
only the spatial encoder differs. `--pretrain-datasets <one>` makes the other two
the held-out set, both probed from one set of pretrain runs.

## The novel-position subtlety (read before interpreting)

The geometric claim is strongest when the **test montage's positions were never
seen in pretraining**. Across these 3 datasets that is hard to arrange cleanly:

- **A) PhysioNet → {Sleep, BCIC}:** PhysioNet's dense 64-ch layout *spans* both
  targets. BCIC's C3/Cz/C4 are a literal subset; Sleep's midpoints fall *between*
  PhysioNet electrodes (0.036 and 0.201 away, both < PhysioNet's 0.224 electrode
  spacing). So the test positions are effectively **in-distribution** — this arm
  tests transfer to a novel *configuration/sparsity*, not novel *positions*.
- **B) Sleep → {BCIC, PhysioNet}:** Sleep's 2 midpoints are far from BCIC's
  central strip and most of PhysioNet's head, so the test positions are
  **genuinely novel**. *But* a Sleep-pretrained model sees only **2 positions**
  (one pairwise `g_ij`) — almost no coverage to learn a position function. So B
  is a **hard, possibly-null** test: a flat result means "2 points isn't enough
  to learn the function," **not** "geometry doesn't transfer."

Net: triangulate, don't expect a single clean proof. Codex is absent in every
arm (cannot run on an unseen montage at all) — that point holds regardless.

## Run

```bash
bash scripts/run_v2_3.sh
# defaults: N_SUBJECTS=full, EPOCHS_PER_DATASET=4096, DEVICE=cuda, PYTHON=python
# override e.g.: N_SUBJECTS=78 DEVICE=cuda PYTHON=.venv/bin/python bash scripts/run_v2_3.sh
```

The script pretrains all 6 runs (3 variants × 2 sources), probes each on its two
held-out montages (Sleep = within-subject night split; BCIC/PhysioNet = LOSO),
then prints the paired geom-vs-chind / pos_only-vs-chind comparison via
`scripts/v2_3_stats.py` (paired t + Wilcoxon, mean diff, win count, d_z, 95% CI).
Outputs: `runs/pretrain/v2_3_<source>_<variant>/`.

## Implementation note

Only code change: `scripts/v2_pretrain.py`'s pretrain-dataset guard was relaxed
from 2–3 to **1–3**. `MixedCorpus` round-robin degenerates cleanly to a single
dataset (verified; no `>=2` assumption).
