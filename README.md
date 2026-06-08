# Geometrically Inductive Self-Supervised Learning for EEG

ECEN 525 (Brain–Computer Interaction), Spring 2026 — final project. Sole author:
Tianxin Zhou.

A self-supervised EEG encoder whose spatial attention is conditioned on a
geometric descriptor `g_ij` (built from 3-D scalp coordinates) instead of a
learned per-electrode codex. Because the attention function carries no
per-electrode parameters, the trained encoder applies to **any montage given
coordinates alone**, and can be frozen and evaluated **zero-shot on a montage
never seen in pretraining**.

**Headline (zero-shot BCIC-2B, pretrained on 156 mixed subjects):** geometric
0.545 BAC beats the channel-independent baseline 0.509 by +0.036 (p=0.023,
7/9 subjects); the codex baseline, evaluated by name-aligned transfer, reaches
only 0.522.

The compiled deliverables are [report/main.pdf](report/main.pdf) (report) and
[report/slides.pdf](report/slides.pdf) (talk). The code below reproduces the
results they cite.

## Pipeline

```
scripts/pretrain.py   # mixed-corpus pretrain (variant × leave-one-montage-out split)
scripts/probe.py      # frozen zero-shot linear probe on the held-out montage
scripts/results.py    # aggregate per-checkpoint JSONs -> results/{n9,n78}/headline.txt
```

## Dependencies

MNE-Python, MOABB, PyTorch, scikit-learn, NumPy, PyYAML, SciPy. See
`pyproject.toml`.
