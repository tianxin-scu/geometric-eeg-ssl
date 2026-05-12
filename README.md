# Geometrically Inductive Self-Supervised Learning for EEG

This is the implementation repository for the ECEN 525 Spring 2026 final project. The project explores a self-supervised EEG representation learning framework that replaces per-electrode codex embeddings with geometry-conditioned self-attention, so the same model can adapt more naturally across electrode placements, subjects, and montages. The design borrows the alignment-and-reconstruction training recipe from recent EEG foundation-model work, but the repository is for a course project, not a publication.

Status: in progress (Spring 2026 ECEN 525 final project).

## Setup

```bash
git clone <your-github-url> geometric-eeg-ssl
cd geometric-eeg-ssl
pip install -e ".[dev]"
```

Datasets are not auto-downloaded. Use the dataset preparation utilities and download the required EEG corpora through the official MNE and MOABB workflows, then point the YAML configs at your local paths.

## Reproducing Experiments

```bash
python scripts/pretrain.py --config configs/pretrain/geometric_g1.yaml
python scripts/probe.py --config configs/eval/e1_indist_probe.yaml
python scripts/run_e1.py --config configs/eval/e1_indist_probe.yaml
python scripts/run_e2.py --config configs/eval/e2a_cross_session.yaml
python scripts/run_e5.py --config configs/eval/e5_g_variants.yaml
python scripts/run_e7.py --config configs/eval/e7_channel_indep.yaml
```

## Repo Layout

```text
geometric-eeg-ssl/
├── README.md
├── LICENSE                 # MIT
├── .gitignore
├── pyproject.toml          # setuptools backend
├── configs/
│   ├── pretrain/
│   │   ├── geometric_g1.yaml
│   │   ├── geometric_g2.yaml
│   │   ├── geometric_g3.yaml
│   │   ├── transductive_codex.yaml
│   │   └── channel_independent.yaml
│   ├── data/
│   │   ├── physionet_mi.yaml
│   │   ├── bcic_2a.yaml
│   │   ├── bcic_2b.yaml
│   │   └── sleep_edfx.yaml
│   └── eval/
│       ├── e1_indist_probe.yaml
│       ├── e2a_cross_session.yaml
│       ├── e2b_loso.yaml
│       ├── e2c_cross_montage.yaml
│       ├── e5_g_variants.yaml
│       └── e7_channel_indep.yaml
├── src/
│   └── geo_eeg/
│       ├── __init__.py
│       ├── data/
│       ├── models/
│       ├── losses/
│       ├── training/
│       ├── eval/
│       └── utils/
├── scripts/
│   ├── pretrain.py
│   ├── probe.py
│   ├── run_e1.py
│   ├── run_e2.py
│   ├── run_e5.py
│   └── run_e7.py
├── notebooks/
├── tests/
│   └── test_smoke.py
└── results/
    ├── tables/
    └── figures/
```

## Citation / License

MIT License, copyright 2026 Tianxin Zhou.

```bibtex
@misc{geometric_eeg_ssl_2026,
  title        = {Geometrically Inductive Self-Supervised Learning for EEG},
  author       = {Tianxin Zhou},
  year         = {2026},
  note         = {ECEN 525 Spring 2026 final project. Citation details to be filled in at submission.}
}
```
