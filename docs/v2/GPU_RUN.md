# GPU machine run sheet — v2.3 single-montage experiments

Instructions for the Claude Code instance on the GPU box. The repo does **not**
pull automatically — start by fetching this branch.

## 1. Get the code

```bash
cd <repo>
git fetch origin
git checkout v2-improvements
git pull origin v2-improvements
```

## 2. Environment

- Python env with the project deps (torch + CUDA, mne, moabb, scikit-learn,
  scipy, pyyaml). Use the box's GPU env; pass it via `PYTHON=...` if it is not
  the default `python`.
- Datasets: the script auto-downloads via MNE/MOABB on first use. Set
  `MNE_DATA` / `EEG_CACHE_DIR` if a cache already exists, to skip re-download
  and re-preprocessing. First run will be slow while data downloads.

## 3. Launch everything

```bash
DEVICE=cuda PYTHON=python bash scripts/run_v2_3.sh
```

This pretrains 6 models (3 variants × 2 sources) on **full** single montages and
zero-shot probes each on its two held-out montages, then prints the paired
significance tests. Defaults: `N_SUBJECTS=full`, `EPOCHS_PER_DATASET=4096`.

What it runs:
- **A** PhysioNet → {Sleep-EDFx, BCIC-2B}
- **B** Sleep-EDFx → {BCIC-2B, PhysioNet MI}

Long job — run under `tmux`/`nohup` so it survives disconnects:
```bash
nohup env DEVICE=cuda bash scripts/run_v2_3.sh > runs/v2_3.log 2>&1 &
tail -f runs/v2_3.log
```

## 4. Collect results

- Probe JSONs: `runs/pretrain/v2_3_<source>_<variant>/probe_<dataset>_<protocol>.json`
- The `=== stats: ... ===` blocks in the log give geom-vs-chind and
  pos_only-vs-chind: mean BAC, mean diff, wins, paired-t p, Wilcoxon p, d_z, 95% CI.
- Re-print stats for any pair without re-running, e.g.:
  ```bash
  python scripts/v2_3_stats.py \
    --geom     runs/pretrain/v2_3_sleep_edfx_geometric/probe_bcic_2b_loso.json \
    --pos-only runs/pretrain/v2_3_sleep_edfx_pos_only/probe_bcic_2b_loso.json \
    --chind    runs/pretrain/v2_3_sleep_edfx_chind/probe_bcic_2b_loso.json
  ```

## 5. Report back

Commit the probe JSONs + the log and push to `v2-improvements`, or paste the four
`=== stats ===` blocks (one per source × held-out pair) back to Tianxin.

## Interpretation (so the result isn't over-read)

- **A / PhysioNet source:** test positions are effectively in-distribution
  (dense coverage spans the targets). A win shows *mixing isn't required*; it does
  **not** prove novel-position transfer.
- **B / Sleep source:** genuinely novel test positions, but only **2** positions
  seen in pretraining — a hard, possibly-null test. A flat result means "2 points
  isn't enough to learn the position function," not "geometry fails."
- Full discussion: `docs/v2/v2_3_single_montage.md`.
