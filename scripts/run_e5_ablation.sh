#!/usr/bin/env bash
# E5 ablation (proposal §5 / O1): G1 vs G2 vs G3 x {full, pos_only}, n=156.
# G3 x {full, pos_only} already exist (v2_geometric_no_bcic_n78,
# v2_geomposonly_no_bcic_n78). This runs the 4 missing cells:
#   G1+full, G1+pos_only, G2+full, G2+pos_only
# Each: pretrain 100 epochs on PhysioNet+Sleep (78+78=156), then zero-shot
# LOSO probe on held-out BCIC-2B. Sequential to avoid GPU OOM. ~45 min/run.
#
# Autonomous overnight run: continues past individual failures, logs each
# step, and writes a DONE sentinel + summary at the end.

set -u
cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"
export EEG_CACHE_DIR="$ROOT/cache"   # reuse preprocessed signal cache; else each run re-reads raw EDFs
PY="$ROOT/.venv/Scripts/python.exe"
OUTDIR="$ROOT/runs/e5_ablation"
mkdir -p "$OUTDIR"
LOG="$OUTDIR/run.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

# name | config
RUNS=(
  "v2_g1full_no_bcic_n78|configs/v2/pretrain/v2_g1_full.yaml"
  "v2_g1posonly_no_bcic_n78|configs/v2/pretrain/v2_g1_pos_only.yaml"
  "v2_g2full_no_bcic_n78|configs/v2/pretrain/v2_g2_full.yaml"
  "v2_g2posonly_no_bcic_n78|configs/v2/pretrain/v2_g2_pos_only.yaml"
)

log "=== E5 ablation orchestration START ==="
log "python: $PY"

for entry in "${RUNS[@]}"; do
  name="${entry%%|*}"
  cfg="${entry##*|}"
  ckptdir="runs/pretrain/$name"
  ckpt="$ckptdir/epoch_0099.pt"

  if [ -f "$ckpt" ]; then
    log ">>> SKIP pretrain $name (epoch_0099.pt already exists)"
  else
    log ">>> PRETRAIN $name  (config=$cfg)"
    "$PY" scripts/v2_pretrain.py \
      --config "$cfg" \
      --variant geometric \
      --pretrain-datasets physionet_mi,sleep_edfx \
      --epochs-per-dataset 4096 \
      --n-subjects-per-dataset 78 \
      --ckpt-dir "$ckptdir" \
      --device cuda >> "$LOG" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
      log "!!! PRETRAIN $name FAILED (rc=$rc) -- skipping probe, continuing"
      continue
    fi
    log "<<< PRETRAIN $name done"
  fi

  log ">>> PROBE $name"
  "$PY" scripts/v2_probe.py \
    --checkpoint "$ckpt" \
    --device cuda >> "$LOG" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    log "!!! PROBE $name FAILED (rc=$rc) -- continuing"
  else
    log "<<< PROBE $name done -> $ckptdir/probe_bcic_2b_loso.json"
  fi
done

log "=== summarizing ==="
"$PY" scripts/v2_e5_summary.py >> "$LOG" 2>&1 || log "!!! summary script failed"

log "=== E5 ablation orchestration DONE ==="
touch "$OUTDIR/DONE"
