#!/usr/bin/env bash
# =============================================================================
# v2.3 — single-montage pretraining, zero-shot to the other two montages.
#
# Two source experiments, each pretrained on ONE montage (no mixing), then
# probed zero-shot on BOTH held-out montages:
#
#   A) PhysioNet MI  ->  {Sleep-EDFx, BCIC-2B}
#   B) Sleep-EDFx    ->  {BCIC-2B, PhysioNet MI}
#
# Three variants per source, identical training, only the spatial encoder
# differs:  geometric (full R^10 g3) / pos_only (R^6 g3) / chind (no geometry).
#
# Run on the GPU box:   bash scripts/run_v2_3.sh
# Full dataset is the default (all subjects). Override knobs inline, e.g.:
#   N_SUBJECTS=78 EPOCHS_PER_DATASET=4096 DEVICE=cuda bash scripts/run_v2_3.sh
#
# Rationale, the novel-position discussion, and the M=2 caveat are in
# docs/v2/v2_3_single_montage.md.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

# ---- knobs ------------------------------------------------------------------
N_SUBJECTS="${N_SUBJECTS:-full}"                  # "full" = all subjects; or an int cap
EPOCHS_PER_DATASET="${EPOCHS_PER_DATASET:-4096}"  # per pass; multiple of batch_size (64)
DEVICE="${DEVICE:-cuda}"
PYTHON="${PYTHON:-python}"
ROOT="runs/pretrain"
DEFAULT_CFG="configs/v2/pretrain/v2_default.yaml"
POSONLY_CFG="configs/v2/pretrain/v2_pos_only.yaml"

SUBJ_ARG=""
[ "$N_SUBJECTS" != "full" ] && SUBJ_ARG="--n-subjects-per-dataset $N_SUBJECTS"

echo "=== v2.3 single-montage (N_SUBJECTS=$N_SUBJECTS, epochs/ds=$EPOCHS_PER_DATASET, device=$DEVICE) ==="

# protocol-aware probe filename: phys/bcic = loso, sleep = within_subject_night
probe_json () {  # $1=ckpt_dir  $2=dataset
  case "$2" in
    sleep_edfx) echo "$1/probe_sleep_edfx_within_subject_night.json" ;;
    *)          echo "$1/probe_$2_loso.json" ;;
  esac
}
latest_ckpt () { ls -1 "$1"/epoch_*.pt | sort | tail -n1; }

pretrain () {  # $1=variant  $2=config  $3=source_dataset  $4=ckpt_dir
  echo ">>> pretrain variant=$1 source=$3 -> $4"
  "$PYTHON" scripts/v2_pretrain.py \
    --config "$2" \
    --pretrain-datasets "$3" \
    --epochs-per-dataset "$EPOCHS_PER_DATASET" \
    $SUBJ_ARG \
    --variant "$1" \
    --device "$DEVICE" \
    --ckpt-dir "$4"
}

probe () {  # $1=ckpt_dir  $2=variant  $3=eval_dataset
  local ckpt; ckpt="$(latest_ckpt "$1")"
  echo ">>> probe $(basename "$1") on $3"
  "$PYTHON" scripts/v2_probe.py \
    --checkpoint "$ckpt" --variant "$2" --dataset "$3" --device "$DEVICE"
}

# run one source experiment: pretrain 3 variants, probe both held-out, stats
run_source () {  # $1=source  $2..=held_out datasets
  local src="$1"; shift
  local held=("$@")
  local g="$ROOT/v2_3_${src}_geometric"
  local p="$ROOT/v2_3_${src}_pos_only"
  local c="$ROOT/v2_3_${src}_chind"

  pretrain geometric "$DEFAULT_CFG" "$src" "$g"
  pretrain geometric "$POSONLY_CFG" "$src" "$p"   # pos_only = geometric + descriptor in cfg
  pretrain chind     "$DEFAULT_CFG" "$src" "$c"

  for ds in "${held[@]}"; do
    probe "$g" geometric "$ds"
    probe "$p" geometric "$ds"
    probe "$c" chind     "$ds"
  done

  for ds in "${held[@]}"; do
    echo "=== stats: source=$src  held-out=$ds ==="
    "$PYTHON" scripts/v2_3_stats.py \
      --geom     "$(probe_json "$g" "$ds")" \
      --pos-only "$(probe_json "$p" "$ds")" \
      --chind    "$(probe_json "$c" "$ds")"
  done
}

# ---- A) PhysioNet -> {Sleep, BCIC} -----------------------------------------
run_source physionet_mi sleep_edfx bcic_2b

# ---- B) Sleep -> {BCIC, PhysioNet} -----------------------------------------
run_source sleep_edfx bcic_2b physionet_mi

echo "=== v2.3 done. checkpoints + probe JSONs under $ROOT/v2_3_* ==="
