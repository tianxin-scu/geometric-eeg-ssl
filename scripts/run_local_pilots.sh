#!/usr/bin/env bash
# Run the four pilot pretraining variants needed for E1/E2/E5/E7 if Colab's
# full-scale runs aren't finished. Each pilot is 10 subjects x 100 epochs on
# MPS, ~35 min/run per Session 9 timings (~2.5 h total).
#
# Already-completed runs are skipped (checkpoint at epoch_0099.pt detected).
# G1 pilot is assumed to already exist under runs/pretrain/g1_pilot/.
#
# Usage:
#   bash scripts/run_local_pilots.sh             # all four
#   bash scripts/run_local_pilots.sh codex g2    # subset
#   PILOT_SUBJECTS=1-20 bash scripts/run_local_pilots.sh   # override subjects
#   PILOT_DEVICE=cuda bash scripts/run_local_pilots.sh     # override device

set -euo pipefail

cd "$(dirname "$0")/.."

PILOT_SUBJECTS="${PILOT_SUBJECTS:-1-10}"
PILOT_DEVICE="${PILOT_DEVICE:-mps}"

# macOS ships bash 3.2 (no associative arrays); use a case statement instead.
config_for() {
    case "$1" in
        codex) echo "transductive_codex.yaml" ;;
        g2)    echo "geometric_g2.yaml" ;;
        g3)    echo "geometric_g3.yaml" ;;
        chind) echo "channel_independent.yaml" ;;
        *)     echo "" ;;
    esac
}

# If no args, run all four. Otherwise run the named subset.
if [[ $# -eq 0 ]]; then
    VARIANTS=(codex g2 g3 chind)
else
    VARIANTS=("$@")
fi

for variant in "${VARIANTS[@]}"; do
    config_yaml="$(config_for "$variant")"
    if [[ -z "$config_yaml" ]]; then
        echo "Unknown variant: $variant (valid: codex g2 g3 chind)"
        exit 1
    fi

    config="configs/pretrain/${config_yaml}"
    ckpt_dir="runs/pretrain/${variant}_pilot"
    final_ckpt="${ckpt_dir}/epoch_0099.pt"

    if [[ -f "$final_ckpt" ]]; then
        echo ">>> ${variant}: already complete (${final_ckpt}); skipping."
        continue
    fi

    mkdir -p "$ckpt_dir"
    echo ""
    echo "=== ${variant} pilot: ${config} on subjects ${PILOT_SUBJECTS} ==="
    python3 scripts/pretrain.py \
        --config "$config" \
        --ckpt-dir "$ckpt_dir" \
        --subjects "$PILOT_SUBJECTS" \
        --device "$PILOT_DEVICE" \
        2>&1 | tee "${ckpt_dir}/train_log.txt"
done

echo ""
echo "All requested pilots done. Checkpoints under runs/pretrain/*_pilot/."
echo "Now run:"
echo "  python scripts/run_e1.py           # in-distribution"
echo "  python scripts/run_e2.py b         # cross-subject LOSO + Wilcoxon"
echo "  python scripts/run_e2.py c         # cross-montage BCIC-2B"
echo "  python scripts/run_e5.py           # G1/G2/G3 ablation"
echo "  python scripts/run_e7.py           # channel-independent sanity"
echo "  python scripts/run_e2.py a         # Sleep-EDFx cross-session"
