#!/usr/bin/env bash
# Unattended finisher: wait for the chind n78 pretrain to complete, eval both
# n78 checkpoints on BCIC, write an n9-vs-n78 comparison, then shut down.
# Launched in the background so it runs while the user is away.
set -u
export EEG_CACHE_DIR="$(pwd)/cache"
export MNE_DATA="$(pwd)/mne_data"
PY=.venv/Scripts/python.exe

CKPT_POS=runs/pretrain/v2_posonly_no_bcic_n78/epoch_0099.pt
CKPT_CHIND=runs/pretrain/v2_chind_no_bcic_n78/epoch_0099.pt

echo "[orch] $(date) waiting for chind n78 to finish (epoch_0099.pt)..."
until [ -f "$CKPT_CHIND" ]; do sleep 30; done
echo "[orch] $(date) chind checkpoint present; 30s grace for final flush"
sleep 30

mkdir -p runs/eval/v2_n78

echo "[orch] $(date) eval pos_only n78 on BCIC"
$PY scripts/v2_probe.py --checkpoint "$CKPT_POS" --subjects 1-9 \
    --output runs/eval/v2_n78/geometric_pos_only_bcic_2b.json --device cuda
echo "[orch] EXIT_pos=$?"

echo "[orch] $(date) eval chind n78 on BCIC"
$PY scripts/v2_probe.py --checkpoint "$CKPT_CHIND" --subjects 1-9 \
    --output runs/eval/v2_n78/chind_bcic_2b.json --device cuda
echo "[orch] EXIT_chind=$?"

# Standalone n78 table (best effort).
$PY scripts/v2_results.py --eval-dir runs/eval/v2_n78 --out-dir results/v2_n78 || true

echo "[orch] $(date) building n9-vs-n78 comparison"
$PY - <<'PYEOF'
import json, os
def bac(p):
    try:
        a = json.load(open(p))["aggregate"]
        return a["bac_mean"], a["bac_std"]
    except Exception:
        return None, None
rows = [
    ("pos_only", "runs/eval/v2/geometric_pos_only_bcic_2b.json",
                 "runs/eval/v2_n78/geometric_pos_only_bcic_2b.json"),
    ("chind",    "runs/eval/v2/chind_bcic_2b.json",
                 "runs/eval/v2_n78/chind_bcic_2b.json"),
]
out = ["BCIC-2B zero-shot BAC -- pretraining corpus 9+9 vs 78+78 subjects",
       "=" * 64,
       f"{'variant':<12}{'n=9 (18 subj)':>18}{'n=78 (156 subj)':>18}{'delta':>10}",
       "-" * 58]
for name, p9, p78 in rows:
    m9, s9 = bac(p9); m78, s78 = bac(p78)
    c9 = f"{m9:.3f}+/-{s9:.3f}" if m9 is not None else "MISSING"
    c78 = f"{m78:.3f}+/-{s78:.3f}" if m78 is not None else "MISSING"
    dl = f"{m78 - m9:+.3f}" if (m9 is not None and m78 is not None) else "-"
    out.append(f"{name:<12}{c9:>18}{c78:>18}{dl:>10}")
os.makedirs("results/v2_n78", exist_ok=True)
txt = "\n".join(out) + "\n"
open("results/v2_n78/comparison_n9_vs_n78.txt", "w").write(txt)
print(txt)
PYEOF

echo "[orch] $(date) results saved under results/v2_n78/. Scheduling shutdown (120s; abort with: shutdown /a)"
powershell.exe -NoProfile -Command 'shutdown /s /t 120 /f'
echo "[orch] $(date) shutdown scheduled; orchestration complete."
