#!/usr/bin/env bash
# Run the full eval suite locally on MPS. Logs to runs/eval/macbook/.
set -u
cd "$(dirname "$0")/.."

PY=".venv/bin/python"
LOG_DIR="runs/eval/macbook"
mkdir -p "$LOG_DIR"

run() {
    local tag="$1"; shift
    echo "=== [$tag] start: $(date) ==="
    "$PY" "$@" 2>&1 | tee "$LOG_DIR/${tag}_log.txt"
    echo "=== [$tag] done:  $(date) (exit=${PIPESTATUS[0]}) ==="
}

run e1  scripts/run_e1.py    --device mps --out-tag full
run e2a scripts/run_e2.py a  --device mps --out-tag full
run e2b scripts/run_e2.py b  --device mps --out-tag full
run e2c scripts/run_e2.py c  --device mps --out-tag full
run e5  scripts/run_e5.py    --device mps --out-tag full
run e7  scripts/run_e7.py    --device mps --out-tag full

echo "=== ALL DONE: $(date) ==="
