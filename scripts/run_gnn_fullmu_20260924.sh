#!/usr/bin/env bash
# Run in its own tmux session; logs include graph preparation and training.
set -euo pipefail
REPO=/data/home/matmauro/codice/PV-Finder
RUN_DIR="$REPO/outputs/09_24_2026_output/gnn_fullmu_v1"
mkdir -p "$RUN_DIR"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=2
export PYTHONUNBUFFERED=1
exec > >(tee -a "$RUN_DIR/campaign.log") 2>&1
date -u
printf 'Campaign shell PID: %s\n' "$$"
printf '%s\n' "$$" > "$RUN_DIR/campaign.pid"
"$REPO/venv/bin/python" -u scripts/gnn_fullmu_campaign.py prepare
"$REPO/venv/bin/python" -u scripts/gnn_fullmu_campaign.py cache
# Do not change or terminate other jobs. Wait if the chosen shared GPU has
# insufficient headroom for this small GNN; preparation needs no GPU.
while true; do
    GNN_FREE_MB=$(nvidia-smi --id=2 --query-gpu=memory.free --format=csv,noheader,nounits)
    if [ "$GNN_FREE_MB" -ge 10000 ]; then break; fi
    printf 'Waiting for 10 GB free on GPU 2; currently %s MB.\n' "$GNN_FREE_MB"
    sleep 30
done
"$REPO/venv/bin/python" -u scripts/gnn_fullmu_campaign.py train 2>&1 | tee -a "$RUN_DIR/training.log"
date -u
