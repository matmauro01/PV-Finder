#!/usr/bin/env bash
# Full-pileup GNN v2 campaign. Run in tmux "gnn_fullmu_v2".
set -euo pipefail
REPO=/data/home/matmauro/codice/PV-Finder
RUN_DIR="$REPO/outputs/11_11_2026_output/gnn_fullmu_v2"
mkdir -p "$RUN_DIR"
cd "$REPO"
ulimit -v 67108864  # 64 GB: the trainer sits at a stable 47.8 GB VSZ (CUDA reservations, RSS 9 GB)
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=2
export PYTHONUNBUFFERED=1
exec > >(tee -a "$RUN_DIR/campaign.log") 2>&1
date -u
"$REPO/venv/bin/python" -u scripts/gnn_fullmu_v2_campaign.py prepare
# Wait for headroom on the shared GPU; never touch other jobs.
while true; do
    FREE_MB=$(nvidia-smi --id=2 --query-gpu=memory.free --format=csv,noheader,nounits)
    if [ "$FREE_MB" -ge 10000 ]; then break; fi
    printf 'Waiting for 10 GB free on GPU 2; currently %s MB.\n' "$FREE_MB"
    sleep 30
done
"$REPO/venv/bin/python" -u scripts/gnn_fullmu_v2_campaign.py train 2>&1 | tee -a "$RUN_DIR/training.log"
date -u
