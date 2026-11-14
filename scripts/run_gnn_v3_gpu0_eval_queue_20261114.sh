#!/usr/bin/env bash
# GNN v3 campaign, 14 Nov: GPU 0 evaluation queue (eval only, never shares the GPU with training):
# p2c full chain, then run e at its last epoch (single t), then p2a full chain. tmux gnnv3_evalq0.
set -u
cd /data/home/matmauro/codice/PV-Finder
export CUDA_VISIBLE_DEVICES=0
L=outputs/11_13_2026_output/gnn_v3_campaign
bash scripts/run_gnn_v3_phase2_eval_chain.sh p2c > $L/chain_p2c.log 2>&1
# e at its last epoch (the selection rule picked epoch 2; this shows the fully trained truth network on s8)
EPOCH_e=539 TAG=phase1_e539 GPU=0 bash scripts/run_gnn_v3_eval.sh e > $L/eval_phase1_e539.log 2>&1
bash scripts/run_gnn_v3_phase2_eval_chain.sh p2a > $L/chain_p2a.log 2>&1
echo "GPU0 QUEUE DONE $(date)" >> $L/chain_p2a.log
