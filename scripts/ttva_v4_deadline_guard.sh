#!/bin/bash
# Hard wall-clock guard for the v4 TTVA training campaign.
#
# The campaign has a fixed training budget. Both arms checkpoint every epoch
# (save_frequency: 1), so stopping at the deadline always leaves a loadable
# state_dict; what it does NOT leave is a completed cosine anneal, so any arm
# stopped by this guard must be reported as un-annealed rather than final.
#
# SIGTERM only. Never SIGKILL: on this kernel a hard-killed process that is
# blocked in the wrong place becomes an unkillable core-spinner, and this box
# already has nine of those.
#
# Usage (detached):
#   tmux new-session -d -s deadline "bash scripts/ttva_v4_deadline_guard.sh 09:45"
set -u
DEADLINE=${1:?usage: ttva_v4_deadline_guard.sh HH:MM}
SESSIONS=${SESSIONS:-"ttva_train_noaug ttva_train_aug"}

target=$(date -d "today $DEADLINE" +%s)
now=$(date +%s)
[ "$target" -le "$now" ] && target=$(date -d "tomorrow $DEADLINE" +%s)

echo "[guard] deadline $DEADLINE ($(date -d "@$target")), $((target - now)) s away"
while [ "$(date +%s)" -lt "$target" ]; do
    sleep 30
done

echo "[guard] $(date +%H:%M:%S) DEADLINE REACHED - stopping training"
for s in $SESSIONS; do
    if tmux has-session -t "$s" 2>/dev/null; then
        echo "[guard] SIGTERM -> $s"
        pkill -TERM -f "train_ttva.*$( [ "$s" = ttva_train_aug ] && echo v4_aug || echo v4_noaug )" || true
    fi
done

sleep 60
echo "[guard] remaining train_ttva processes: $(pgrep -cf train_ttva || echo 0)"
echo "[guard] done"
