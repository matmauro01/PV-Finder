#!/usr/bin/env bash
# Resumable fetch of the July-2026 extended-|eta| PU200 ntuples from CERN EOS.
#
# Why not `sftp get -r`: sftp overwrites rather than skips, so re-running a
# recursive get re-downloads (and temporarily destroys) files that were already
# complete. rsync skips size-matched files and repairs partial ones in place via
# the delta algorithm, so this script is safe to re-run any number of times.
#
# Authentication: relies on the ssh ControlMaster declared for host `lxplus` in
# ~/.ssh/config. Open it once by hand (password + 2FA) before running:
#     ssh lxplus true
# then this script reuses the socket with no further prompts.
#
# Usage (from the repo root, inside tmux):
#     bash scripts/fetch_run4_all_etas.sh
set -uo pipefail

REMOTE_HOST="lxplus"
REMOTE_DIR="/eos/project/r/rocky-bala-garg/Rocky/Athena_Project/DataSets/PVfinderData/ATLAS_Data_processed/Run4_MC21_ITk_LatestJuly2026"
LOCAL_DIR="/data/home/matmauro/codice/PV-Finder/data/run4_all_etas/Run4_MC21_ITk_LatestJuly2026"
MAX_RETRIES=20

if ! ssh -o BatchMode=yes "$REMOTE_HOST" true 2>/dev/null; then
    echo "ERROR: no usable ssh session to $REMOTE_HOST." >&2
    echo "Run 'ssh $REMOTE_HOST true' interactively first (password + 2FA)," >&2
    echo "then re-run this script; the ControlMaster socket will be reused." >&2
    exit 1
fi

mkdir -p "$LOCAL_DIR"

# --inplace  : repair the partial file where it lies (no 100 GB temp copy)
# --size-only: EOS mtimes are not meaningful; size match == already complete
# --partial  : keep what arrived if the link drops mid-file
RSYNC_OPTS=(-rlt --inplace --partial --size-only --human-readable --info=progress2)

attempt=1
while [ "$attempt" -le "$MAX_RETRIES" ]; do
    echo "=== rsync attempt $attempt/$MAX_RETRIES  ($(date '+%F %T')) ==="
    rsync "${RSYNC_OPTS[@]}" -e ssh \
        "${REMOTE_HOST}:${REMOTE_DIR}/" "${LOCAL_DIR}/"
    rc=$?
    if [ "$rc" -eq 0 ]; then
        echo "=== rsync completed cleanly ($(date '+%F %T')) ==="
        break
    fi
    echo "--- rsync exited $rc; retrying in 30 s ---"
    sleep 30
    attempt=$((attempt + 1))
done

echo
echo "=== Verifying every ROOT file opens and reporting event counts ==="
cd /data/home/matmauro/codice/PV-Finder || exit 1
source venv/bin/activate
python -u - "$LOCAL_DIR" <<'PY'
import glob
import os
import sys

import uproot

root_dir = sys.argv[1]
total, bad = 0, []
for path in sorted(glob.glob(os.path.join(root_dir, "**", "*.root"), recursive=True)):
    name = os.path.basename(path)
    size_gb = os.path.getsize(path) / 1e9
    try:
        n = uproot.open(path)["PVFinderData"].num_entries
    except Exception as exc:
        bad.append(name)
        print(f"  FAIL {name:62s} {size_gb:7.1f} GB  {type(exc).__name__}")
        continue
    total += n
    print(f"  ok   {name:62s} {size_gb:7.1f} GB  {n:>9,} events")

print(f"\nTotal readable events: {total:,}")
if bad:
    print(f"UNREADABLE ({len(bad)}): " + ", ".join(bad))
    sys.exit(1)
print("All files readable.")
PY
