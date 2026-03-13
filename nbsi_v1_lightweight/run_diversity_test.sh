#!/bin/bash
# =============================================================================
# NBSI Diversity Test Runner
#
# Cycles through files in the_file_folder one at a time:
#   1. Move file to nbsi-inbox
#   2. Start NBSI, wait for ingestion, run queries, exit
#   3. Delete file from nbsi-inbox
#   4. Repeat for next file
#
# Usage:
#   chmod +x run_diversity_test.sh
#   ./run_diversity_test.sh
#
# Setup:
#   mkdir -p the_file_folder nbsi-inbox
#   # Put test files in the_file_folder/
# =============================================================================

set -e

# -- Config -------------------------------------------------------------------
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
FILE_FOLDER="$PROJECT_DIR/the_file_folder"
INBOX="$HOME/nbsi-inbox"
MAIN="$PROJECT_DIR/nbsi/main.py"
QUERY_1="structural layer"
QUERY_2="observation node lifecycle"
WAIT_FOR_INGEST=60   # seconds to wait for worker to finish ingesting
LOG_FILE="$PROJECT_DIR/diversity_test.log"

# -- Colours ------------------------------------------------------------------
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log() { echo -e "${GREEN}[test]${NC} $1" | tee -a "$LOG_FILE"; }
warn() { echo -e "${YELLOW}[warn]${NC} $1" | tee -a "$LOG_FILE"; }
fail() { echo -e "${RED}[fail]${NC} $1" | tee -a "$LOG_FILE"; exit 1; }

# -- Checks -------------------------------------------------------------------
[ -d "$FILE_FOLDER" ] || fail "the_file_folder not found at $FILE_FOLDER"
[ -f "$MAIN" ] || fail "main.py not found at $MAIN"
mkdir -p "$INBOX"

FILES=("$FILE_FOLDER"/*)
[ ${#FILES[@]} -gt 0 ] || fail "No files found in $FILE_FOLDER"

log "========================================================"
log "NBSI Diversity Test — $(date)"
log "Files to cycle: ${#FILES[@]}"
log "Project: $PROJECT_DIR"
log "Inbox: $INBOX"
log "========================================================"

# -- Cycle --------------------------------------------------------------------
CYCLE=0
for FILE in "${FILES[@]}"; do
    [ -f "$FILE" ] || continue
    CYCLE=$((CYCLE + 1))
    FNAME=$(basename "$FILE")

    log ""
    log "Cycle $CYCLE — $FNAME"
    log "  Moving to inbox..."
    cp "$FILE" "$INBOX/$FNAME"

    log "  Starting NBSI session..."

    # Build input: wait for ingestion then send queries then exit
    # The sleep gives the worker time to ingest before we query
    INPUT=$(printf "dummy_startup_query\n")  # ignored — just wakes stdin
    
    # Use expect-style here-doc piped to python
    # We send queries after waiting for ingestion to complete
    (
        sleep "$WAIT_FOR_INGEST"
        echo "$QUERY_1"
        sleep 3
        echo "$QUERY_2"
        sleep 3
        echo ":exit"
    ) | PYTHONPATH="$PROJECT_DIR" python "$MAIN" --watch "$INBOX" --no-synth \
        >> "$LOG_FILE" 2>&1

    log "  Session complete."

    log "  Cleaning inbox..."
    rm -f "$INBOX/$FNAME"

    log "  Cycle $CYCLE done."

    # Brief pause between cycles
    sleep 2
done

log ""
log "========================================================"
log "All $CYCLE cycles complete."
log ""
log "Checking library state..."
python3 - << 'PYEOF' | tee -a "$LOG_FILE"
import json, sys, os, numpy as np
from scipy.stats import wasserstein_distance
from itertools import combinations

path = os.path.expanduser("~/.nbsi/library.json")
if not os.path.exists(path):
    print("No library found.")
    sys.exit(0)

with open(path) as f:
    d = json.load(f)

obs = d.get("observations", [])
print(f"Structural nodes : {d['node_count']}")
print(f"Observation nodes: {d['observation_count']}")
print()

# Confirmation distribution
from collections import Counter
counts = Counter(o["confirmation_count"] for o in obs)
print("Confirmation distribution:")
for k in sorted(counts):
    print(f"  confirmations={k}: {counts[k]} nodes")
print()

# Diversity check
dists = d.get("session_distributions", {})
print(f"Session distributions stored: {len(dists)}")
diversity_values = []
for obs_id, dist_lists in dists.items():
    arrays = [np.array(x) for x in dist_lists]
    if len(arrays) >= 2:
        pairwise = [wasserstein_distance(a, b) for a, b in combinations(arrays, 2)]
        diversity_values.append(np.mean(pairwise))

if diversity_values:
    print(f"Diversity scores across obs nodes:")
    print(f"  min:  {min(diversity_values):.4f}")
    print(f"  max:  {max(diversity_values):.4f}")
    print(f"  mean: {np.mean(diversity_values):.4f}")
    non_zero = sum(1 for v in diversity_values if v > 0.001)
    print(f"  non-zero: {non_zero}/{len(diversity_values)}")
else:
    print("No diversity data yet.")

# Top candidates
high = [o for o in obs if o["confirmation_count"] >= 5]
print(f"\nNodes with confirmations >= 5: {len(high)}")
for o in sorted(high, key=lambda x: -x["confirmation_count"])[:5]:
    print(f"  confirmations={o['confirmation_count']}  "
          f"diversity={o['diversity_score']:.3f}  "
          f"stab={o['stabilisation_score']:.3f}")
PYEOF

log "========================================================"
log "Log saved to: $LOG_FILE"
