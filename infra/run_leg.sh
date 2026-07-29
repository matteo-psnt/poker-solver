#!/usr/bin/env bash
# Node-side wrapper for one experiment leg. This is what a Batch task runs.
#
# Shape:  fetch last published checkpoint -> train/resume to an ABSOLUTE target
#         -> publish each retained rung as it appears -> publish on any exit
#
# WHY publish mid-run rather than only at the end: the node's disk is ephemeral.
# A task killed by OOM or maxWallClockTime loses everything it has not published,
# so a multi-hour leg would die with nothing to show -- the same class of failure
# as REBUILD.md Decision 5 ("checkpoint retention destroyed the primary
# evidence"), in a new form.
#
# WHY that is safe to retry: `poker-solver-run resume --to-iteration` is ABSOLUTE
# and no-ops once the target is reached, so a Batch retry converges on the same
# endpoint however many times it runs. Publishing the ladder makes a retry cheap;
# the absolute target is what makes it correct.
#
# Env (set by the justfile's `_task` recipe):
#   RUN_CONFIG      training config stem, e.g. production  (fresh runs only)
#   RUN_TO          ABSOLUTE target iteration
#   RUN_ID          run id to resume; empty for a fresh train
#   CODE_SNAPSHOT   names <share>/code/<snapshot>.tar.gz, pinned by this submission
#   CODE_DIR        where the task already extracted that tarball
#   RUN_EXPERIMENT  experiment id      RUN_ARM     arm label
#   RUN_PARENT      parent run id
#   RUN_SETS_HEX    hex-encoded, space-separated k=v config overrides
#   RUN_WORKERS     worker count (empty = all CPUs)
set -euo pipefail

WORK=/mnt/work
SHARE="${AZ_BATCH_NODE_MOUNTS_DIR:-/mnt/batch/tasks/fsmounts}/shared"
# Set by the task command line, which extracts there. Task-owned, and unique
# per task so concurrent legs on one node cannot share a tree.
CODE="${CODE_DIR:-$WORK/code}"
DATA="$WORK/data"
RUNS="$DATA/runs"
ARCHIVE="$SHARE/archive"

log() { echo "[run_leg $(date -u +%H:%M:%S)] $*"; }

# --- publish ----------------------------------------------------------------- #
# Idempotent and safe to call while training continues. `cp -u` skips rungs
# already on the share, so the cost tracks what is NEW rather than the run's total
# size. Never fatal: a failed publish must not kill a leg that is still making
# progress on local disk.
publish_all() {
  [ -d "$RUNS" ] || return 0
  mkdir -p "$ARCHIVE"
  local run_dir
  for run_dir in "$RUNS"/*/; do
    [ -d "$run_dir" ] || continue
    local name
    name=$(basename "$run_dir")
    mkdir -p "$ARCHIVE/$name"

    # ORDER IS THE CORRECTNESS ARGUMENT. Training continues during this copy, and
    # committing a checkpoint prunes superseded snapshots -- so a naive recursive
    # copy can have zarr directories deleted out from under it. CHECKPOINT.json is
    # uppercase and sorts BEFORE checkpoint-*.zarr in the C locale, so the default
    # order publishes the manifest first: interrupt it there and the share is left
    # naming a snapshot that was only half copied, which a later fetch or resume
    # reads as complete. Silent corruption of the durable artifact is worse than
    # losing it.
    #
    # So: everything EXCEPT the manifest first, manifest last. An interrupted copy
    # then leaves the previous manifest in place, still naming data that is fully
    # present.
    # NO --preserve=timestamps: the SMB mount refuses utime ("Operation not
    # permitted"), and cp reports that as failure AFTER copying the data -- which
    # then suppresses the manifest and makes a successful publish look broken.
    # `cp -u` still behaves correctly without it: the destination takes the copy
    # time, which is newer than the source, so an already-published file is
    # skipped and a genuinely updated one is not.
    local failed=0
    cp -ru \
        $(find "$run_dir" -maxdepth 1 -mindepth 1 ! -name CHECKPOINT.json) \
        "$ARCHIVE/$name/" 2>/tmp/publish_err || failed=1
    if [ "$failed" -eq 0 ] && [ -f "$run_dir/CHECKPOINT.json" ]; then
      cp -u "$run_dir/CHECKPOINT.json" "$ARCHIVE/$name/" 2>>/tmp/publish_err || failed=1
    fi

    if [ "$failed" -eq 0 ]; then
      log "published $name"
    else
      # Reported, never swallowed: a publish that silently fails every time turns
      # "a killed task loses one rung" into "a killed task loses everything".
      log "WARN publish incomplete for $name -- manifest NOT updated, so the share"
      log "     still describes the last fully-copied checkpoint. Reason:"
      sed 's/^/       /' /tmp/publish_err >&2 || true
    fi
  done
}

# Publish on ANY exit -- success, failure, or cancellation. An operator-cancelled
# task still leaves its progress on the share.
trap publish_all EXIT

# --- watcher ----------------------------------------------------------------- #
# Polls the checkpoint manifest rather than hooking the trainer, so the training
# layer stays unaware it is running in the cloud -- the same reason a cloud job is
# a shell invocation of the headless CLI, not a provider-specific reimplementation.
#
# Watches the whole runs directory, so it covers a FRESH train too: that run's id
# does not exist until the trainer creates it, and waiting for the id would leave
# exactly the long unprotected window this exists to close.
watch_rungs() {
  local seen=""
  while sleep 120; do
    local rungs
    rungs=$(python3 - "$RUNS" <<'PY' 2>/dev/null || true
import json, pathlib, sys
out = []
for m in sorted(pathlib.Path(sys.argv[1]).glob("*/CHECKPOINT.json")):
    try:
        d = json.loads(m.read_text())
    except Exception:
        continue
    out.append(f"{m.parent.name}:{','.join(str(r['iteration']) for r in d.get('retained', []))}")
print("|".join(out))
PY
    )
    if [ -n "$rungs" ] && [ "$rungs" != "$seen" ]; then
      log "retained ladder changed -> $rungs"
      publish_all
      seen="$rungs"
    fi
  done
}

# --- stage code -------------------------------------------------------------- #
# The code is ALREADY extracted: the task command line untars it before invoking
# this script, because this script lives inside that tarball. A sealed archive is
# used rather than a directory copy because Azure Files needs every parent path to
# exist before a write, so publishing a repo tree file-by-file both fails and
# costs a round-trip per file.
log "code snapshot '$CODE_SNAPSHOT' staged at $CODE"
mkdir -p "$RUNS"
cd "$CODE"
ln -sfn "$DATA" "$CODE/data"

log "syncing dependencies"
uv sync --quiet

if [ -n "${RUN_ID:-}" ] && [ -d "$ARCHIVE/$RUN_ID" ]; then
  log "fetching published checkpoint for $RUN_ID"
  mkdir -p "$RUNS/$RUN_ID"
  cp -ru "$ARCHIVE/$RUN_ID/." "$RUNS/$RUN_ID/"
fi

watch_rungs &
WATCHER=$!
trap 'kill "$WATCHER" 2>/dev/null || true; publish_all' EXIT

# --- run --------------------------------------------------------------------- #
# Optional flags are appended only when set: passing `--arm ""` would record an
# arm literally named empty string rather than an unaffiliated run.
ARGS=()
[ -n "${RUN_WORKERS:-}" ] && ARGS+=(--workers "$RUN_WORKERS")
[ -n "${RUN_EXPERIMENT:-}" ] && ARGS+=(--experiment "$RUN_EXPERIMENT")
[ -n "${RUN_ARM:-}" ] && ARGS+=(--arm "$RUN_ARM")
[ -n "${RUN_PARENT:-}" ] && ARGS+=(--parent "$RUN_PARENT")
# Hex-decoded: see the justfile -- a config override's value contains `=`, which
# Batch's KEY=VALUE environment-setting parser rejects.
RUN_SETS=""
if [ -n "${RUN_SETS_HEX:-}" ]; then
  RUN_SETS=$(python3 -c "import sys; sys.stdout.write(bytes.fromhex(sys.argv[1]).decode())" "$RUN_SETS_HEX")
fi
if [ -n "$RUN_SETS" ]; then
  for kv in $RUN_SETS; do [ -n "$kv" ] && ARGS+=(--set "$kv"); done
fi

if [ -z "${RUN_ID:-}" ]; then
  log "fresh train: config=$RUN_CONFIG iterations=$RUN_TO"
  uv run poker-solver-run train --config "$RUN_CONFIG" --iterations "$RUN_TO" "${ARGS[@]}"
else
  log "resume: run=$RUN_ID to=$RUN_TO (absolute)"
  uv run poker-solver-run resume --run "$RUN_ID" --to-iteration "$RUN_TO" \
    ${RUN_WORKERS:+--workers "$RUN_WORKERS"}
fi

log "leg complete"
