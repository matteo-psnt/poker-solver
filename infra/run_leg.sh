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
# WHY that is safe to retry: `poker-solver-run train-static --iterations` is ABSOLUTE
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
#   RUN_CHECKPOINT_EVERY  static only: checkpoint every N iterations
#   RUN_OP          train (default) | evaluate
#   RUN_EVAL_METHOD lbr | exact_br | rollout      RUN_EVAL_AT  comma-separated rungs
#   RUN_EVAL_FLAGS_HEX  hex-encoded extra evaluate flags
set -euo pipefail

# Overridable ONLY so this script can be exercised off-node; the default is the
# node's data disk and nothing in Batch sets it. Without this the publish/refresh
# logic below could not be tested anywhere but on a live node, which is how a
# publish ordering bug would reach the share before anyone noticed.
WORK="${WORK:-/mnt/work}"
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
    # Per-snapshot COMPLETION MARKERS. Manifest-last protects the manifest, but a
    # kill during the copy of one snapshot still leaves that directory partial --
    # and since the manifest may already name it, a later fetch pulls down
    # truncated arrays ("mmap length is greater than file size"). Writing a marker
    # only after a snapshot copies cleanly moves the same guarantee down to
    # per-directory granularity, and needs no atomic rename (SMB has none).
    local failed=0 d base
    for d in "$run_dir"*/ ; do
      [ -d "$d" ] || continue
      base=$(basename "$d")
      case "$base" in
        # static-* MUST be here too. The static ladder's snapshots are named
        # static-<iter>.zarr, so they fell to the unguarded branch below and were
        # published with NO completion marker -- an interrupted publish then left
        # a partial rung on the share that nothing could tell from a whole one.
        # That is how static-10000000.zarr of the 30M run became a corrupt
        # artifact: a scoring leg fetched it cleanly, then died with "error
        # during blosc decompression: 0". Publishing every ~250k iterations
        # meant many chances to be interrupted mid-copy.
        checkpoint-*|keys-*|static-*)
          rm -f "$ARCHIVE/$name/.complete-$base" 2>/dev/null || true
          if cp -ru "$d" "$ARCHIVE/$name/" 2>>/tmp/publish_err; then
            : > "$ARCHIVE/$name/.complete-$base" 2>/dev/null || true
          else
            failed=1
          fi ;;
        *) cp -ru "$d" "$ARCHIVE/$name/" 2>>/tmp/publish_err || failed=1 ;;
      esac
    done
    # Loose files (.run.json, metrics.jsonl, result json), manifest excluded.
    find "$run_dir" -maxdepth 1 -type f ! -name CHECKPOINT.json \
        -exec cp -u {} "$ARCHIVE/$name/" \; 2>>/tmp/publish_err || failed=1
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
  publish_log
}

# Batch keeps a task's stdout/stderr ON THE NODE. The pool scales to zero the
# moment a task ends, so a failed leg's logs are destroyed within minutes of
# being the only record of why it failed -- which is exactly what happened to a
# 30M leg: it died at ~720k iterations and the node was reclaimed before the
# logs could be read, leaving `exit 1` and nothing else.
#
# Copied to the share on every publish, not only at exit, so a leg that is later
# killed outright still leaves the log behind.
publish_log() {
  [ -n "${LEG_LOG:-}" ] && [ -f "$LEG_LOG" ] || return 0
  mkdir -p "$SHARE/logs" 2>/dev/null || true
  # tail: the interesting part is always the end, and a multi-hour tqdm stream is
  # mostly progress-bar repaints that would cost more to copy than they inform.
  tail -c 2000000 "$LEG_LOG" > "$SHARE/logs/${AZ_BATCH_TASK_ID:-leg}.log" 2>/dev/null || true
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

# Refresh node-local abstractions from the share.
#
# The pool's START TASK copies `$SHARE/combo_abstraction/.` down, but it runs
# once, at BOOT. An abstraction published after a node came up is therefore
# invisible to that node -- so a `precompute` leg followed by a `train` leg would
# work only if the training happened to land on a node that booted later. This
# closes that window: without it the new precompute op is usable only after the
# pool recycles.
#
# Cheap in steady state: `-u` copies only what is missing or newer, so a node
# that already has every abstraction pays a directory walk, not the ~773 MB.
# Deliberately NOT fatal -- a share hiccup must not kill a leg whose abstraction
# is already on local disk.
if [ -d "$SHARE/combo_abstraction" ]; then
  mkdir -p "$DATA/combo_abstraction"
  # ONLY directories a completion marker vouches for. An unmarked one is either
  # mid-publish or was interrupted, and copying it down yields the truncated-mmap
  # failure the ladder fetch above documents at length.
  #
  # Abstractions uploaded by the older `just push-data` carry no marker, and that
  # is fine: the start task already copied them down at boot, so this refresh
  # only ever needs to catch what was published SINCE. Requiring the marker keeps
  # the completeness guarantee absolute instead of trading it for a
  # backward-compatibility path nothing depends on.
  for marker in "$SHARE"/combo_abstraction/.complete-*; do
    [ -f "$marker" ] || continue
    name=$(basename "$marker"); name="${name#.complete-}"
    [ -d "$SHARE/combo_abstraction/$name" ] || continue
    cp -ru "$SHARE/combo_abstraction/$name" "$DATA/combo_abstraction/" 2>/dev/null || true
  done
fi

# Fetch ONLY what the manifest names, never the whole archive directory.
#
# A task killed mid-publish leaves PARTIALLY-COPIED snapshot directories behind.
# Publishing the manifest last keeps the manifest itself honest, but the orphan
# files still sit there -- and copying them down produced a truncated checkpoint
# that failed with "mmap length is greater than file size". Worse, a later
# `cp -u` would skip them as already present, so the corruption would persist.
#
# The manifest IS the definition of what is complete. Anything it does not name
# is by construction unfinished, and is ignored.
# repair-ladder reads the share IN PLACE and needs nothing on the node, so it
# must not pay this. Without the guard it fell to the catch-all copy below and
# spent 25+ minutes duplicating a 16 GB ladder it then ignored.
if [ "${RUN_OP:-train}" != "repair-ladder" ] \
   && [ -n "${RUN_ID:-}" ] && [ -d "$ARCHIVE/$RUN_ID" ]; then
  log "fetching published checkpoint for $RUN_ID"
  mkdir -p "$RUNS/$RUN_ID"
  src="$ARCHIVE/$RUN_ID"
  if [ -f "$src/CHECKPOINT.json" ]; then
    wanted=$(python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
names = {d['zarr'], d['key_table']}
for e in d.get('retained', []):
    names.update((e['zarr'], e['key_table']))
print(chr(10).join(sorted(n for n in names if n)))
" "$src/CHECKPOINT.json" || true)
    # Everything that is not a snapshot dir: metadata, metrics, eval records.
    find "$src" -maxdepth 1 -mindepth 1 \
         ! -name 'checkpoint-*' ! -name 'keys-*' ! -name CHECKPOINT.json \
         -exec cp -ru {} "$RUNS/$RUN_ID/" \; 2>/dev/null || true
    # Only snapshots that BOTH the manifest names and a completion marker
    # vouches for. A dir without its marker was interrupted mid-copy.
    kept=0
    while IFS= read -r n; do
      [ -n "$n" ] || continue
      if [ -e "$src/$n" ] && [ -f "$src/.complete-$n" ]; then
        cp -ru "$src/$n" "$RUNS/$RUN_ID/" 2>/dev/null || true
        kept=$((kept + 1))
      elif [ -e "$src/$n" ]; then
        log "  skipping $n: no completion marker (published copy was interrupted)"
      fi
    done <<<"$wanted"
    # Manifest last here too, so a torn fetch never claims more than it copied.
    cp -u "$src/CHECKPOINT.json" "$RUNS/$RUN_ID/" 2>/dev/null || true
    log "fetched $kept complete snapshot dir(s)"
  elif [ "${RUN_OP:-train}" = "evaluate" ] && [ -n "${RUN_EVAL_AT:-}" ]; then
    # SELECTIVE. A static run has no CHECKPOINT.json, so the copy below would
    # take the WHOLE ladder -- thirty ~540 MB rungs, ~16 GB, to score three of
    # them. Scoring needs the manifest and the rungs actually named.
    find "$src" -maxdepth 1 -mindepth 1 ! -name 'static-*' \
         -exec cp -ru {} "$RUNS/$RUN_ID/" \; 2>/dev/null || true
    IFS=',' read -ra WANT <<< "$RUN_EVAL_AT"
    for it in "${WANT[@]}"; do
      [ -n "$it" ] || continue
      if [ -d "$src/static-$it.zarr" ]; then
        # rm first, and NO -u. A cancelled task leaves partial rungs on the node,
        # and `cp -u` treats those as already-present and skips them -- so the
        # next task inherits a TRUNCATED checkpoint and dies inside zarr. That
        # is what happened to rung 10000000: "fetched" in one second, then a
        # read error. Node-local state is never evidence of a complete copy.
        if [ ! -f "$src/.complete-static-$it.zarr" ]; then
          # Refuse rather than load: an unmarked rung is either pre-marker or was
          # interrupted mid-publish, and the two are indistinguishable from here.
          # Loading it yields a corrupt-chunk error deep in zarr minutes later.
          log "  WARN rung $it has no completion marker -- refusing (may be partial)"
          continue
        fi
        rm -rf "$RUNS/$RUN_ID/static-$it.zarr"
        if cp -r "$src/static-$it.zarr" "$RUNS/$RUN_ID/" 2>/tmp/fetch_err; then
          log "  fetched rung $it"
        else
          # Reported, not swallowed: a silent copy failure becomes a confusing
          # load error several minutes later, in a different subsystem.
          log "  WARN rung $it copy FAILED: $(tail -1 /tmp/fetch_err 2>/dev/null)"
        fi
      else
        log "  WARN rung $it not on the share"
      fi
    done
  else
    cp -ru "$src/." "$RUNS/$RUN_ID/"
  fi
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

# Wall-clock ceiling for the training process itself. The task-level
# maxWallClockTime (P1D) is not a backstop for a hang: it is longer than any leg
# is meant to run, so a wedged process bills a full node-day before Batch acts.
# One leg proved this -- training died, the process could not exit, and the task
# stayed `running` indefinitely.
#
# --signal=TERM first so the trap below still publishes; --kill-after guarantees
# the process dies even if it ignores TERM, which is the case that hung.
RUN_TIMEOUT="${RUN_TIMEOUT:-6h}"
GUARD=(timeout --signal=TERM --kill-after=120s "$RUN_TIMEOUT")

# Tee'd to node-local disk, then copied to the share by publish_log. Node-local
# first because the training stream is chatty and writing every line straight to
# SMB would put the leg's throughput at the mercy of the share.
LEG_LOG="$WORK/leg-${AZ_BATCH_TASK_ID:-local}.log"

# `|| rc=$?` because `set -e` would abort before the exit code could be read,
# and a timed-out leg must still reach the reporting below. PIPESTATUS, not $?,
# because the tee makes this a pipeline and $? would report tee's success.
# REPAIR mode. Rungs published before completion markers covered `static-*` are
# indistinguishable from interrupted ones, so the fetch now (correctly) refuses
# them all -- including the 30M run's entire ladder. Deleting and retraining is
# absurd for a 2.6h run whose data is mostly fine, and blanket-marking them would
# reinstate exactly the bug the markers exist to prevent.
#
# So PROVE each rung instead: copy it, load it, and mark it only if zarr can
# actually decompress every chunk. A rung that fails is left unmarked and named,
# which is a permanently honest record -- corrupt data is discovered here, once,
# rather than in a scoring leg minutes deep.
# PRECOMPUTE mode. Builds a card abstraction on the node and publishes it to the
# share, so an abstraction experiment never needs a laptop.
#
# WHY this belongs here at all: every other op consumed abstractions that were
# built locally and pushed with `just push-data`, which made "precompute" the one
# step that could not leave a workstation. Precompute saturates every core for
# ~10-40 minutes depending on bucket count, which is exactly the kind of work a
# node exists for.
#
# RUN_CONFIG names an ABSTRACTION config here (config/abstraction/<stem>.yaml),
# not a training config. The two namespaces are disjoint, the op is explicit, and
# reusing the variable keeps `_task`'s signature unchanged.
#
# Output lands node-locally for free: `$CODE/data` is symlinked to `$DATA`, so
# the CLI's own `data/combo_abstraction/<name>` IS `/mnt/work/data/...`. That
# means a training task landing on THIS node can use it immediately; the copy to
# the share is what makes it durable and available to every future node, whose
# start task pulls the share down at boot.
if [ "${RUN_OP:-train}" = "precompute" ]; then
  [ -n "${RUN_CONFIG:-}" ] || { log "FATAL precompute needs RUN_CONFIG (abstraction stem)"; exit 1; }
  log "precompute: abstraction config=$RUN_CONFIG (timeout $RUN_TIMEOUT)"
  set +o pipefail
  "${GUARD[@]}" uv run poker-solver-run precompute --config "$RUN_CONFIG" 2>&1 | tee -a "$LEG_LOG"
  rc=${PIPESTATUS[0]}
  set -o pipefail
  if [ "$rc" != 0 ]; then
    publish_log
    log "precompute FAILED rc=$rc"
    exit "$rc"
  fi

  # Publish every abstraction the node holds that the share does not.
  #
  # ORDER IS THE CORRECTNESS ARGUMENT, exactly as for the ladder above: a node's
  # start task copies `$SHARE/combo_abstraction/.` down wholesale, so a
  # half-copied directory would be consumed as if complete and fail later inside
  # a memory-mapped read. Copy into a TEMP name, then rename, then write the
  # marker. A reader that sees the marker is guaranteed a complete directory.
  mkdir -p "$SHARE/combo_abstraction"
  published=0
  for src in "$DATA"/combo_abstraction/*/; do
    [ -d "$src" ] || continue
    name=$(basename "$src")
    if [ -f "$SHARE/combo_abstraction/.complete-$name" ]; then
      continue
    fi
    log "  publishing $name"
    staging="$SHARE/combo_abstraction/.staging-$name-${AZ_BATCH_TASK_ID:-local}"
    rm -rf "$staging" 2>/dev/null || true
    if cp -r "$src" "$staging" 2>/dev/null; then
      rm -rf "$SHARE/combo_abstraction/$name" 2>/dev/null || true
      mv "$staging" "$SHARE/combo_abstraction/$name"
      : > "$SHARE/combo_abstraction/.complete-$name"
      published=$((published + 1))
    else
      log "  WARN failed to publish $name; leaving the share untouched"
      rm -rf "$staging" 2>/dev/null || true
    fi
  done
  publish_log
  log "precompute complete: $published abstraction(s) published"
  exit 0
fi

if [ "${RUN_OP:-train}" = "repair-ladder" ]; then
  src="$ARCHIVE/$RUN_ID"
  [ -d "$src" ] || { log "FATAL no such run on the share: $RUN_ID"; exit 1; }
  log "verifying ladder for $RUN_ID (reading the share in place)"
  set +o pipefail
  "${GUARD[@]}" uv run python "$CODE/infra/verify_ladder.py" "$RUN_CONFIG" "$src" 2>&1 | tee -a "$LEG_LOG"
  rc=${PIPESTATUS[0]}
  set -o pipefail
  publish_log
  log "repair finished rc=$rc"
  exit "$rc"
fi

# EVALUATE mode. Scoring belongs on the node, not on a laptop: the share is a
# local mount here and a WAN download away from anywhere else -- one checkpoint
# is ~540 MB of small zarr chunks, which took ~20 minutes to pull over SMB and
# would make scoring a 30-rung ladder impractical.
#
# RUN_EVAL_AT takes a COMMA-SEPARATED list of rungs, scored in one task. The
# fetch dominates the cost, so a whole convergence curve for the price of one.
if [ "${RUN_OP:-train}" = "evaluate" ]; then
  EVAL_FLAGS=""
  if [ -n "${RUN_EVAL_FLAGS_HEX:-}" ]; then
    EVAL_FLAGS=$(python3 -c "import sys; sys.stdout.write(bytes.fromhex(sys.argv[1]).decode())" "$RUN_EVAL_FLAGS_HEX")
  fi
  # shellcheck disable=SC2206
  EXTRA=($EVAL_FLAGS)
  METHOD="${RUN_EVAL_METHOD:-exact_br}"
  ok=0
  bad=0
  IFS=',' read -ra RUNGS <<< "${RUN_EVAL_AT:-}"
  [ "${#RUNGS[@]}" -eq 0 ] && RUNGS=("")
  for rung in "${RUNGS[@]}"; do
    AT=()
    [ -n "$rung" ] && AT=(--at "$rung")
    log "evaluate: run=$RUN_ID method=$METHOD ${rung:+at=$rung}"
    set +o pipefail
    "${GUARD[@]}" uv run poker-solver-run evaluate \
      --run "$RUN_ID" --method "$METHOD" "${AT[@]}" "${EXTRA[@]}" 2>&1 | tee -a "$LEG_LOG"
    step=${PIPESTATUS[0]}
    set -o pipefail
    # One bad rung must not abandon the rest: a partial curve beats none, and
    # the failure is visible in the log and absent from the ledger.
    if [ "$step" = 0 ]; then ok=$((ok + 1)); else
      bad=$((bad + 1)); log "WARN rung ${rung:-latest} failed (rc=$step)"
    fi
    publish_log
  done
  log "evaluate complete: $ok scored, $bad failed"
  # Exit 0 when ANYTHING scored. A non-zero exit makes Batch retry the WHOLE
  # task, re-fetching and re-scoring the rungs that already succeeded -- one bad
  # rung turned a 30-minute job into nearly four hours and wrote every record
  # twice. A partial result is not a failure to retry; the gap is visible in this
  # log and absent from the ledger. Only a clean sweep of failures is worth a
  # retry, since that is what a transient node fault looks like.
  [ "$ok" -gt 0 ] && exit 0
  exit 1
fi

# One command covers both fresh and continuing: `train-static --run <id>`
# continues an existing run and `--iterations` is an ABSOLUTE target, so
# re-running past it is a no-op. The fresh/resume split the dynamic path needed
# no longer exists, and neither does the dynamic path.
rc=0
# Derive a STABLE run id from the task when none was given. A Batch retry keeps
# the same task id, so the retry continues this run rather than starting a second
# one from zero -- which is what makes retries safe here.
STATIC_RUN="${RUN_ID:-run-${AZ_BATCH_TASK_ID:-local}}"
STATIC_ARGS=(--run "$STATIC_RUN")
[ -n "${RUN_CHECKPOINT_EVERY:-}" ] && STATIC_ARGS+=(--checkpoint-every "$RUN_CHECKPOINT_EVERY")
log "train-static: config=${RUN_CONFIG:-} run=$STATIC_RUN to=$RUN_TO (timeout $RUN_TIMEOUT)"
set +o pipefail
"${GUARD[@]}" uv run poker-solver-run train-static \
  --config "$RUN_CONFIG" --iterations "$RUN_TO" \
  "${STATIC_ARGS[@]}" "${ARGS[@]}" 2>&1 | tee -a "$LEG_LOG"
rc=${PIPESTATUS[0]}
set -o pipefail
publish_log
# 124 is timeout's own "deadline expired"; surface it as itself rather than as a
# training failure, so `just jobs` distinguishes a hang from a crash.
if [ "$rc" = 124 ] || [ "$rc" = 137 ]; then
  log "TIMEOUT after $RUN_TIMEOUT -- leg killed; published rungs are on the share"
  exit 124
fi
[ "$rc" = 0 ] || exit "$rc"

log "leg complete"
