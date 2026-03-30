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
#   RUN_CONFIG      training config stem, e.g. production. Needed by a CONTINUING
#                   leg too: the config builds the tree and the solver, and the
#                   checkpoint stores neither.
#   RUN_TO          ABSOLUTE target iteration
#   RUN_ID          run id to resume; empty for a fresh train
#   CODE_SNAPSHOT   names <share>/code/<snapshot>.tar.gz, pinned by this submission
#   CODE_DIR        where the task already extracted that tarball
#   RUN_EXPERIMENT  experiment id      RUN_ARM     arm label
#   RUN_PARENT      parent run id
#   RUN_SETS_JSON   JSON array of k=v config overrides
#   RUN_WORKERS     worker count (empty = `nproc`, filled in below)
#   RUN_CHECKPOINT_EVERY  static only: checkpoint every N iterations
#   RUN_OP          train (default) | evaluate
#   RUN_EVAL_METHOD lbr | exact_br | rollout      RUN_EVAL_AT  comma-separated rungs
#   RUN_EVAL_FLAGS_JSON JSON array of extra evaluate flags
set -euo pipefail

WORK=/mnt/work
SHARE="${AZ_BATCH_NODE_MOUNTS_DIR:-/mnt/batch/tasks/fsmounts}/shared"
# Set by the task command line, which extracts there. Task-owned, and unique
# per task so concurrent legs on one node cannot share a tree.
CODE="${CODE_DIR:-$WORK/code}"
DATA="$WORK/data"
RUNS="$DATA/runs"
ARCHIVE="$SHARE/archive"

# Tee'd into the published leg log, not just stdout. Batch keeps a task's
# stdout ON THE NODE and the pool drains within minutes of a task ending, so
# anything only echoed here is gone for exactly the legs worth reading later.
# The wrapper's own lines -- which op ran, how many overrides were applied,
# what got published -- are the ones that explain a leg, and they were the ones
# not surviving. `${LEG_LOG:-/dev/null}` because this is defined before it.
log() { echo "[run_leg $(date -u +%H:%M:%S)] $*" | tee -a "${LEG_LOG:-/dev/null}"; }

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
    # Loose files (.run.json, metrics.jsonl, result json), manifests excluded.
    # BOTH manifests: the static backend writes STATIC_CHECKPOINT.json, which fell
    # to this unguarded copy and so was published even when a snapshot copy above
    # had failed -- a manifest naming a half-copied rung, exactly what publishing
    # the manifest last exists to prevent. The static ladder was the unguarded one.
    find "$run_dir" -maxdepth 1 -type f \
        ! -name CHECKPOINT.json ! -name STATIC_CHECKPOINT.json \
        -exec cp -u {} "$ARCHIVE/$name/" \; 2>>/tmp/publish_err || failed=1
    local manifest
    for manifest in CHECKPOINT.json STATIC_CHECKPOINT.json; do
      if [ "$failed" -eq 0 ] && [ -f "$run_dir/$manifest" ]; then
        cp -u "$run_dir/$manifest" "$ARCHIVE/$name/" 2>>/tmp/publish_err || failed=1
      fi
    done

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
  elif [ "${RUN_OP:-train}" = "train" ] && [ -f "$src/STATIC_CHECKPOINT.json" ]; then
    # CONTINUING A STATIC RUN NEEDS EXACTLY ONE RUNG: the manifest's current
    # snapshot. Falling to the catch-all below took the whole retained ladder --
    # 31 rungs, ~25 GB over SMB, ~40 minutes -- to load the 809 MB the trainer
    # actually reads.
    #
    # Leaving the older rungs on the share is not a loss. `_extend_ladder` builds
    # the next manifest from the PREVIOUS manifest, not from what is on disk, so
    # the ladder's history survives; `_prune` only deletes what the manifest does
    # not name, and absent rungs are simply absent; and publish copies per
    # directory, so rungs this node never had are neither re-uploaded nor removed.
    cur=$(python3 -c "
import json, sys
print(json.load(open(sys.argv[1]))['zarr'])
" "$src/STATIC_CHECKPOINT.json" || true)
    find "$src" -maxdepth 1 -mindepth 1 \
         ! -name 'static-*' ! -name STATIC_CHECKPOINT.json \
         -exec cp -ru {} "$RUNS/$RUN_ID/" \; 2>/dev/null || true
    if [ -z "$cur" ]; then
      log "FATAL STATIC_CHECKPOINT.json on the share names no current snapshot"
      exit 1
    elif [ ! -d "$src/$cur" ]; then
      log "FATAL manifest names $cur but it is not on the share"
      exit 1
    elif [ ! -f "$src/.complete-$cur" ]; then
      # Same reasoning as the evaluate branch: an unmarked rung is either
      # pre-marker or interrupted, and resuming from a truncated one would train
      # on garbage rather than fail. `repair-ladder` is how a rung earns a marker.
      log "FATAL $cur has no completion marker -- refusing to resume from a"
      log "      possibly-partial snapshot. Run repair-ladder on this run first."
      exit 1
    else
      # rm first and NO -u, for the reason the evaluate branch documents: a
      # cancelled task's partial rung on the node would be skipped as present.
      rm -rf "$RUNS/$RUN_ID/$cur"
      cp -r "$src/$cur" "$RUNS/$RUN_ID/" 2>/tmp/fetch_err || {
        log "FATAL fetching $cur failed: $(tail -1 /tmp/fetch_err 2>/dev/null)"; exit 1
      }
      cp -u "$src/STATIC_CHECKPOINT.json" "$RUNS/$RUN_ID/" 2>/dev/null || true
      log "fetched current rung $cur (ladder left on the share)"
    fi
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
# "Empty means all CPUs" was documented but never implemented: `train-static`
# defaults --workers to 1, so an omitted worker count trained SINGLE-THREADED on
# a 16-vCPU node -- a ~16x throughput loss that looks like a slow leg rather than
# a misconfiguration, and that turns a 1.8h leg into one the 6h ceiling kills.
# The node is the only place that knows its own core count, so it fills the
# default in rather than leaving the CLI's local-friendly 1 to stand.
# `|| echo 1` because a bare failing $(nproc) under `set -e` would abort the leg
# outright -- a missing core-count utility must degrade, not kill the run.
ARGS+=(--workers "${RUN_WORKERS:-$(nproc 2>/dev/null || echo 1)}")
[ -n "${RUN_EXPERIMENT:-}" ] && ARGS+=(--experiment "$RUN_EXPERIMENT")
[ -n "${RUN_ARM:-}" ] && ARGS+=(--arm "$RUN_ARM")
[ -n "${RUN_PARENT:-}" ] && ARGS+=(--parent "$RUN_PARENT")
# A JSON ARRAY, not a space-separated string. The submitter passes environment
# settings as typed name/value pairs, so the old hex encoding (which existed
# only because the `az` CLI parses `--environment-settings` as KEY=VALUE, and a
# config override's value contains `=`) is gone. JSON stays because these are
# LISTS whose elements may contain spaces: the previous `for kv in $RUN_SETS`
# split on whitespace, silently cutting any such value in half.
#
# NUL-separated, read with `read -d ''`, because a newline inside a value would
# defeat line-splitting the same way a space defeated word-splitting. `read -d`
# rather than `mapfile -d`: the latter needs bash 4.4+, and a node-side script
# should not carry a bash-version floor it does not need.
# Decoded to a FILE first, and a decode failure is FATAL. Reading straight from
# a process substitution would hide the failure: `set -e` does not see into it,
# so a malformed payload would yield zero overrides and the leg would train with
# the BASE config -- an experiment arm silently running as its own control, and
# recorded that way in .run.json. That exact class of silent rebucketing has
# already cost one curve.
if [ -n "${RUN_SETS_JSON:-}" ]; then
  SETS_FILE="$WORK/sets-${AZ_BATCH_TASK_ID:-local}.nul"
  python3 -c '
import json, sys
for item in json.loads(sys.argv[1]):
    sys.stdout.write(item + "\0")
' "$RUN_SETS_JSON" > "$SETS_FILE" || {
    log "FATAL could not decode RUN_SETS_JSON: $RUN_SETS_JSON"
    exit 1
  }
  n_sets=0
  while IFS= read -r -d '' kv; do
    if [ -n "$kv" ]; then
      ARGS+=(--set "$kv")
      n_sets=$((n_sets + 1))
      log "  override: $kv"
    fi
  done < "$SETS_FILE"
  # Count the OVERRIDES, not ${#ARGS[@]} -- ARGS already holds the --workers /
  # --experiment / --arm / --parent pairs, so a tagged leg with zero overrides
  # would report eight. The one diagnostic this block exists to emit has to be
  # the one number that can be trusted.
  log "overrides: $n_sets from RUN_SETS_JSON"
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
# PRECOMPUTE mode. Builds a card abstraction on a node and publishes it once.
#
# The output path needs no special handling: `$CODE/data` is a symlink to
# $DATA (set above), and `precompute` writes to <base>/data/combo_abstraction/
# <name>, so it lands on the node's data disk exactly where a training leg
# would look for it.
#
# THE GUARD IS THE POINT. The invariant is "computed ONCE, never recomputed" --
# not "computed locally", which is what the local-only workflow confused it
# with. Bucket ASSIGNMENT is not pinned by card_abstraction_hash, so
# republishing under the same name can silently change which bucket a hand
# lands in, and every run trained against the old copy keeps a provenance check
# that now passes over different buckets. Refusing to overwrite is what makes
# running this in the cloud as safe as running it on a laptop.
if [ "${RUN_OP:-train}" = "precompute" ]; then
  [ -n "${RUN_CONFIG:-}" ] || { log "FATAL precompute needs RUN_CONFIG"; exit 1; }
  log "precompute: config=$RUN_CONFIG (timeout $RUN_TIMEOUT)"
  set +o pipefail
  "${GUARD[@]}" uv run poker-solver-run precompute --config "$RUN_CONFIG" --json \
    > "$WORK/precompute.json" 2> >(tee -a "$LEG_LOG" >&2)
  rc=${PIPESTATUS[0]}
  set -o pipefail
  if [ "$rc" != 0 ]; then
    log "precompute failed rc=$rc"; publish_log; exit "$rc"
  fi

  # The command reports where it wrote; do not re-derive the directory name.
  OUT_DIR=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["output_dir"])' \
    "$WORK/precompute.json")
  NAME=$(basename "$OUT_DIR")
  DEST="$SHARE/combo_abstraction/$NAME"
  log "precomputed $NAME -> $OUT_DIR"

  if [ -d "$DEST" ] && [ -z "${RUN_FORCE_PUBLISH:-}" ]; then
    log "REFUSING to republish: $NAME already exists on the share."
    log "  Bucket assignment is not pinned by the abstraction hash, so replacing"
    log "  it would silently invalidate every run trained against the old copy."
    log "  Set RUN_FORCE_PUBLISH=1 only if no such run matters."
    publish_log
    exit 1
  fi

  mkdir -p "$DEST"
  cp -ru "$OUT_DIR/." "$DEST/" || { log "FATAL publish failed"; publish_log; exit 1; }
  log "published $NAME to the share ($(du -sh "$OUT_DIR" | cut -f1))"
  publish_log
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
  # A JSON array, decoded exactly like RUN_SETS_JSON above: an eval flag's
  # value can contain a space, which the old word-split form could not carry.
  # Same file-first decode as RUN_SETS_JSON, and fatal for the same reason:
  # scoring with silently-dropped flags produces a number measured by a
  # different instrument than the one asked for, which is worse than no number.
  EXTRA=()
  if [ -n "${RUN_EVAL_FLAGS_JSON:-}" ]; then
    FLAGS_FILE="$WORK/evalflags-${AZ_BATCH_TASK_ID:-local}.nul"
    python3 -c '
import json, sys
for item in json.loads(sys.argv[1]):
    sys.stdout.write(item + "\0")
' "$RUN_EVAL_FLAGS_JSON" > "$FLAGS_FILE" || {
      log "FATAL could not decode RUN_EVAL_FLAGS_JSON: $RUN_EVAL_FLAGS_JSON"
      exit 1
    }
    while IFS= read -r -d '' flag; do
      [ -n "$flag" ] && EXTRA+=("$flag")
    done < "$FLAGS_FILE"
  fi
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
