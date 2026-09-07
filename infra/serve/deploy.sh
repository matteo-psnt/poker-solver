#!/usr/bin/env bash
# Put code, the card abstraction and one run onto the blueprint host, then point
# the service at it.
#
# RUNS ON THE BOX, PIPED IN OVER SSH -- `just serve-deploy <run>` sends this file
# to `bash -s`. Deliberately not baked into cloud-init: `custom_data` carries
# `ignore_changes`, so anything living there can only be changed by replacing the
# machine, and this is the part that will be iterated on. Piping it means the box
# always runs the version in the repo and stores none of it.
#
# EVERYTHING LANDS ON LOCAL DISK, never read from the store at runtime. The
# copies survive a deallocate because /mnt/work is a managed disk, which is what
# makes waking the box a two-minute boot rather than a re-download.
#
# THE SHARE NO LONGER HOLDS CHECKPOINT BYTES. Measured 09-07: across the whole
# archive there are zero `.ckpt.zst` objects and zero `.zarr` directories -- a
# run directory is now its manifest, its loose result files and a completion
# marker per rung. The bytes are one Blob object per rung in the `checkpoints`
# container. So the record comes off the share and the RUNG comes off Blob, and
# there is deliberately no share fallback: a fallback that cannot succeed turns
# "this rung is not published" into 400 MB copied and a failure later inside the
# loader.
#
# Idempotent by SIZE: a rung is re-fetched only when what is on disk does not
# match what the container holds. Node-local state is not evidence of a complete
# copy, and a truncated snapshot fails deep in the reader minutes later.

set -euo pipefail

RUN="${1:-}"
RECORD_DSN="${2:-}"
if [ -z "$RUN" ] || [ -z "$RECORD_DSN" ]; then
    echo "usage: deploy.sh <run-id> <record-dsn>   (just serve-deploy <run> supplies both)" >&2
    exit 2
fi

SHARE=/mnt/shared
WORK=/mnt/work
IDLE="${IDLE_TIMEOUT:-1800}"

if ! mountpoint -q "$SHARE"; then
    echo "$SHARE is not mounted -- the box cannot see the store." >&2
    exit 1
fi

# --------------------------------------------------------------------------- #
# the run, resolved first
# --------------------------------------------------------------------------- #
# Before anything is copied: a typo'd fragment should cost a message, not a
# 773 MB abstraction sync followed by a message. Matched as a FRAGMENT the way
# every reader command does, since run ids differ only at the tail.
matches=$(find "$SHARE/archive" -maxdepth 1 -type d -name "*${RUN}*" -printf '%f\n' 2>/dev/null || true)
count=$(printf '%s' "$matches" | grep -c . || true)

if [ "$count" -eq 0 ]; then
    echo "No published run matching '$RUN'. Try: poker-solver runs" >&2
    exit 1
fi
if [ "$count" -gt 1 ]; then
    echo "'$RUN' matches more than one run:" >&2
    printf '  %s\n' $matches >&2
    exit 1
fi
RUN_ID="$matches"
echo "==> run $RUN_ID"

# --------------------------------------------------------------------------- #
# code
# --------------------------------------------------------------------------- #
# Snapshots live in the store's `code` blob container, not on the share. The
# account is read off the mount so nothing here hard-codes it, and the box
# reads Blob as ITSELF -- the managed identity it already logs in with to
# deallocate. (It needs Storage Blob Data Reader on the account for this.)
STORE_ACCOUNT=$(findmnt -n -o SOURCE "$SHARE" | sed -E 's#^//([^.]+)\..*#\1#')
az login --identity --output none

# $CODE pins a snapshot; without it, the newest. Names sort lexicographically by
# timestamp, which is what makes the last one the newest rather than merely last.
#
# PIN IT when it matters. The store is shared: three other sessions pushed
# snapshots within 15 seconds of one here, so "newest" deployed somebody else's
# tree and the box came up without the package this deploy existed to ship.
# `push-code` echoes the id to pass back in.
if [ -n "${CODE:-}" ]; then
    snapshot="${CODE%.tar.gz}.tar.gz"
    exists=$(az storage blob exists --auth-mode login --account-name "$STORE_ACCOUNT" \
        --container-name code --name "$snapshot" --query exists -o tsv)
    if [ "$exists" != "true" ]; then
        echo "No snapshot '$CODE' in the store." >&2
        exit 1
    fi
else
    snapshot=$(az storage blob list --auth-mode login --account-name "$STORE_ACCOUNT" \
        --container-name code --query "sort_by([].name, &name)[-1]" -o tsv)
    if [ -z "$snapshot" ]; then
        echo "No code snapshot in the store. Run: poker-solver push-code" >&2
        exit 1
    fi
fi
echo "==> code $snapshot"

# Extracted beside the live tree and swapped in, so a failed or interrupted
# extraction never leaves a half-written checkout that `uv sync` would then
# build against.
rm -rf "$WORK/code.incoming"
mkdir -p "$WORK/code.incoming"
az storage blob download --auth-mode login --account-name "$STORE_ACCOUNT" \
    --container-name code --name "$snapshot" --file "$WORK/code.incoming.tar.gz" --output none
tar -xzf "$WORK/code.incoming.tar.gz" -C "$WORK/code.incoming"
rm -f "$WORK/code.incoming.tar.gz"
rm -rf "$WORK/code.previous"
[ -d "$WORK/code" ] && mv "$WORK/code" "$WORK/code.previous"
mv "$WORK/code.incoming" "$WORK/code"

# The abstraction resolver scans `<cwd>/data/combo_abstraction`, and the service
# runs with WorkingDirectory=/mnt/work/code -- so `data` has to be there. A
# symlink rather than a copy: the snapshot deliberately excludes `data`, and the
# artifacts are far too big to live inside a tree that gets replaced.
mkdir -p "$WORK/data/combo_abstraction" "$WORK/data/runs"
ln -sfn "$WORK/data" "$WORK/code/data"

# --------------------------------------------------------------------------- #
# the card abstraction
# --------------------------------------------------------------------------- #
# Abstractions moved off the share too: one `<name>.tar.zst` object each in the
# `abstractions` container. MERGES rather than mirrors, exactly like the node's
# `fetch_abstractions` -- which abstraction a run resolves against is not known
# until its config is read, so all of them have to be here. An abstraction
# already unpacked is not re-fetched, so the steady state is one list call.
echo "==> card abstraction"
mkdir -p "$WORK/data/combo_abstraction"
for packed in $(az storage blob list --auth-mode login --account-name "$STORE_ACCOUNT" \
    --container-name abstractions --query "[].name" -o tsv); do
    case "$packed" in *.tar.zst) ;; *) continue ;; esac
    unpacked="$WORK/data/combo_abstraction/${packed%.tar.zst}"
    [ -d "$unpacked" ] && continue
    echo "    fetching abstraction ${packed%.tar.zst}"
    az storage blob download --auth-mode login --account-name "$STORE_ACCOUNT" \
        --container-name abstractions --name "$packed" \
        --file "$WORK/data/combo_abstraction/$packed" --output none
    # Drop the archive once unpacked: a packed file left beside the directories
    # is hundreds of MB the resolver will never read. On a tar failure `set -e`
    # aborts and it stays, which the next download simply overwrites.
    tar --zstd -xf "$WORK/data/combo_abstraction/$packed" -C "$WORK/data/combo_abstraction"
    rm -f "$WORK/data/combo_abstraction/$packed"
done

# --------------------------------------------------------------------------- #
# the checkpoint
# --------------------------------------------------------------------------- #
# ONE RUNG, not the run directory -- the same rule `blueprint/staging.py` states
# and this script used to ignore. A published run keeps its whole retained
# ladder and the reader loads exactly one of them, so fetching the lot bought
# nothing and made staging a run an afternoon's job. One object per rung has
# made that cheaper, not moot: the 300M control run is 60 rungs, and the eight
# rungs this ladder actually seats are 1.8 GB against that.
#
# $AT names a rung to stage instead of the head, mirroring `--at`.
# Staging is a FUNCTION because the depth ladder stages several runs the same
# way. Every rule below was learned the hard way once; a second hand-rolled copy
# of it would relearn them.
stage_run() {
    local run="$1" at="$2" dest="$WORK/data/runs/$1"
    mkdir -p "$dest"
    chmod -R u+w "$dest" 2>/dev/null || true
    cp -u "$SHARE/archive/$run/STATIC_CHECKPOINT.json" "$dest/"
    cp -ru "$SHARE/archive/$run/evals" "$dest/" 2>/dev/null || true
    for small in run.jsonl .run.json progress.jsonl; do
        cp -u "$SHARE/archive/$run/$small" "$dest/" 2>/dev/null || true
    done

    # THE NAME IS MAPPED, NOT BUILT. A manifest still spells `static-N.zarr` and
    # is never repointed -- rewriting 300+ of them would mutate the durable
    # share -- while the container holds `static-N.ckpt.zst`. `records.object_name`
    # is the single place that maps between the two, and hand-building the
    # spelling instead is the bug that has already hit evaluation fetch,
    # warm-start, dispatch verification and prune. Imported from the code just
    # extracted above: `src.shared.records` is stdlib-only, so the system
    # interpreter can read it without the venv.
    local object
    object=$(AT="$at" PYTHONPATH="$WORK/code" python3 - \
        "$SHARE/archive/$run/STATIC_CHECKPOINT.json" <<'PY'
import json, os, sys

from src.shared import records

manifest = json.load(open(sys.argv[1]))
rungs = manifest.get("retained") or []
at = os.environ.get("AT") or ""
if at:
    match = [r for r in rungs if str(r.get("iteration")) == at]
    if not match:
        sys.exit(f"no rung at iteration {at}; have {[r.get('iteration') for r in rungs]}")
    named = match[0]["zarr"]
else:
    named = manifest.get("zarr") or (rungs[-1]["zarr"] if rungs else "")
    if not named:
        sys.exit("manifest names no head checkpoint")
print(records.object_name(named))
PY
    )

    # A MANIFEST OVER-CLAIMS BY DESIGN: `prune-checkpoints` drops a snapshot
    # without rewriting the ladder that advertises it, so being named here is no
    # evidence the bytes exist. Ask the container, which is the only store that
    # holds them.
    local held
    held=$(az storage blob show --auth-mode login --account-name "$STORE_ACCOUNT" \
        --container-name checkpoints --name "$run/$object" \
        --query properties.contentLength -o tsv 2>/dev/null || true)
    if [ -z "$held" ]; then
        echo "Rung $object of $run is not in the checkpoints container." >&2
        echo "  The share holds no checkpoint bytes any more, so there is nowhere" >&2
        echo "  else to look. Re-publish it from a node that has it." >&2
        return 1
    fi

    # BOTH SPELLINGS, not merely the one being fetched. A box that staged under
    # the old name can still be holding a `.zarr` directory, and the reader can
    # no longer open one at all -- the zarr path is deleted. Leaving it behind
    # wastes the disk it is on and confuses the next person to look.
    local legacy="${object%.ckpt.zst}.zarr"
    rm -rf "$dest/$legacy" "$dest/.complete-$legacy"

    local have
    have=$(stat -c %s "$dest/$object" 2>/dev/null || echo 0)
    if [ "$have" = "$held" ]; then
        echo "==> $run rung $object already staged ($((held / 1000000)) MB)"
    else
        echo "==> staging $run rung $object ($((held / 1000000)) MB from the container)"
        # To a temp name and moved into place: a half-downloaded object under the
        # real name is something the seat would try to load on the next restart.
        az storage blob download --auth-mode login --account-name "$STORE_ACCOUNT" \
            --container-name checkpoints --name "$run/$object" \
            --file "$dest/$object.partial" --output none
        mv "$dest/$object.partial" "$dest/$object"
    fi
    chmod -R u+w "$dest" 2>/dev/null || true
}

echo "==> checkpoints (one object per rung, from the container)"
mkdir -p "$WORK/data/runs/$RUN_ID"
stage_run "$RUN_ID" "${AT:-}"

# TWO CONSUMERS, TWO RUNGS. The seat pins the rung a number was measured at;
# the reader's `ExecStart` is baked into cloud-init with no `--at`, so it opens
# whatever the manifest calls the head. Staging only $AT left the reader looking
# for `static-4200.ckpt.zst` that no deploy had fetched, and `blueprint.service`
# died on FileNotFoundError while the seat itself was fine.
if [ -n "${AT:-}" ]; then
    stage_run "$RUN_ID" ""
fi

# $RUNGS is a comma-separated `run[:at]` list: the SHALLOWER blueprints of the
# depth ladder. Each is staged exactly like the deepest, and each becomes a
# `--rung` on the seat's command line.
SEAT_RUNGS=""
if [ -n "${RUNGS:-}" ]; then
    IFS=',' read -ra _rungs <<< "$RUNGS"
    for spec in "${_rungs[@]}"; do
        [ -n "$spec" ] || continue
        # `run[:at[:threshold]]`. Split on ALL THREE fields: `${spec##*:}` took
        # the LAST one, so a spec carrying a threshold staged "0.20" as the
        # iteration, the manifest lookup failed, and `set -e` aborted the deploy
        # after the env file had already been left alone -- which looks exactly
        # like a deploy that worked and changed nothing.
        IFS=':' read -r _name _at _thr <<< "$spec"
        stage_run "$_name" "${_at:-}"
        SEAT_RUNGS="$SEAT_RUNGS --rung $spec"
    done
fi

# --------------------------------------------------------------------------- #
# dependencies
# --------------------------------------------------------------------------- #
echo "==> uv sync"
cd "$WORK/code"
# `--extra chipzen` because this box is also where `chipzen-seat` holds its
# WebSocket, and the SDK is an optional extra that a bare `--no-dev` skips --
# the seat then refuses at import with "the Chipzen SDK is not installed".
# Harmless when nothing seats: one small pure-Python dependency.
"$HOME/.local/bin/uv" sync --no-dev --extra chipzen

# --------------------------------------------------------------------------- #
# point the service at it
# --------------------------------------------------------------------------- #
# Rewritten whole rather than patched line by line, so the file cannot drift into
# a shape the unit reads differently from what is here.
sudo tee /etc/blueprint.env >/dev/null <<EOF
RUN=$RUN_ID
RUNS_DIR=$WORK/data/runs
IDLE_TIMEOUT=$IDLE
POKER_SOLVER_RECORD_DSN=$RECORD_DSN
EOF
# The DSN carries a password; the unit runs as root and is the only reader.
sudo chmod 600 /etc/blueprint.env

# The shutdown half of the unit, rewritten on every deploy.
#
# It ships in cloud-init `write_files`, which runs ONCE at first boot -- so
# before this block, the only way to correct it was to recreate the box. That is
# how a box spent 62 hours idling out every 30 minutes and restarting itself:
# the bug was one line in a file no deploy could reach.
#
# Only the two pieces that encode the shutdown contract are written here. The
# rest of the unit is first-boot territory and does not change.
sudo install -m 0755 "$WORK/code/infra/serve/box-may-sleep" /usr/local/bin/
sudo tee /usr/local/bin/deallocate-if-idle >/dev/null <<'EOF'
#!/bin/bash
# 42 is IDLE_EXIT_CODE: nobody was here, switch the box off. NOT 0 (a deliberate
# stop) and NOT 143 (SIGTERM -- `systemctl stop`, and the restart below).
if [ "${EXIT_STATUS:-1}" != "42" ]; then
  echo "blueprint exited ${EXIT_STATUS} -- not deallocating"
  exit 0
fi
/usr/local/bin/box-may-sleep || exit 0
exec /usr/local/bin/deallocate-box
EOF
sudo chmod 0755 /usr/local/bin/deallocate-if-idle

# The OTHER route to the same deallocate. `OnFailure=blueprint-deallocate` runs
# `deallocate-box` directly and has never consulted `deallocate-if-idle` -- this
# file says so eleven lines up, about a bug that switched the box off after every
# deploy. An ExecCondition rather than a wrapper: a refused condition SKIPS the
# unit instead of failing it, so a seated box does not accumulate failed units.
sudo mkdir -p /etc/systemd/system/blueprint-deallocate.service.d
sudo tee /etc/systemd/system/blueprint-deallocate.service.d/seat-guard.conf >/dev/null <<'EOF'
[Service]
ExecCondition=/usr/local/bin/box-may-sleep
EOF

# `SuccessExitStatus=42` as a drop-in, so the idle exit is not read as a failure
# and restarted before the deallocate lands. A drop-in rather than a rewrite of
# the unit: everything else in it is first-boot configuration this script has no
# business restating.
#
# 143 is in there for a bug this script caused to ITSELF. 143 is SIGTERM, which
# is what `systemctl restart` sends -- the restart four lines below. Without it
# systemd read the deploy's own restart as a failure and fired
# `OnFailure=blueprint-deallocate`, so every deploy switched the box off a
# minute after reporting success. `deallocate-if-idle` already refused that exit
# ("blueprint exited 143 -- not deallocating"), but `OnFailure` is a SECOND and
# independent path to the same deallocate and never consulted it.
sudo mkdir -p /etc/systemd/system/blueprint.service.d
sudo tee /etc/systemd/system/blueprint.service.d/idle-exit.conf >/dev/null <<'EOF'
[Service]
SuccessExitStatus=42 143
EOF

# --------------------------------------------------------------------------- #
# the chipzen seat
# --------------------------------------------------------------------------- #
# INSTALLED, never enabled or started here. Seating puts a bot on a live public
# ladder, and `serve-deploy` is run to stage a checkpoint -- those are different
# decisions and a deploy must not silently make the second one. `just seat-on`
# is the deliberate act; `enable` is what carries it across a reboot.
#
# These come out of the extracted snapshot rather than a heredoc, because this
# script reaches the box down a pipe with no siblings, but the tree it just
# unpacked has them.
units="$WORK/code/infra/serve"
sudo install -m 0644 "$units/chipzen-seat.service" /etc/systemd/system/
sudo install -m 0644 "$units/chipzen-seat-watchdog.service" /etc/systemd/system/
sudo install -m 0644 "$units/chipzen-seat-watchdog.timer" /etc/systemd/system/
sudo install -m 0755 "$units/chipzen-seat-watchdog" /usr/local/bin/

# POLICY_THRESHOLD must never be empty here: the unit expands it unquoted into
# argparse. 0.02 is the measured point (940.1 -> 854.0 mbb/hand on the gate over
# three seeds); 0 fields the table as trained.
#
# SEAT_EXTRA carries $AT through to the seat. `blueprint-serve` grew an `--at`
# of its own, but the READER'S unit is baked into cloud-init without one, so in
# practice the two still want different rungs from one staged run: the reader
# takes the manifest head, the seat fields the rung a NUMBER was measured at.
# That is why both are staged above. Empty when $AT is unset, which the unit
# expands to no arguments.
sudo tee /etc/chipzen-seat.env >/dev/null <<EOF
RUN=$RUN_ID
RUNS_DIR=$WORK/data/runs
CHIPZEN_ENV=${CHIPZEN_ENV:-prod}
POLICY_THRESHOLD=${POLICY_THRESHOLD:-0.02}
SEAT_EXTRA=${AT:+--at $AT}$SEAT_RUNGS
EOF

sudo systemctl daemon-reload
sudo systemctl enable blueprint
sudo systemctl restart blueprint

# A running seat imported the code it started with, so after the tree underneath
# it is replaced its behaviour no longer matches what is on disk -- and checking
# the disk is how you would try to find that out. Only when already seated:
# `is-enabled` is the record of the human decision above.
if systemctl is-enabled --quiet chipzen-seat 2>/dev/null; then
    echo "==> restarting the seat onto the new code"
    sudo systemctl restart chipzen-seat chipzen-seat-watchdog.timer
fi

echo "==> waiting for it to load (a production run takes ~1 min)"
for _ in $(seq 1 90); do
    if curl -fsS --max-time 2 http://127.0.0.1:8790/api/health >/dev/null 2>&1; then
        echo
        curl -fsS http://127.0.0.1:8790/api/health
        echo
        echo "==> serving $RUN_ID"
        exit 0
    fi
    if ! systemctl is-active --quiet blueprint; then
        echo "blueprint.service is not running:" >&2
        sudo journalctl -u blueprint -n 40 --no-pager >&2
        exit 1
    fi
    sleep 2
done

echo "Timed out waiting for the server. Recent log:" >&2
sudo journalctl -u blueprint -n 40 --no-pager >&2
exit 1
