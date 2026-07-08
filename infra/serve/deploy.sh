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
# EVERYTHING LANDS ON LOCAL DISK, never read from the share at runtime. A
# checkpoint is ~5,500 small files that the read path mmaps, and SMB turns every
# page fault into a network round trip -- the same constraint that keeps
# `runs_dir` off the share everywhere else. The copies survive a deallocate
# because /mnt/work is a managed disk, which is what makes waking the box a
# two-minute boot rather than a re-download.
#
# Idempotent: `cp -u` skips what is already current, so re-running after a
# checkpoint is published copies only the new rungs.

set -euo pipefail

RUN="${1:-}"
if [ -z "$RUN" ]; then
    echo "usage: deploy.sh <run-id>" >&2
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
# $CODE pins a snapshot; without it, the newest. Names sort lexicographically by
# timestamp, which is what makes `tail -1` the newest rather than merely last.
#
# PIN IT when it matters. The share is shared: three other sessions pushed
# snapshots within 15 seconds of one here, so "newest" deployed somebody else's
# tree and the box came up without the package this deploy existed to ship.
# `push-code` echoes the id to pass back in.
if [ -n "${CODE:-}" ]; then
    snapshot="$SHARE/code/${CODE%.tar.gz}.tar.gz"
    if [ ! -f "$snapshot" ]; then
        echo "No snapshot '$CODE' on the share." >&2
        exit 1
    fi
else
    snapshot=$(find "$SHARE/code" -maxdepth 1 -name '*.tar.gz' | sort | tail -1)
    if [ -z "$snapshot" ]; then
        echo "No code snapshot on the share. Run: poker-solver push-code" >&2
        exit 1
    fi
fi
echo "==> code $(basename "$snapshot")"

# Extracted beside the live tree and swapped in, so a failed or interrupted
# extraction never leaves a half-written checkout that `uv sync` would then
# build against.
rm -rf "$WORK/code.incoming"
mkdir -p "$WORK/code.incoming"
tar -xzf "$snapshot" -C "$WORK/code.incoming"
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
echo "==> card abstraction"
cp -ru "$SHARE/combo_abstraction/." "$WORK/data/combo_abstraction/"

# --------------------------------------------------------------------------- #
# the checkpoint
# --------------------------------------------------------------------------- #
# ONE RUNG, not the run directory -- the same rule `blueprint/staging.py` states
# and this script used to ignore. A published run keeps its whole retained
# ladder: the 300M control run is 60 rungs, ~51 GB in ~330,000 files, about six
# hours at this box's file rate. The reader loads exactly one of them. Copying
# the lot bought nothing and made staging a run an afternoon's job.
#
# $AT names a rung to stage instead of the head, mirroring `--at`.
echo "==> checkpoint (thousands of small files, a few minutes on first copy)"
mkdir -p "$WORK/data/runs/$RUN_ID"
# The share serves the archive read-only (444 files, 555 directories) and `cp`
# carries that mode across, so the SECOND deploy of a run cannot overwrite what
# the first one staged: `cp -u` fails EACCES on a file this user owns, and the
# script's `set -e` turns that into an aborted deploy. Restore write on our own
# copy first -- it is a cache, not a source of truth, and nothing reads its mode.
chmod -R u+w "$WORK/data/runs/$RUN_ID" 2>/dev/null || true
cp -u "$SHARE/archive/$RUN_ID/STATIC_CHECKPOINT.json" "$WORK/data/runs/$RUN_ID/"
cp -ru "$SHARE/archive/$RUN_ID/evals" "$WORK/data/runs/$RUN_ID/" 2>/dev/null || true
# Every small file beside the manifest: `run.jsonl` OR the older `.run.json`
# (`RunTracker.load` takes either, and a legacy run has only the latter), plus
# `progress.jsonl`. Copying just `run.jsonl` staged a legacy run "successfully"
# and then failed at load with `No run record`.
for small in run.jsonl .run.json progress.jsonl; do
    cp -u "$SHARE/archive/$RUN_ID/$small" "$WORK/data/runs/$RUN_ID/" 2>/dev/null || true
done

# The ladder key is `retained` (`static_checkpoint.py` writes it; `staging.py`
# and `archive.py` read it). `checkpoints` is not a key, so $AT always missed and
# `set -e` turned that into an aborted deploy.
rung=$(AT="${AT:-}" python3 - "$SHARE/archive/$RUN_ID/STATIC_CHECKPOINT.json" <<'PY'
import json, os, sys
manifest = json.load(open(sys.argv[1]))
rungs = manifest.get("retained") or []
at = os.environ.get("AT") or ""
if at:
    match = [r for r in rungs if str(r.get("iteration")) == at]
    if not match:
        sys.exit(f"no rung at iteration {at}; have {[r.get('iteration') for r in rungs]}")
    print(match[0]["zarr"])
    raise SystemExit
head = manifest.get("zarr") or (rungs[-1]["zarr"] if rungs else "")
if not head:
    sys.exit("manifest names no head checkpoint")
print(head)
PY
)
echo "==> rung $rung"
cp -ru "$SHARE/archive/$RUN_ID/$rung" "$WORK/data/runs/$RUN_ID/"
# The writer's own completion sentinel. `staging._complete` requires it, so
# without it the console's next `stage_run` for this run decides nothing is here
# and re-copies the whole ~850 MB rung over SMB.
cp -u "$SHARE/archive/$RUN_ID/.complete-$rung" "$WORK/data/runs/$RUN_ID/" 2>/dev/null || true

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
EOF

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
# SEAT_EXTRA carries $AT through to the seat, which `blueprint-serve` has no flag
# for. The two want different things from one staged run: the reader takes the
# manifest head, and the seat should field the rung a NUMBER was measured at.
# Empty when $AT is unset, which the unit expands to no arguments.
sudo tee /etc/chipzen-seat.env >/dev/null <<EOF
RUN=$RUN_ID
RUNS_DIR=$WORK/data/runs
CHIPZEN_ENV=${CHIPZEN_ENV:-prod}
POLICY_THRESHOLD=${POLICY_THRESHOLD:-0.02}
SEAT_EXTRA=${AT:+--at $AT}
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
