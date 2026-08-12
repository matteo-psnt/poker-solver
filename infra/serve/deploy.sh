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
# The newest snapshot. Snapshot names sort lexicographically by timestamp, which
# is what makes `tail -1` the newest rather than merely the last listed.
snapshot=$(find "$SHARE/code" -maxdepth 1 -name '*.tar.gz' | sort | tail -1)
if [ -z "$snapshot" ]; then
    echo "No code snapshot on the share. Run: poker-solver push-code" >&2
    exit 1
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
echo "==> checkpoint (thousands of small files, a few minutes on first copy)"
mkdir -p "$WORK/data/runs/$RUN_ID"
cp -ru "$SHARE/archive/$RUN_ID/." "$WORK/data/runs/$RUN_ID/"

# --------------------------------------------------------------------------- #
# dependencies
# --------------------------------------------------------------------------- #
echo "==> uv sync"
cd "$WORK/code"
"$HOME/.local/bin/uv" sync --no-dev

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
sudo tee /usr/local/bin/deallocate-if-idle >/dev/null <<'EOF'
#!/bin/bash
# 42 is IDLE_EXIT_CODE: nobody was here, switch the box off. NOT 0 (a deliberate
# stop) and NOT 143 (SIGTERM -- `systemctl stop`, and the restart below).
if [ "${EXIT_STATUS:-1}" != "42" ]; then
  echo "blueprint exited ${EXIT_STATUS} -- not deallocating"
  exit 0
fi
exec /usr/local/bin/deallocate-box
EOF
sudo chmod 0755 /usr/local/bin/deallocate-if-idle

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

sudo systemctl daemon-reload
sudo systemctl enable blueprint
sudo systemctl restart blueprint

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
