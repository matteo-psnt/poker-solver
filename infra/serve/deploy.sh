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
# THE SHARE IS GONE FROM THIS SCRIPT. Measured 09-10: `/mnt/shared` is mounted
# and completely EMPTY -- zero archive entries, no abstraction directory. A run
# is now its manifest, its loose metadata and one object per rung, all under
# `<run>/` in the `checkpoints` container, with the record in Postgres.
#
# There is deliberately no share fallback anywhere below. A fallback that cannot
# succeed turns "this is not published" into bytes copied and a failure later
# inside the loader, and an empty listing reads as "nothing there" rather than
# as "wrong store" -- which is how every share reader answered confidently and
# wrongly when the share emptied underneath them.
#
# Idempotent by SIZE: a rung is re-fetched only when what is on disk does not
# match what the container holds. Node-local state is not evidence of a complete
# copy, and a truncated snapshot fails deep in the reader minutes later.
#
# DO NOT DEPLOY BETWEEN 00:00 AND 05:00 UTC. Measured 09-14 off the seat journal:
# chipzen allows 20 matches per HOUR and the queue only opens for about five
# hours a day, so 00:00-05:00 is the entire day's ~100 matches. Deploying restarts
# the seat, and each restart re-joins the queue. On 09-14 three redeploys at
# 01:21-01:34 sit exactly inside a day that produced 20 matches where the seven
# days before it produced ~101 each. Not proven to be the cause -- but the window
# is the only one there is, and it costs nothing to deploy after 05:00 UTC.

set -euo pipefail

RUN="${1:-}"
RECORD_DSN="${2:-}"
STORE_ACCOUNT="${3:-}"
if [ -z "$RUN" ] || [ -z "$RECORD_DSN" ] || [ -z "$STORE_ACCOUNT" ]; then
    echo "usage: deploy.sh <run-id> <record-dsn> <storage-account>" >&2
    echo "       (just serve-deploy <run> supplies all three)" >&2
    exit 2
fi

WORK=/mnt/work

# NO SHARE. It is empty and nothing writes to it: a run is its manifest, its
# loose metadata and one object per rung, all under `<run>/` in the
# `checkpoints` container, with the record in Postgres. The account name is an
# ARGUMENT rather than read off the mount with `findmnt` -- the mount is being
# removed, and deriving a store's name from a filesystem that is going away is
# the same mistake as reading the run from it.
#
# The box reads Blob as ITSELF, its managed identity, so nothing here needs an
# account key. (It needs Storage Blob Data Reader on the account, which is now the
# only thing that identity is for -- nothing on this box stops it any more.) That is also why this stays `az` rather than the
# project's own `blob.py`: `CloudConfig.load()` wants Terraform, which the box
# does not have -- the same reason the record DSN is passed in.
az login --identity --output none


# --------------------------------------------------------------------------- #
# code
# --------------------------------------------------------------------------- #
# Snapshots live in the store's `code` blob container.
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
        --container-name code --query "[].name | sort(@) | [-1]" -o tsv)
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
# the run, resolved first
# --------------------------------------------------------------------------- #
# Before anything is fetched: a typo'd fragment should cost a message, not an
# abstraction sync followed by a message. One listing of run PREFIXES rather
# than every object -- `--delimiter /` makes the container answer with the ~300
# run names instead of the thousands of things inside them.
#
# Matched by `src.interfaces.run_names`, the same rule every reader uses and the
# same one `pull_metadata` applies, so an ambiguous fragment gets the message it
# gets everywhere else instead of a third hand-rolled copy of the rule.
prefixes=$(az storage blob list --auth-mode login --account-name "$STORE_ACCOUNT" \
    --container-name checkpoints --delimiter "/" --query "[].name" -o tsv)
RUN_ID=$(RUN="$RUN" PREFIXES="$prefixes" PYTHONPATH="$WORK/code" python3 <<'PY'
import os
import sys

from src.interfaces import run_names

# The delimiter listing returns `<run>/`; the trailing slash is not part of the
# name any reader knows.
published = sorted({line.rstrip("/") for line in os.environ["PREFIXES"].split() if line})
fragment = os.environ["RUN"]
matches = run_names.matching(fragment, published)
if len(matches) > 1:
    sys.exit(run_names.ambiguous_message(fragment, matches))
if not matches:
    sys.exit(run_names.unknown_message(fragment, published))
print(matches[0])
PY
)
echo "==> run $RUN_ID"

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
    # THE MANIFEST AND THE LOOSE METADATA, both from the container: they live
    # under `<run>/` beside the rungs they describe. There is no share fallback
    # because there is no share -- it is empty and nothing writes to it.
    az storage blob download --auth-mode login --account-name "$STORE_ACCOUNT" \
        --container-name checkpoints --name "$run/STATIC_CHECKPOINT.json" \
        --file "$dest/STATIC_CHECKPOINT.json" --output none
    # Best-effort: a run legitimately may not have published every one of these,
    # and none of them is needed to LOAD a checkpoint -- they are what the
    # reader commands render. A missing one must not fail a deploy.
    for small in run.jsonl .run.json progress.jsonl; do
        az storage blob download --auth-mode login --account-name "$STORE_ACCOUNT" \
            --container-name checkpoints --name "$run/$small" \
            --file "$dest/$small" --output none 2>/dev/null || true
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
    # Resolved from the manifest just STAGED, not the share's: resolving against
    # one store and fetching from another is how a rung gets named that the
    # container does not hold.
    object=$(AT="$at" PYTHONPATH="$WORK/code" python3 - \
        "$dest/STATIC_CHECKPOINT.json" <<'PY'
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
# THE FIELDED LADDER, here rather than in a caller's argument.
#
# On 09-13 at 05:28 another session ran `serve-deploy` for unrelated work and,
# because this file is rewritten from ITS arguments, every `--rung` vanished.
# The seat played 1 bb-effective spots on the 100 bb blueprint for a day and
# nothing said so: the journal line reads "1 rung(s) loaded" and looks fine.
# `--no-resolver` survived the same deploy because its default lives here, and
# the rungs deserve the same protection.
#
# Pass `rungs=...` to override, or `rungs=none` to field the deepest run alone.
# A DEFAULT rather than a refusal on purpose: a deploy that fails is noticed, but
# a deploy that half-works is what actually happened, and the safe half is the
# ladder staying up.
# REPLACED 09-29. The 08-31 ladder was trained to the config of the 940.1-mbb
# exact-BR gate arm -- cfr_br=river + linplus + runouts_per_flop=1 -- i.e. chosen by
# the instrument since shown to be a fold-meter. Its 50 bb rung folds the SB first-in
# **61.7%**, which `--policy-profile` itself calls ">50% NOT A CANDIDATE", and carries
# only 68.1 mbb of postflop against 566.5 preflop because it rarely reaches its own
# postflop. chipzen sat EVEN for a month with it: 48.5% over 99 matches.
#
# These are the same depths on the CURRENT recipe (cfr_br off, dcfr beta=0,
# runouts_per_flop=4) at 14 workers, 12,000 rungs, seed 7. Each beat the rung it
# replaces on CHIPS over 20,000 duplicate deals, and the edge is MONOTONE in depth --
# which is why this is a result and not seven coin flips against a ~+53 null:
#
#   d03 +9.5+/-3   d06 +25.3+/-7   d10 +59.0+/-13   d15 +104.2+/-18
#   d25 +109.0+/-22   d35 +115.4+/-25   d50 +152.4+/-31   (mbb/hand)
#
# The four deepest are matched 12,000-vs-12,000, single-knob on the recipe. 69% of
# hands are played at 45+ bb, so d50/d35 carry the gain and d03 is a rounding error.
#
# TO REVERT: the 08-31 ladder was
#   d50 run-pcs-production-to12k-d50-long-s7-161947-15074:12000:0.20
#   d35 run-pcs-production-to12k-d35-long-s7-161938-11549:12000:0.20
#   d25 run-pcs-production-to12k-d25-long-s7-102205-23109:12000:0.20
#   d15 run-pcs-production-to4k-d15-cfrbr-s7-082749-388:4000:0.20
#   d10 run-pcs-production-to4k-d10-cfrbr-s7-082804-23297:4000:0.20
#   d06 run-pcs-production-to4k-d06-cfrbr-s7-082811-16288:4000:0.20
#   d03 run-pcs-production-to4k-d03-cfrbr-s7-095309-24766:4000:0.20
#
# Serving thresholds, MEASURED 09-29 (the 08-31 values were tuned against strategies
# that fold 62%, so they had to be re-read on ranges that actually PLAY). Purification
# wins at every depth -- ten of ten self-match arms positive, and confirmed against a
# DIFFERENT opponent at three depths with the raw-vs-raw pairing as the control -- and
# the optimum RISES WITH DEPTH: the peak is 0.20 at 25 bb, 0.35 at 50 bb and 0.50 at
# 100 bb. d50 takes 0.35 (+86.5 mbb/hand over the raw control against the old rung,
# against +68.3 at 0.20); the shallower rungs keep the 0.20 that every arm confirmed.
# 0.35 and 0.50 are UNMEASURED below 50 bb, which is the next sweep.
DEFAULT_RUNGS="run-pcs-production-to12k-d50-040438-26625:12000:0.35,\
run-pcs-production-to12k-d35-040443-23758:12000:0.20,\
run-pcs-production-to12k-d25-040447-19378:12000:0.20,\
run-pcs-production-to12k-d15-041035-5670:12000:0.20,\
run-pcs-production-to12k-d10-041039-12284:12000:0.20,\
run-pcs-production-to12k-d06-041042-19674:12000:0.20,\
run-pcs-production-to12k-d03-041046-2454:12000:0.20"
if [ -z "${RUNGS:-}" ]; then
    RUNGS="$DEFAULT_RUNGS"
    echo "==> no rungs given; fielding the default 8-rung ladder"
elif [ "${RUNGS}" = "none" ]; then
    RUNGS=""
    echo "==> rungs=none; the deepest run plays every depth alone"
fi

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

# $EXTRA_RUNS stages runs the seat will NOT play: the same `run[:at]` list, put
# on local disk so a duel can be run against the fielded ladder without giving a
# candidate the chair first. A candidate has to beat what is deployed before it
# replaces it, and the box is where that is measured -- staging it by hand was a
# second copy of every rule `stage_run` already encodes.
if [ -n "${EXTRA_RUNS:-}" ]; then
    IFS=',' read -ra _extra <<< "$EXTRA_RUNS"
    for spec in "${_extra[@]}"; do
        [ -n "$spec" ] || continue
        IFS=':' read -r _name _at _thr <<< "$spec"
        echo "==> staging $_name (not seated)"
        stage_run "$_name" "${_at:-}"
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
#
# $READER_EXTRA carries $AT through, so the chart reads the SAME rung the seat
# plays. Before it, the reader's arguments were baked into cloud-init with no
# `--at` and the two could silently disagree about what "Blueprint" means.
#
# The reader gets THE SAME LADDER AS THE SEAT. The rungs are staged once above and
# the reader adds no copy of them: measured 09-29, one 100 bb blueprint under
# `play_only` is 0.77 GB against 1.92 GB loading all five arrays, so the whole
# ladder fits beside the seat's 2.89 GB on a 15 GB box. `/api/run` then lists every
# stack and `/api/node?stack=N` picks one, which is what lets the chart show a 6 bb
# range rather than implying the 100 bb one applies at every depth.
#
# `$SEAT_RUNGS` is reused verbatim -- it is already `--rung run:at:thr` per rung and
# `blueprint-serve` ignores the threshold field. Building a second list here is how
# the two would drift into disagreeing about what is fielded.
#
# These notes live HERE, not inside the heredoc below. It is UNQUOTED -- it has to
# interpolate $RUN_ID and $SEAT_RUNGS -- so a backtick in a comment there is command
# substitution: every deploy since 09-29 printed `play_only: command not found` five
# times and wrote the comment with those words deleted.
sudo tee /etc/blueprint.env >/dev/null <<EOF
RUN=$RUN_ID
RUNS_DIR=$WORK/data/runs
POKER_SOLVER_RECORD_DSN=$RECORD_DSN
READER_EXTRA=${AT:+--at $AT}$SEAT_RUNGS
EOF
# The DSN carries a password; the unit runs as root and is the only reader.
sudo chmod 600 /etc/blueprint.env

# THE UNIT ITSELF, from the tree just extracted -- not from cloud-init.
#
# It lived in `write_files`, which runs ONCE at first boot, so the only way to
# correct a line in it was to recreate the machine. That is how a box spent
# 62 hours idling out every 30 minutes and waking itself back up, and how the
# reader then spent two days dead on a box that was up: an idle exit systemd
# read as success, so `Restart=on-failure` never fired and `box-may-sleep`
# (rightly) refused to deallocate. Both bugs were one line in a file no deploy
# could reach. Now a deploy ships the unit, exactly as it ships the seat's.
sudo install -m 0644 "$WORK/code/infra/serve/blueprint.service" /etc/systemd/system/
# The idle era's leftovers, which an `install` does not remove: a drop-in saying
# exit 42 is a success, and the deallocate this unit no longer calls. Left in
# place they would outlive every trace of the feature in the repo.
sudo rm -rf /etc/systemd/system/blueprint.service.d \
    /etc/systemd/system/blueprint-deallocate.service.d \
    /etc/systemd/system/blueprint-deallocate.service \
    /usr/local/bin/deallocate-if-idle /usr/local/bin/box-may-sleep \
    /usr/local/bin/deallocate-box

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
# argparse. It purifies the BASE blueprint -- every hand deeper than the shallowest
# rung below -- while each `$RUNGS` entry carries its own suffix.
#
# 0.20, not the 0.02 fielded until 09-29. That 0.02 was tuned on EXPLOITABILITY
# (940.1 -> 854.0 mbb/hand, and 0.10 measured WORSE there), which is a fold-meter;
# on CHIPS the ordering inverts. MEASURED 09-29 on duplicate deals, `26596@16000`
# purified against `3512@12000` raw with the raw-vs-raw pairing as the control:
# the cut is worth +47.7 mbb/hand at 100 bb, +71.1 at 50 bb and +35.2 at 25 bb,
# and a self-match at eight depths put every one of ten arms positive. 0 fields
# the table as trained.
#
# 0.35, because the optimum RISES WITH DEPTH and this serves the deepest hands
# (the seat takes the rung nearest in LOG depth, so this one covers ~70 bb up).
# Cross-opponent gain over the raw control, 40,000 duplicate deals: at 100 bb
# 0.20 +57.6, 0.35 +61.3, 0.50 +91.4; at 50 bb 0.20 +68.3, 0.35 +86.5, 0.50
# +65.4; at 25 bb 0.20 is already the peak. 0.35 is the highest value no
# measurement argues against.
#
# NOT 0.50, though it scored best at 100 bb: 0.50 and above are ARGMAX (a row
# keeps its largest action when everything is below the cut, so at most one
# action can hold half the mass), it was WORSE than 0.35 at 50 bb, and 0.7 sigma
# is not enough to field a fully deterministic strategy -- which is the one thing
# these static opponents cannot punish and an adaptive one could.
#
# SEAT_EXTRA carries $AT through to the seat. `blueprint-serve` grew an `--at`
# of its own, but the READER'S unit is baked into cloud-init without one, so in
# practice the two still want different rungs from one staged run: the reader
# takes the manifest head, the seat fields the rung a NUMBER was measured at.
# That is why both are staged above. Empty when $AT is unset, which the unit
# expands to no arguments.
# $BUDGET_MS caps the per-decision budget. Unset leaves the seat's own sizing,
# which is what is fielded, and MEASURED over 26 h of live matches that is
# 900 ms -- the `MAX_BUDGET_MS` cap -- not the tight-clock default:
#
#   Clock 30,000 ms -> budget 900 ms   x100   every real match
#   Clock  5,000 ms -> budget 900 ms     x2
#   Clock unstated  -> budget 200 ms    x32   warm/startup path only
#
# `budget_for` is clock/2 - 800 capped at 900, so the 2,000 ms clock assumed
# when a frame states none gives 200 -- but frames in a MATCH state 30 s. Read
# the journal for this, not the formula: reasoning from the code alone got it
# backwards once already.
#
# That leaves a live question. The resolver's 528 mbb/hand was measured near
# 300 ms and it is fielded at 900, three times the work at the one knob
# `DEC-0013` says converges to shipping stacks.
#
# It is NOT a stack-off lever, though one reading said so: a single spot jammed
# 32.5% at 200 ms against 41.0% at 900. Over a full duel the sign is the other
# way -- 6.11% of 1,800 decisions at 300 ms against 4.78% of 1,779 at 900 --
# and both readings are ~1.8 sigma, i.e. neither is a result. One node's rate
# is not a policy's rate.
#
# $RESOLVER turns the runtime resolver back ON; it is OFF by default.
# MEASURED 09-12 against
# a genuine off-tree opponent (the `gears` arm is a different action tree, so
# it makes off-menu sizes by construction) at the budget actually fielded:
#
#   resolver on @900ms   -227.8 +/- 235.4 mbb/hand vs gears   6.62% stack-offs/hand
#   resolver off          -27.8 +/- 144.5                      3.62%
#
# The resolver is worth -200 +/- 276 mbb/hand -- indistinguishable from zero --
# while multiplying stack-offs by 1.83 (z=2.72). Its whole justification was
# +528 mbb/hand OFF-TREE, and that figure sits 2.6 sigma from this measurement:
# it was taken at the EVALUATOR's settings and never re-checked at 900 ms.
#
# The mechanism is visible in single spots: with Kc8h at 53 bb the blueprint
# folds 150 times out of 150 and the resolver ships the whole stack 27.3% of
# the time, offering only fold-or-jam -- a local tree ending at the next street
# has no future betting to play for. In an ELIMINATION format that trade is
# worse than the mbb figure suggests: a stack-off that loses ends the match.
#
# $RESOLVER_OFF_TREE runs the resolver only once the hand has left our tree.
# Its measured +528 mbb/hand is an OFF-TREE gain; on tree it is worth
# -203.6 +/- 133.1 mbb/hand (1.5 sigma, nothing) while DOUBLING the stack-off
# rate, 5.25% against 2.37% over ~5,500 decisions. Doubling variance for a
# statistically-zero edge loses in an ELIMINATION format, where a stack-off
# that loses ends the match and there is no next hand to earn it back.
#
# These notes stay ABOVE the heredoc on purpose. It is unquoted -- $RUN_ID
# and $RECORD_DSN must expand -- so everything inside it is shell input:
# backticks were SUBSTITUTED on the box and $BUDGET_MS expanded to nothing.
# THE RESOLVER IS OFF UNLESS A DEPLOY ASKS FOR IT, and that default lives HERE
# rather than in a caller's argument. It was fielded off on 09-12 and silently
# came back 22 minutes later, because another session's `serve-deploy` rewrites
# this file from ITS arguments and simply omitted the flag. A fielded setting
# that survives only while every future caller remembers a flag is not fielded.
# Pass `resolver=1` to `just serve-deploy` to turn it back on.
RESOLVER_FLAG=" --no-resolver"
[ -n "${RESOLVER:-}" ] && RESOLVER_FLAG=""
sudo tee /etc/chipzen-seat.env >/dev/null <<EOF
RUN=$RUN_ID
RUNS_DIR=$WORK/data/runs
CHIPZEN_ENV=${CHIPZEN_ENV:-prod}
POKER_SOLVER_RECORD_DSN=$RECORD_DSN
POLICY_THRESHOLD=${POLICY_THRESHOLD:-0.35}
SEAT_EXTRA=${AT:+--at $AT}${BUDGET_MS:+ --budget-ms $BUDGET_MS}${RESOLVER_FLAG}${RESOLVER_OFF_TREE:+ --resolver-off-tree-only}$SEAT_RUNGS
EOF
# The DSN carries a password, and this file gained one the moment the seat began
# reading run metadata from the record. Same mode as `/etc/blueprint.env`.
sudo chmod 600 /etc/chipzen-seat.env

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
