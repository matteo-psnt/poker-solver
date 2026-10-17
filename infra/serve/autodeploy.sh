#!/usr/bin/env bash
# Deploy every commit that lands on `main` to the serving box.
#
#   autodeploy.sh hook     what .git/hooks/post-commit runs (`just autodeploy-install`)
#   autodeploy.sh worker   the background half; never run it by hand
#
# The hook returns at once: it records the commit as WANTED and forks a worker.
# One worker at a time holds the lock, waits out the 00:00-05:00 UTC match window
# (deploy.sh says why), and deploys the newest wanted commit, so a burst of
# merges coalesces into one deploy. The settings the seat fields are NOT this
# script's: `just serve-redeploy` reads them off the box at deploy time.
#
# What is deployed is the COMMIT, built in a scratch checkout, never the primary
# working tree, which other sessions leave dirty.

set -euo pipefail

STATE="${POKER_SOLVER_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/poker-solver}/autodeploy"
mkdir -p "$STATE"
LOG="$STATE/log"

log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >> "$LOG"; }

notify() {
    # Best effort: a background failure nobody hears about is the failure mode
    # this exists to prevent, but a missing notifier must not fail the deploy.
    osascript -e "display notification \"$2\" with title \"autodeploy: $1\"" \
        >/dev/null 2>&1 || true
}

# Paths whose change alters what the box runs. Anything else (memory, docs,
# tests alone) skips the restart.
RELEVANT='^(src/|console/|infra/serve/|config/|pyproject\.toml$|uv\.lock$|justfile$)'

hook() {
    local repo common sha
    repo=$(git rev-parse --show-toplevel)
    common=$(cd "$repo" && cd "$(git rev-parse --git-common-dir)" && pwd)
    # The primary checkout's `main` only: a worktree's commits are a branch.
    [ "$common" = "$repo/.git" ] || return 0
    [ "$(git symbolic-ref --short -q HEAD || true)" = main ] || return 0
    sha=$(git rev-parse HEAD)
    if [ -e "$STATE/off" ]; then
        log "off: skipping ${sha:0:8} ($STATE/off exists)"
        return 0
    fi
    if ! git diff-tree --no-commit-id --name-only -r --root HEAD | grep -qE "$RELEVANT"; then
        log "skip ${sha:0:8}: nothing the box runs changed"
        return 0
    fi
    echo "$sha" > "$STATE/wanted"
    log "wanted ${sha:0:8}"
    echo "autodeploy: ${sha:0:8} queued; log at $LOG" >&2
    nohup "$repo/infra/serve/autodeploy.sh" worker "$repo" >> "$LOG" 2>&1 < /dev/null &
}

take_lock() {
    if mkdir "$STATE/lock" 2>/dev/null; then
        echo $$ > "$STATE/lock/pid"
        return 0
    fi
    # A worker killed by a reboot or a closed lid leaves its lock behind.
    local holder
    holder=$(cat "$STATE/lock/pid" 2>/dev/null || true)
    if [ -n "$holder" ] && kill -0 "$holder" 2>/dev/null; then
        return 1
    fi
    log "stealing a stale lock from pid ${holder:-?}"
    echo $$ > "$STATE/lock/pid"
}

in_match_window() { [ "$(date -u +%H)" -lt 5 ]; }

# Every step checks itself: this runs as an `if` condition, where bash ignores
# `set -e`, so an unchecked failure would deploy whatever came before it.
deploy() {
    local repo="$1" sha="$2" tree code
    tree=$(mktemp -d)/tree
    git -C "$repo" worktree add --detach -q "$tree" "$sha" || return 1
    # Expanded now, not at return: $tree is local to this call.
    # shellcheck disable=SC2064
    trap "git -C '$repo' worktree remove --force '$tree' >/dev/null 2>&1 || true" RETURN
    ln -s "$repo/console/node_modules" "$tree/console/node_modules" || return 1
    log "building console for ${sha:0:8}"
    npm --prefix "$tree/console" run build >> "$LOG" 2>&1 || return 1
    code=$(cd "$repo" && uv run poker-solver push-code --root "$tree" | tail -1) || return 1
    [[ "$code" == code-* ]] || { log "push-code printed '$code', not a snapshot id"; return 1; }
    log "pushed $code; redeploying with the seat's current settings"
    (cd "$repo" && just serve-redeploy "$code") >> "$LOG" 2>&1 || return 1
    log "deployed ${sha:0:8} as $code"
}

worker() {
    local repo="$1" sha deployed=""
    take_lock || exit 0
    trap 'rm -rf "$STATE/lock"' EXIT
    while :; do
        while in_match_window; do
            sleep 60
        done
        sha=$(cat "$STATE/wanted")
        [ "$sha" != "$deployed" ] || break
        if deploy "$repo" "$sha"; then
            deployed="$sha"
            echo "$sha" > "$STATE/deployed"
            notify ok "deployed ${sha:0:8}"
        else
            log "FAILED ${sha:0:8}; see above"
            notify FAILED "${sha:0:8} did not deploy; log: $LOG"
            exit 1
        fi
    done
}

case "${1:-}" in
    hook) hook ;;
    worker) worker "$2" ;;
    *) echo "usage: autodeploy.sh hook|worker <repo>" >&2; exit 2 ;;
esac
