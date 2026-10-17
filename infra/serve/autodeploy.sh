#!/usr/bin/env bash
# Deploy what is pushed to GitHub's `main` to the serving box.
#
#   autodeploy.sh push <remote>   what .git/hooks/pre-push runs (`just autodeploy-install`)
#   autodeploy.sh worker <repo>   the background half; never run it by hand
#
# Two kinds of deploy. Every push that touches what the box runs redeploys the
# CONSOLE at once, from its own tree on the box, at any hour. A push that also
# touches anything else redeploys EVERYTHING, which restarts the seat, so that one
# waits out the 00:00-05:00 UTC match window (deploy.sh says why).
#
# The hook returns at once: it records the pushed commit as wanted and forks a
# worker. One worker at a time holds the lock and always deploys the NEWEST
# wanted commit, so a burst of pushes coalesces. The settings the seat fields
# are NOT this script's: `just serve-redeploy` reads them off the box.
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
# What only the console runs: a push confined to these leaves the seat alone.
CONSOLE_ONLY='^(console/|src/interfaces/web/|infra/serve/console\.service$)'

# pre-push hands over "<local ref> <local sha> <remote ref> <remote sha>" lines.
push() {
    local remote="$1" common repo local_ref local_sha remote_ref remote_sha changed
    [ "$remote" = origin ] || return 0
    common=$(cd "$(git rev-parse --git-common-dir)" && pwd)
    repo=$(dirname "$common")
    # shellcheck disable=SC2034
    while read -r local_ref local_sha remote_ref remote_sha; do
        [ "$remote_ref" = refs/heads/main ] || continue
        case "$local_sha" in *[!0]*) ;; *) continue ;; esac
        if [ -e "$STATE/off" ]; then
            log "off: skipping ${local_sha:0:8} ($STATE/off exists)"
            continue
        fi
        if git cat-file -e "$remote_sha^{commit}" 2>/dev/null; then
            changed=$(git diff --name-only "$remote_sha" "$local_sha")
        else
            changed=$(git ls-tree -r --name-only "$local_sha")
        fi
        if ! grep -qE "$RELEVANT" <<< "$changed"; then
            log "skip ${local_sha:0:8}: nothing the box runs changed"
            continue
        fi
        echo "$local_sha" > "$STATE/wanted-console"
        if grep -E "$RELEVANT" <<< "$changed" | grep -qvE "$CONSOLE_ONLY"; then
            echo "$local_sha" > "$STATE/wanted-full"
            log "wanted ${local_sha:0:8}: console now, the seat after the match window"
        else
            log "wanted ${local_sha:0:8}: console only"
        fi
        echo "autodeploy: ${local_sha:0:8} queued; log at $LOG" >&2
        nohup "$repo/infra/serve/autodeploy.sh" worker "$repo" >> "$LOG" 2>&1 < /dev/null &
    done
}

# pre-push runs BEFORE the push: deploy only what GitHub accepted, which a
# successful push records by moving origin/main.
landed() {
    local repo="$1" sha="$2" _
    for _ in $(seq 60); do
        git -C "$repo" merge-base --is-ancestor "$sha" refs/remotes/origin/main 2>/dev/null && return 0
        sleep 2
    done
    return 1
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
    local repo="$1" kind="$2" sha="$3" tree code recipe=serve-console
    [ "$kind" = full ] && recipe=serve-redeploy
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
    log "pushed $code; just $recipe"
    (cd "$repo" && just "$recipe" "$code") >> "$LOG" 2>&1 || return 1
    log "deployed ${sha:0:8} as $code ($kind)"
}

# A kind is pending while what was wanted is not yet inside what was deployed.
pending() {
    local repo="$1" kind="$2" wanted deployed
    wanted=$(cat "$STATE/wanted-$kind" 2>/dev/null) || return 1
    deployed=$(cat "$STATE/deployed-$kind" 2>/dev/null) || return 0
    ! git -C "$repo" merge-base --is-ancestor "$wanted" "$deployed" 2>/dev/null
}

worker() {
    local repo="$1" kind sha
    take_lock || exit 0
    trap 'rm -rf "$STATE/lock"' EXIT
    while :; do
        if pending "$repo" full && ! in_match_window; then
            kind=full
        elif pending "$repo" console; then
            kind=console
        elif pending "$repo" full; then
            sleep 60
            continue
        else
            break
        fi
        # Every wanted push writes wanted-console, so it is the newest: a full
        # deploy of anything older would roll the console back.
        sha=$(cat "$STATE/wanted-console")
        if ! landed "$repo" "$sha"; then
            log "not deploying ${sha:0:8}: origin/main does not contain it (push rejected?)"
            break
        fi
        if deploy "$repo" "$kind" "$sha"; then
            echo "$sha" > "$STATE/deployed-console"
            [ "$kind" = console ] || echo "$sha" > "$STATE/deployed-full"
            notify ok "deployed ${sha:0:8} ($kind)"
        else
            log "FAILED ${sha:0:8} ($kind); see above"
            notify FAILED "${sha:0:8} did not deploy ($kind); log: $LOG"
            exit 1
        fi
    done
}

case "${1:-}" in
    push) push "$2" ;;
    worker) worker "$2" ;;
    *) echo "usage: autodeploy.sh push <remote> | worker <repo>" >&2; exit 2 ;;
esac
