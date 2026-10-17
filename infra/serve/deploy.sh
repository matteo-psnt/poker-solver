#!/usr/bin/env bash
# Bring the serving box to one code snapshot and one seat plan.
#
#   deploy.sh console|full <record-dsn> <storage-account>
#
# RUNS ON THE BOX, from the snapshot it deploys: `infra/serve/ci-deploy.sh` (what
# GitHub Actions and `just deploy` run) starts it through Run Command. `console`
# swaps the console's own tree and restarts nothing else, so it is safe at any
# hour. `full` also stages the seat's rungs, swaps the tree the seat and the chart
# run from, and restarts them -- never inside 00:00-05:00 UTC, chipzen's whole
# daily match window, which ci-deploy.sh enforces.
#
# What the seat fields arrives already resolved by `poker-solver seat-plan`, as
# SEAT_RUN, SEAT_AT, SEAT_THRESHOLD and SEAT_RUNGS (`run:at:threshold`, space
# separated). This script decides nothing about it: the file that does is
# `infra/serve/seat.toml`, and two writers of the box's env files is how a bare
# deploy twice reverted the shallow rungs to 12000.
#
# NOTHING LIVE CHANGES UNTIL EVERYTHING IS IN PLACE. Every rung is downloaded and
# both trees are extracted before a tree is swapped, an env file is written or a
# service restarts, so a failed deploy leaves the box as it was.

set -euo pipefail

MODE="${1:-}" RECORD_DSN="${2:-}" STORE_ACCOUNT="${3:-}"
case "$MODE" in
    console | full) ;;
    *) echo "usage: deploy.sh console|full <record-dsn> <storage-account>" >&2; exit 2 ;;
esac
for var in RECORD_DSN STORE_ACCOUNT CODE API_TOKEN PUBLIC_HOST OUTPUTS_INFRA OUTPUTS_STORE; do
    [ -n "${!var:-}" ] || { echo "deploy.sh: $var is empty" >&2; exit 2; }
done
if [ "$MODE" = full ]; then
    for var in SEAT_RUN SEAT_AT SEAT_THRESHOLD; do
        [ -n "${!var:-}" ] || { echo "deploy.sh full: $var is empty" >&2; exit 2; }
    done
fi

WORK=/mnt/work
UV="$HOME/.local/bin/uv"

# The box reads the store as its managed identity (Storage Blob Data Reader), so
# no key lives here.
az login --identity --output none
fetch() {
    az storage blob download --auth-mode login --account-name "$STORE_ACCOUNT" \
        --container-name "$1" --name "$2" --file "$3" --output none
}

# --------------------------------------------------------------------------- #
# code: extracted beside each live tree, swapped in only at the end
# --------------------------------------------------------------------------- #
tarball="$WORK/code.incoming.tar.gz"
trap 'rm -f "$tarball"' EXIT
fetch code "${CODE:?}.tar.gz" "$tarball"
echo "==> code $CODE"

extract() {
    rm -rf "$1.incoming"
    mkdir -p "$1.incoming"
    tar -xzf "$tarball" -C "$1.incoming"
}

# The services run with the tree as WorkingDirectory and the abstraction resolver
# scans `<cwd>/data`, so `data` is linked in: the rungs are far too big to live in
# a tree that gets replaced. The venv is synced after the move, not before, because
# a venv's scripts carry its absolute path.
swap() {
    local tree="$1" extras="$2"
    rm -rf "$tree.previous"
    if [ -d "$tree" ]; then mv "$tree" "$tree.previous"; fi
    mv "$tree.incoming" "$tree"
    ln -sfn "$WORK/data" "$tree/data"
    # shellcheck disable=SC2086  # $extras is zero or more flags
    (cd "$tree" && "$UV" sync --no-dev $extras)
}

# --------------------------------------------------------------------------- #
# the read-only console, on the tailnet
# --------------------------------------------------------------------------- #
# `poker-solver serve --read-only` on loopback, published to the tailnet ONLY by
# `tailscale serve`: no Caddy route, no NSG rule. It reads as this box's identity
# and from Terraform outputs rendered WITHOUT the store's key or the DSN, which
# reaches it through its environment instead. It runs from a tree of its own, so
# swapping it never touches the seat.
field_console() {
    local outputs="$WORK/console/outputs" env_file
    mkdir -p "$outputs"
    echo "$OUTPUTS_INFRA" | base64 -d > "$outputs/infra.json"
    echo "$OUTPUTS_STORE" | base64 -d > "$outputs/infra-store.json"
    env_file=$(mktemp)
    {
        echo "POKER_SOLVER_OUTPUTS_DIR=$outputs"
        echo "POKER_SOLVER_RECORD_DSN=$RECORD_DSN"
        echo "POKER_SOLVER_BLUEPRINT_URL=http://127.0.0.1:8790"
        echo "POKER_SOLVER_BLUEPRINT_TOKEN=$API_TOKEN"
    } > "$env_file"
    sudo install -m 0600 -o solver -g solver "$env_file" /etc/console.env
    rm -f "$env_file"
    sudo install -m 0644 "$WORK/console-code/infra/serve/console.service" /etc/systemd/system/
    if ! command -v tailscale >/dev/null 2>&1; then
        curl -fsSL https://tailscale.com/install.sh | sudo sh >/dev/null
    fi
    # This box has no npm: the page is whatever `console/dist` the snapshot carries.
    [ -f "$WORK/console-code/console/dist/index.html" ] ||
        echo "==> console: this snapshot has no console/dist; push one built with \`just console-build\`"
    sudo systemctl daemon-reload
    sudo systemctl enable --quiet console
    sudo systemctl restart console
    # Joining the tailnet is a login, done once by hand; until then the console
    # runs on loopback and nothing can reach it.
    if sudo tailscale status >/dev/null 2>&1; then
        sudo tailscale serve --bg --https=443 http://127.0.0.1:8765 >/dev/null
        echo "==> console: on the tailnet, read-only"
    else
        echo "==> console: loopback only; join the tailnet once with" \
             "\`sudo tailscale up --hostname poker-console\` and redeploy"
    fi
}

extract "$WORK/console-code"
if [ "$MODE" = console ]; then
    swap "$WORK/console-code" ""
    field_console
    exit 0
fi
extract "$WORK/code"

# --------------------------------------------------------------------------- #
# the card abstractions
# --------------------------------------------------------------------------- #
# One `<name>.tar.zst` each in the `abstractions` container, MERGED rather than
# mirrored: which one a run resolves against is not known until its config is
# read, so all of them have to be here. One already unpacked is not re-fetched.
mkdir -p "$WORK/data/combo_abstraction" "$WORK/data/runs"
for packed in $(az storage blob list --auth-mode login --account-name "$STORE_ACCOUNT" \
    --container-name abstractions --query "[].name" -o tsv); do
    case "$packed" in *.tar.zst) ;; *) continue ;; esac
    unpacked="$WORK/data/combo_abstraction/${packed%.tar.zst}"
    [ -d "$unpacked" ] && continue
    echo "==> abstraction ${packed%.tar.zst}"
    fetch abstractions "$packed" "$unpacked.tar.zst"
    tar --zstd -xf "$unpacked.tar.zst" -C "$WORK/data/combo_abstraction"
    rm -f "$unpacked.tar.zst"
done

# --------------------------------------------------------------------------- #
# the rungs the seat fields
# --------------------------------------------------------------------------- #
# One object per rung, `<run>/static-<at>.ckpt.zst`, which seat-plan has already
# confirmed the container holds. Idempotent by SIZE: a file on disk is no evidence
# of a complete copy, and a truncated one fails deep in the reader minutes later.
# The manifest and the loose metadata come along because the readers render them.
stage() {
    local run="$1" at="$2" dest="$WORK/data/runs/$1" object="static-$2.ckpt.zst" held have
    mkdir -p "$dest"
    fetch checkpoints "$run/STATIC_CHECKPOINT.json" "$dest/STATIC_CHECKPOINT.json"
    for small in run.jsonl .run.json progress.jsonl; do
        fetch checkpoints "$run/$small" "$dest/$small" 2>/dev/null || true
    done
    held=$(az storage blob show --auth-mode login --account-name "$STORE_ACCOUNT" \
        --container-name checkpoints --name "$run/$object" --query properties.contentLength -o tsv)
    have=$(stat -c %s "$dest/$object" 2>/dev/null || echo 0)
    if [ "$have" = "$held" ]; then
        echo "==> $run@$at already staged"
    else
        echo "==> staging $run@$at ($((held / 1000000)) MB)"
        # Under a temp name: a half-written rung under the real one is what the
        # seat would try to load on its next restart.
        fetch checkpoints "$run/$object" "$dest/$object.partial"
        mv "$dest/$object.partial" "$dest/$object"
    fi
    fielded+=("$dest/$object")
}

fielded=()
stage "$SEAT_RUN" "$SEAT_AT"
rung_flags=""
read -ra rung_specs <<< "${SEAT_RUNGS:-}"
for spec in "${rung_specs[@]}"; do
    IFS=':' read -r run at _ <<< "$spec"
    stage "$run" "$at"
    rung_flags="$rung_flags --rung $spec"
done

# --------------------------------------------------------------------------- #
# go live
# --------------------------------------------------------------------------- #
swap "$WORK/console-code" ""
# `--extra chipzen`: the seat holds its WebSocket through an optional extra that a
# bare `--no-dev` skips.
swap "$WORK/code" "--extra chipzen"

# The chart reads the same rungs the seat plays; `blueprint-serve` ignores each
# rung's threshold. The seat's `--no-resolver` is in its unit: the resolver
# measured worth nothing (09-12) and doubled the stack-off rate. Explanations stay
# out of these heredocs: they interpolate, so a backtick would run.
sudo tee /etc/blueprint.env >/dev/null <<EOF
RUN=$SEAT_RUN
RUNS_DIR=$WORK/data/runs
POKER_SOLVER_RECORD_DSN=$RECORD_DSN
READER_EXTRA=--at $SEAT_AT$rung_flags
EOF
sudo tee /etc/chipzen-seat.env >/dev/null <<EOF
RUN=$SEAT_RUN
RUNS_DIR=$WORK/data/runs
CHIPZEN_ENV=prod
POKER_SOLVER_RECORD_DSN=$RECORD_DSN
POLICY_THRESHOLD=$SEAT_THRESHOLD
SEAT_EXTRA=--at $SEAT_AT$rung_flags
EOF
# Both carry the DSN's password.
sudo chmod 600 /etc/blueprint.env /etc/chipzen-seat.env

units="$WORK/code/infra/serve"
sudo install -m 0644 "$units/blueprint.service" "$units/chipzen-seat.service" \
    "$units/chipzen-seat-watchdog.service" "$units/chipzen-seat-watchdog.timer" /etc/systemd/system/
sudo install -m 0755 "$units/chipzen-seat-watchdog" /usr/local/bin/

# Caddy holds the token door and the public play page. Rendered from the tree,
# validated, and left alone rather than installed half-rendered: a blank token
# would match `Bearer ` and open the private surface to everyone.
rendered=$(mktemp)
sed -e "s|__HOST__|$PUBLIC_HOST|" -e "s|__TOKEN__|$API_TOKEN|" "$units/Caddyfile" > "$rendered"
if ! grep -q '__HOST__\|__TOKEN__' "$rendered" &&
    sudo caddy validate --adapter caddyfile --config "$rendered" >/dev/null 2>&1; then
    sudo install -m 0640 -o root -g caddy "$rendered" /etc/caddy/Caddyfile
    sudo systemctl reload caddy
else
    echo "==> caddy: the rendered Caddyfile does not validate; left as it was" >&2
fi
rm -f "$rendered"

field_console
sudo systemctl daemon-reload
sudo systemctl enable --quiet blueprint
sudo systemctl restart blueprint
# Seating is a deliberate act (`just seat-on`), so a deploy restarts the seat only
# when it is already enabled -- onto the new code, which it imported at start.
if systemctl is-enabled --quiet chipzen-seat 2>/dev/null; then
    echo "==> restarting the seat"
    sudo systemctl restart chipzen-seat chipzen-seat-watchdog.timer
fi

echo "==> waiting for the blueprint server (a production run takes ~1 min)"
for _ in $(seq 1 90); do
    if curl -fsS --max-time 2 http://127.0.0.1:8790/api/health >/dev/null 2>&1; then
        echo "==> serving $SEAT_RUN@$SEAT_AT"
        # Following `latest` adds a ~1.5 GB rung every two hours and the container
        # keeps them all, so a fielded run's directory keeps only its fielded rung
        # -- pruned now, once nothing can restart onto the old one.
        for object in "${fielded[@]}"; do
            find "$(dirname "$object")" -maxdepth 1 -name 'static-*.ckpt.zst' \
                ! -name "$(basename "$object")" -print -delete
        done
        exit 0
    fi
    if ! systemctl is-active --quiet blueprint; then
        sudo journalctl -u blueprint -n 40 --no-pager >&2
        exit 1
    fi
    sleep 2
done
echo "the blueprint server did not come up:" >&2
sudo journalctl -u blueprint -n 40 --no-pager >&2
exit 1
