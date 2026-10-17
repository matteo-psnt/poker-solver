#!/usr/bin/env bash
# Bring the serving box to HEAD and to what `infra/serve/seat.toml` fields. The
# only way anything is deployed: `.github/workflows/deploy.yml` runs it on every
# push to main and every two hours, and `just deploy` runs it from a laptop
# signed in with `az login`.
#
# Two kinds of deploy. A change only the CONSOLE runs goes out at once, from the
# console's own tree on the box, at any hour. A change to the seat's code, or a
# new rung for a seat entry that follows `latest`, restarts the seat, so it waits
# until 05:00 UTC (deploy.sh says why) and the next scheduled run deploys it.
# Markers in the store record what is deployed, so every trigger runs the same
# reconcile and none of them needs to remember anything.
#
# The box is reached through Run Command, not SSH: its NSG admits one address on
# port 22, and this way GitHub holds no key.

set -euo pipefail

RELEVANT='^(src/|console/|infra/serve/|config/|pyproject\.toml$|uv\.lock$|justfile$)'
CONSOLE_ONLY='^(console/|src/interfaces/web/|infra/serve/console\.service$)'
MARKERS="serve-deploy"  # <diagnostics>/serve-deploy/{console,full,seat}: sha, sha, seat-plan digest

# This repo is public, and so are its Actions logs: every secret is masked the
# moment it is read, before anything could print it.
mask() {
    [ -z "${GITHUB_ACTIONS:-}" ] || echo "::add-mask::$1"
}

store=$(terraform -chdir=infra/store output -json)
serve=$(terraform -chdir=infra/serve output -json)
account=$(jq -r .storage_account.value <<< "$store")
markers=$(jq -r .diagnostics_container_name.value <<< "$store")
key=$(jq -r .access_key.value <<< "$store") && mask "$key"
dsn=$(jq -r .postgres_dsn.value <<< "$store") && mask "$dsn"
token=$(jq -r .api_token.value <<< "$serve") && mask "$token"
rg=$(jq -r .resource_group.value <<< "$serve")
vm=$(jq -r .vm_name.value <<< "$serve")
host=$(jq -r .url.value <<< "$serve" | sed 's|^https://||')
sha=$(git rev-parse HEAD)

power=$(az vm get-instance-view -g "$rg" -n "$vm" \
    --query "instanceView.statuses[?starts_with(code, 'PowerState/')].code | [0]" -o tsv)
if [ "$power" != PowerState/running ]; then
    # Waking the box is the user's decision; the next run after it wakes deploys.
    echo "The box is $power: not deploying. Wake it with \`just serve-start\`."
    exit 0
fi

scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
deployed() {
    az storage blob download --account-name "$account" --account-key "$key" \
        --container-name "$markers" --name "$MARKERS/$1" --file "$scratch/$1" \
        --output none 2>/dev/null && cat "$scratch/$1" || true
}

# Files changed since a marker; with no usable marker, everything counts.
changed_since() {
    if [ -n "$1" ] && git merge-base --is-ancestor "$1" "$sha" 2>/dev/null; then
        git diff --name-only "$1" "$sha" | grep -E "$RELEVANT" || true
    else
        echo "(no deployed commit on record)"
    fi
}

# Resolved before anything else: a typo'd run or a rung the container lacks
# fails here, with the box untouched.
plan=$(uv run poker-solver seat-plan --json)
digest=$(jq -r .digest <<< "$plan")
jq -r '"seat: \(.base.run)@\(.base.at)", (.rungs[] | "      \(.run)@\(.at)")' <<< "$plan"

full_changes=$(changed_since "$(deployed full)" | grep -vE "$CONSOLE_ONLY" || true)
[ "$(deployed seat)" = "$digest" ] || full_changes="$full_changes (seat plan $digest)"
console_changes=$(changed_since "$(deployed console)")
hour=$(date -u +%H)

if [ -n "$full_changes" ] && [ "$((10#$hour))" -ge 5 ]; then
    kind=full
elif [ -n "$console_changes" ]; then
    kind=console
elif [ -n "$full_changes" ]; then
    echo "The seat has changes, but it is ${hour}h UTC, inside the match window: a run after 05:00 deploys them."
    exit 0
else
    echo "Nothing to deploy: the box runs ${sha:0:8}'s code and seat plan $digest."
    exit 0
fi
echo "==> $kind deploy of ${sha:0:8}"
[ "$kind" = console ] && [ -n "$full_changes" ] &&
    echo "    (the seat has changes too: a run after 05:00 UTC deploys them)"

npm --prefix console run build
code=$(uv run poker-solver push-code --root . | tail -1)
[[ "$code" == code-* ]] || { echo "push-code printed '$code', not a snapshot id" >&2; exit 1; }
echo "==> pushed $code"

# Run Command delivers parameters as environment variables to a ROOT script, and
# `--run-as-user` drops them (sudo resets the environment), so the script hands
# them to `solver` itself. deploy.sh comes from the snapshot: the commit's own.
# shellcheck disable=SC2016  # expanded on the box, not here
script='set -uo pipefail
log=/mnt/work/deploy.log
sudo -u solver -H --preserve-env=MODE,CODE,RECORD_DSN,STORE_ACCOUNT,API_TOKEN,PUBLIC_HOST,OUTPUTS_INFRA,OUTPUTS_STORE,SEAT_RUN,SEAT_AT,SEAT_THRESHOLD,SEAT_RUNGS bash -c "
    set -euo pipefail
    az login --identity --output none
    t=\$(mktemp -d)
    trap \"rm -rf \$t\" EXIT
    az storage blob download --auth-mode login --account-name \"\$STORE_ACCOUNT\" \
        --container-name code --name \"\$CODE.tar.gz\" --file \"\$t/code.tar.gz\" --output none
    tar -xzf \"\$t/code.tar.gz\" -C \"\$t\" infra/serve/deploy.sh
    bash \"\$t/infra/serve/deploy.sh\" \"\$MODE\" \"\$RECORD_DSN\" \"\$STORE_ACCOUNT\" < /dev/null
" > "$log" 2>&1
rc=$?
grep -v "ts\.net" "$log" | tail -c 3500
exit $rc'
name="deploy-${sha:0:8}-$kind-$(date -u +%H%M%S)"
az vm run-command create -g "$rg" --vm-name "$vm" --name "$name" \
    --script "$script" --parameters MODE="$kind" CODE="$code" STORE_ACCOUNT="$account" PUBLIC_HOST="$host" \
        SEAT_RUN="$(jq -r .base.run <<< "$plan")" SEAT_AT="$(jq -r .base.at <<< "$plan")" \
        SEAT_THRESHOLD="$(jq -r '.base.spec | split(":")[2]' <<< "$plan")" \
        SEAT_RUNGS="$(jq -r '[.rungs[].spec] | join(" ")' <<< "$plan")" \
    --protected-parameters RECORD_DSN="$dsn" API_TOKEN="$token" \
        OUTPUTS_INFRA="$(terraform -chdir=infra output -json | base64 | tr -d '\n')" \
        OUTPUTS_STORE="$(jq -c 'del(.access_key, .postgres_dsn)' <<< "$store" | base64 | tr -d '\n')" \
    --timeout-in-seconds 3000 --output none
view=$(az vm run-command show -g "$rg" --vm-name "$vm" --name "$name" --instance-view \
    --query "instanceView.{exit: exitCode, out: output, err: error}" -o json)
az vm run-command delete -g "$rg" --vm-name "$vm" --name "$name" --yes --no-wait
jq -r '.out, .err' <<< "$view"
if [ "$(jq -r .exit <<< "$view")" != 0 ]; then
    echo "deploy.sh failed on the box (full log: /mnt/work/deploy.log there)" >&2
    exit 1
fi

mark() {
    az storage blob upload --account-name "$account" --account-key "$key" --container-name "$markers" \
        --name "$MARKERS/$1" --data "$2" --overwrite --output none --only-show-errors
}
mark console "$sha"
if [ "$kind" = full ]; then
    mark full "$sha"
    mark seat "$digest"
fi
echo "==> deployed ${sha:0:8} as $code ($kind)"
