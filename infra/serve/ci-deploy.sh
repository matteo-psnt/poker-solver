#!/usr/bin/env bash
# Bring the serving box up to HEAD. What `.github/workflows/deploy.yml` runs, on
# every push to main and once a day after the match window; it also runs from a
# laptop signed in with `az login` (`just serve-ci`).
#
# Two kinds of deploy. A change the CONSOLE alone runs is deployed at once, from
# the console's own tree on the box, at any hour. Anything else restarts the
# seat, so it waits for 05:00 UTC (deploy.sh says why) and the daily run picks it
# up. What is already deployed is read from two markers in the store, so both
# triggers do the same reconcile and neither needs to remember anything.
#
# The box is reached through Run Command, not SSH: its NSG admits one address on
# port 22, and this way GitHub holds no key. The settings the seat fields are not
# this script's: `keep` reads them off the box at deploy time.

set -euo pipefail

RELEVANT='^(src/|console/|infra/serve/|config/|pyproject\.toml$|uv\.lock$|justfile$)'
CONSOLE_ONLY='^(console/|src/interfaces/web/|infra/serve/console\.service$)'
MARKERS="serve-deploy"  # <diagnostics>/serve-deploy/{console,full}: the sha each kind last deployed

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

full_changes=$(changed_since "$(deployed full)" | grep -vE "$CONSOLE_ONLY" || true)
console_changes=$(changed_since "$(deployed console)")
hour=$(date -u +%H)

if [ -n "$full_changes" ] && [ "$((10#$hour))" -ge 5 ]; then
    kind=full
elif [ -n "$console_changes" ]; then
    kind=console
elif [ -n "$full_changes" ]; then
    echo "Seat code changed but it is ${hour}h UTC, inside the match window: the 05:10 run deploys it."
    exit 0
else
    echo "Nothing the box runs changed since the last deploy; the box is at ${sha:0:8}'s code."
    exit 0
fi
echo "==> $kind deploy of ${sha:0:8}"
[ "$kind" = console ] && [ -n "$full_changes" ] &&
    echo "    (seat code changed too: the 05:10 UTC run deploys the rest)"

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
sudo -u solver -H --preserve-env=RUN,CODE,RECORD_DSN,STORE_ACCOUNT,API_TOKEN,PUBLIC_HOST,OUTPUTS_INFRA,OUTPUTS_STORE bash -c "
    set -euo pipefail
    az login --identity --output none
    t=\$(mktemp -d)
    trap \"rm -rf \$t\" EXIT
    az storage blob download --auth-mode login --account-name \"\$STORE_ACCOUNT\" \
        --container-name code --name \"\$CODE.tar.gz\" --file \"\$t/code.tar.gz\" --output none
    tar -xzf \"\$t/code.tar.gz\" -C \"\$t\" infra/serve/deploy.sh
    bash \"\$t/infra/serve/deploy.sh\" \"\$RUN\" \"\$RECORD_DSN\" \"\$STORE_ACCOUNT\" < /dev/null
" > "$log" 2>&1
rc=$?
grep -v "ts\.net" "$log" | tail -c 3500
exit $rc'
[ "$kind" = full ] && run=keep || run=console
name="deploy-${sha:0:8}-$kind-$(date -u +%H%M%S)"
az vm run-command create -g "$rg" --vm-name "$vm" --name "$name" \
    --script "$script" --parameters RUN="$run" CODE="$code" STORE_ACCOUNT="$account" PUBLIC_HOST="$host" \
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

for marker in console $( [ "$kind" = full ] && echo full ); do
    az storage blob upload --account-name "$account" --account-key "$key" \
        --container-name "$markers" --name "$MARKERS/$marker" --data "$sha" --overwrite --output none --only-show-errors
done
echo "==> deployed ${sha:0:8} as $code ($kind)"
