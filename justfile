# Fire-and-forget training on Azure Batch. Run `just` to list commands.
#
#   Terraform owns what EXISTS   (infra/*.tf)   -> just create
#   just      owns what HAPPENS  (submissions)  -> just submit / jobs / fetch
#
# The pool holds ZERO nodes at rest. You submit legs and walk away; nodes appear
# while work is queued and disappear when it drains. There is nothing to start,
# nothing to remember to stop, and no idle compute bill.
#
# The durable share (infra/store) is a SEPARATE Terraform state in its own
# resource group: `just destroy` tears down compute and cannot touch the
# experiment record.

tf := "terraform -chdir=infra"
tfs := "terraform -chdir=infra/store"

_default:
    @just --list --unsorted

# Data-plane Batch commands need an AAD login against the account. Shared-key auth
# is disabled in UserSubscription mode, so this is the only way in.
_login:
    @{{tf}} output -json | jq -r '"az batch account login -g \(.resource_group.value) -n \(.batch_account.value) --subscription \(.subscription_id.value)"' | bash

# --------------------------------------------------------------------------- #
# lifecycle (Terraform)
# --------------------------------------------------------------------------- #

# Show what Terraform would change, without changing it.
plan:
    {{tf}} init -input=false
    {{tf}} plan

# Create the durable share. Separate state, `prevent_destroy` — run once, ever.
store-create:
    {{tfs}} init -input=false
    {{tfs}} apply

# Create/update the Batch account, pool and guardrails. Safe to re-run.
# Requires store-create first: the pool mounts the share by name.
create:
    {{tf}} init -input=false
    {{tf}} apply
    @echo ""
    @echo "  next:  just push-data && just submit quick_test 3000"

# Delete the Batch account and pool. The share and every published run survive.
destroy:
    {{tf}} destroy

# --------------------------------------------------------------------------- #
# staging
# --------------------------------------------------------------------------- #

# Upload card abstractions to the share (~773 MB, one time).
#
# COPIED, never recomputed on a node: a recompute can change bucket assignments
# without changing card_abstraction_hash, so the provenance check guarding
# evaluation would pass over silently different buckets.
[doc("Upload card abstractions to the share (~773 MB, one time).")]
push-data:
    #!/usr/bin/env bash
    set -euo pipefail
    J=$({{tfs}} output -json)
    ACCT=$(jq -r '.storage_account.value' <<<"$J")
    SHARE=$(jq -r '.share_name.value' <<<"$J")
    KEY=$({{tfs}} output -raw access_key)
    # Azure Files will NOT create parent directories implicitly: uploading into a
    # path whose parent is absent fails with `ParentNotFound`, and `upload-batch`
    # does not make them for you. Every directory has to exist first.
    for d in combo_abstraction code archive; do
        az storage directory create --account-name "$ACCT" --account-key "$KEY" \
            --share-name "$SHARE" --name "$d" -o none 2>/dev/null || true
    done
    for src in data/combo_abstraction/*/; do
        [ -d "$src" ] || continue
        name=$(basename "$src")
        az storage directory create --account-name "$ACCT" --account-key "$KEY" \
            --share-name "$SHARE" --name "combo_abstraction/$name" -o none 2>/dev/null || true
        echo "  uploading $name"
        az storage file upload-batch --account-name "$ACCT" --account-key "$KEY" \
            --destination "$SHARE/combo_abstraction/$name" --source "$src" --no-progress
    done
    echo "  abstractions uploaded"

# Upload the working tree as an immutable, timestamped snapshot and echo its id.
#
# Pinned per submission on purpose: a push while a job is running must not change
# what that job is executing.
[doc("Upload the working tree as an immutable snapshot; echoes its id.")]
push-code:
    #!/usr/bin/env bash
    set -euo pipefail
    SNAP="code-$(date -u +%Y%m%d_%H%M%S)"
    J=$({{tfs}} output -json)
    ACCT=$(jq -r '.storage_account.value' <<<"$J")
    SHARE=$(jq -r '.share_name.value' <<<"$J")
    KEY=$({{tfs}} output -raw access_key)
    # ONE TARBALL, not a directory tree. Azure Files does not create parent
    # directories implicitly, so uploading a repo would mean pre-creating every
    # nested path (`ParentNotFound` otherwise) and paying a round-trip per file.
    # A sealed archive is one PUT, and it is atomic: a half-uploaded tarball is
    # simply absent rather than a partially-populated tree a node might run.
    TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
    # COPYFILE_DISABLE/--no-xattrs: macOS tar embeds xattrs and fflags that GNU
    # tar on the node only warns about, but the warnings bury real errors.
    COPYFILE_DISABLE=1 tar czf "$TMP/$SNAP.tar.gz" --no-xattrs \
        --exclude '.git' --exclude 'data' --exclude '.venv' --exclude '__pycache__' \
        --exclude 'node_modules' --exclude '.pytest_cache' --exclude '.ruff_cache' \
        --exclude '.mypy_cache' --exclude '.terraform' .
    az storage directory create --account-name "$ACCT" --account-key "$KEY" \
        --share-name "$SHARE" --name code -o none 2>/dev/null || true
    az storage file upload --account-name "$ACCT" --account-key "$KEY" \
        --share-name "$SHARE" --path "code/$SNAP.tar.gz" \
        --source "$TMP/$SNAP.tar.gz" --no-progress >/dev/null
    echo "$SNAP"

# --------------------------------------------------------------------------- #
# submission
# --------------------------------------------------------------------------- #

# Queue one task. Shared by `submit` (fresh) and `resume` (continue a run) so
# both go through identical staging and identical env wiring — the resume path
# being reachable is what makes Batch's automatic retry meaningful, and having it
# diverge from the fresh path is how it silently stops being exercised.
_task snap config to run="" experiment="" arm="" parent="" sets="":
    #!/usr/bin/env bash
    set -euo pipefail
    just _login
    POOL=$({{tf}} output -raw pool_id)
    # One job per day, holding that day's tasks. Created on demand; the `|| true`
    # is the second and later submissions finding it already there.
    JOB="poker-$(date -u +%Y%m%d)"
    # A job that has been STOPPED cannot take new tasks: `az batch task create`
    # answers JobCompleted. Since the id is per-day, one `just panic` -- or one
    # stranded task that had to be cleared at job level -- would otherwise block
    # every further submission until midnight UTC. Fall back to a suffixed id.
    STATE=$(az batch job show --job-id "$JOB" --query state -o tsv 2>/dev/null || echo absent)
    if [ "$STATE" != "absent" ] && [ "$STATE" != "active" ]; then
        JOB="$JOB-$(date -u +%H%M%S)"
        echo "  previous job is $STATE; using $JOB"
    fi
    # Explicit, not defaulted: these are billing controls, and a billing control
    # that depends on a service default is one upgrade away from not existing.
    #   retry 0  -- a task that fails deterministically must not re-burn a node
    #   P2D      -- bounds a whole day's submissions, not just one task
    az batch job create --id "$JOB" --pool-id "$POOL" \
        --job-max-task-retry-count 0 --job-max-wall-clock-time "P2D" 2>/dev/null || true
    LABEL="{{ if run == '' { config } else { run } }}"
    TASK="${LABEL//[^A-Za-z0-9_-]/-}-$(date -u +%H%M%S)-$RANDOM"
    # The wrapper lives INSIDE the code tarball, so the task command line has to
    # bootstrap it: extract, then run. Keep this to the bare minimum.
    #
    # Extract into a directory the TASK creates, so the task owns it. The start
    # task runs elevated, so anything it made is root-owned, and tar restoring
    # the archive root's mode onto it fails with `Cannot change mode`. Keying on
    # AZ_BATCH_TASK_ID also stops two tasks on one node sharing a tree.
    LEG='CODE_DIR=/mnt/work/code-$AZ_BATCH_TASK_ID && mkdir -p $CODE_DIR && tar xzf $AZ_BATCH_NODE_MOUNTS_DIR/shared/code/{{snap}}.tar.gz -C $CODE_DIR --no-same-owner --no-same-permissions && CODE_DIR=$CODE_DIR bash $CODE_DIR/infra/run_leg.sh'
    # maxWallClockTime is the ONLY thing standing between a hung task and
    # indefinite billing: the pool scales down on pending-task count, so a task
    # that never exits keeps its node alive forever. 24h is far above any real
    # leg and far below a month of compute.
    # HEX, not the raw string: `--environment-settings` parses KEY=VALUE, and a
    # config override IS a key=value pair, so the value contains `=` and the CLI
    # rejects it. Base64 would reintroduce the problem via its `=` padding; hex
    # has no special characters at all.
    SETS_HEX=$(printf '%s' "{{sets}}" | python3 -c "import sys; print(sys.stdin.read().encode().hex())")
    # Two nested ceilings, deliberately different:
    #   RUN_TIMEOUT  kills the TRAINING process and still runs the publish trap,
    #                so a wedged leg loses at most one rung interval.
    #   maxWallClock kills the TASK, losing whatever the trap did not reach.
    # RUN_TIMEOUT must stay comfortably below it or the cheap stop never fires.
    RUN_TIMEOUT="${RUN_TIMEOUT:-6h}"
    MAX_WALL="${MAX_WALL:-P1D}"
    az batch task create --job-id "$JOB" --task-id "$TASK" \
        --max-wall-clock-time "$MAX_WALL" \
        --command-line "/bin/bash -c '$LEG'" \
        --environment-settings \
            CODE_SNAPSHOT="{{snap}}" RUN_CONFIG="{{config}}" RUN_TO="{{to}}" \
            RUN_ID="{{run}}" RUN_EXPERIMENT="{{experiment}}" RUN_ARM="{{arm}}" \
            RUN_PARENT="{{parent}}" RUN_SETS_HEX="$SETS_HEX" \
            RUN_TIMEOUT="$RUN_TIMEOUT" \
        -o none
    echo "  ceilings: RUN_TIMEOUT=$RUN_TIMEOUT (training), maxWallClockTime=$MAX_WALL (task)"
    echo "  submitted $TASK to job $JOB — walk away; watch with: just jobs"

# Start a NEW run and train it to an absolute iteration count.
#
#   just submit quick_test 3000
#   just submit production 25000000 exp-7 control
#   just submit production 25000000 exp-7 variant:pruning "" solver__pruning=true
#   just submit production 25000000 exp-7 variant:x run-base-id      # fork lineage
[doc("Start a NEW run. Args: config to [experiment] [arm] [parent] [k=v...]")]
submit config to experiment="" arm="" parent="" *sets:
    #!/usr/bin/env bash
    set -euo pipefail
    SNAP=$(just push-code)
    echo "  code snapshot: $SNAP"
    just _task "$SNAP" "{{config}}" "{{to}}" "" "{{experiment}}" "{{arm}}" \
        "{{parent}}" "$(printf '%s\n' {{sets}})"

# Continue an EXISTING run to an absolute iteration target.
#
#   just resume run-20260728_011716-ca70cf 50000000
#
# Absolute, not "train N more": a retried task re-reads a newer checkpoint, so a
# relative target would compound. The run is fetched from the share's archive on
# the node, so this works against any published run — including one whose
# previous leg was killed partway.
[doc("Continue an EXISTING run to an absolute iteration target. Args: run to")]
resume run to:
    #!/usr/bin/env bash
    set -euo pipefail
    SNAP=$(just push-code)
    echo "  code snapshot: $SNAP"
    just _task "$SNAP" "" "{{to}}" "{{run}}" "" "" "" ""

# Every task and its state. The pool scales up on its own within ~5 minutes.
[doc("List every task and its state.")]
jobs:
    #!/usr/bin/env bash
    set -euo pipefail
    just _login
    for job in $(az batch job list --query "[].id" -o tsv); do
        echo "== $job"
        az batch task list --job-id "$job" \
            --query "[].{task:id, state:state, exit:executionInfo.exitCode, node:nodeInfo.nodeId}" \
            -o table 2>/dev/null | sed 's/^/  /'
    done

# stdout of one task (add `err` as the second arg for stderr).
job-log job task stream="out":
    #!/usr/bin/env bash
    set -euo pipefail
    just _login
    az batch task file download --job-id "{{job}}" --task-id "{{task}}" \
        --file-path "std{{stream}}.txt" --destination /dev/stdout 2>/dev/null

# Cancel a task. Its wrapper publishes what it has before exiting, so progress
# up to the last retained rung survives on the share.
[doc("Cancel a task; its partial progress is published first.")]
cancel job task:
    #!/usr/bin/env bash
    set -euo pipefail
    just _login
    az batch task stop --job-id "{{job}}" --task-id "{{task}}"
    echo "  terminated {{task}} — partial progress is on the share"

# --------------------------------------------------------------------------- #
# status + results
# --------------------------------------------------------------------------- #

# Node counts, and — critically — the REAL reason behind any allocation failure.
#
# Batch reports every allocation problem as a generic `AllocationFailed`; the
# actual cause (Gen1-vs-Gen2 image, a policy denial, quota) is escaped JSON inside
# resizeErrors. Reading it is the difference between a one-line fix and an
# afternoon — it is how the Gen2 requirement was found.
[doc("Pool node counts, and the REAL cause of any allocation failure.")]
pool-status:
    #!/usr/bin/env bash
    set -euo pipefail
    just _login
    POOL=$({{tf}} output -raw pool_id)
    az batch pool show --pool-id "$POOL" \
        --query "{state:allocationState, current:currentDedicatedNodes, target:targetDedicatedNodes}" -o table
    ERR=$(az batch pool show --pool-id "$POOL" --query "resizeErrors" -o json)
    if [ "$ERR" != "[]" ] && [ -n "$ERR" ]; then
        echo "  RESIZE ERRORS — the real cause is inside valuesProperty:"
        jq -r '.[] | "   code: \(.code)\n   \(.valuesProperty[]? | select(.name|test("Json$")) | .value)"' <<<"$ERR"
    fi
    echo "  cost: $({{tf}} output -raw hourly_cost) (pool is 0 nodes at rest)"

# STOP EVERYTHING. Terminates every job and forces the pool to zero nodes.
#
# The one command to reach for when something is wrong and you do not yet know
# what. Deliberately blunt: it kills running tasks rather than waiting for them,
# because the situation where you need this is the one where waiting is the
# problem. Anything a leg had published up to its last retained rung survives on
# the share, and `just resume <run> <to>` picks it back up.
#
# Runs from a phone via Azure Cloud Shell; nothing here needs this repo.
[doc("STOP EVERYTHING: terminate all jobs and force the pool to zero nodes.")]
panic:
    #!/usr/bin/env bash
    set -euo pipefail
    just _login
    POOL=$({{tf}} output -raw pool_id)
    # `stop`, not `terminate`: there is no `az batch job terminate`, and the CLI
    # answers an unknown verb by printing help and exiting 1 -- which the `|| true`
    # then swallowed, so panic reported success while stopping nothing. Found the
    # hard way on a task stranded by an unusable node.
    for job in $(az batch job list --query "[?state!='completed'].id" -o tsv); do
        echo "  stopping job $job"
        az batch job stop --job-id "$job" --terminate-reason "panic" || true
    done
    # Replace the formula outright rather than disabling autoscale: disabling it
    # leaves targetDedicatedNodes wherever it was.
    az batch pool autoscale disable --pool-id "$POOL" 2>/dev/null || true
    az batch pool resize --pool-id "$POOL" --target-dedicated-nodes 0 \
        --node-deallocation-option terminate
    echo "  pool $POOL resizing to 0. Re-arm autoscale with: just create"

# Evaluate the deployed autoscale formula server-side and print BOTH its
# variables and any error.
#
# The error half is the point. An invalid or throwing formula still returns
# partial `results`, so printing results alone makes a broken formula look
# healthy -- which is exactly how a `#` comment (Batch wants `//`) and a
# one-argument GetSample both went unnoticed while the pool quietly stopped
# scaling. Run this after every formula change.
[doc("Evaluate the deployed autoscale formula on the live pool, errors included.")]
autoscale-check:
    #!/usr/bin/env bash
    set -euo pipefail
    just _login
    POOL=$({{tf}} output -raw pool_id)
    OUT=$(mktemp); trap 'rm -f "$OUT"' EXIT
    az batch pool autoscale evaluate --pool-id "$POOL" \
        --auto-scale-formula "$({{tf}} output -raw autoscale_formula)" -o json > "$OUT"
    if jq -e '.error' "$OUT" >/dev/null; then
        jq -r '"  ERROR: \(.error.code)", (.error.values[]? | "    \(.value)")' "$OUT"
        echo "  (results below are PARTIAL — the formula did not fully evaluate)"
    else
        echo "  no error"
    fi
    jq -r '(.results // "") | split(";")[] | select(length > 0) | "    " + .' "$OUT"

# Bring published runs and eval records back, then rebuild the ledger.
#
# `ledger --rebuild` is what makes this safe with several legs finishing at once:
# each eval wrote its complete row into its own run directory, so the index is
# derived rather than appended to by competing writers.
[doc("Bring published runs back and rebuild the ledger.")]
fetch:
    #!/usr/bin/env bash
    set -euo pipefail
    J=$({{tfs}} output -json)
    ACCT=$(jq -r '.storage_account.value' <<<"$J")
    SHARE=$(jq -r '.share_name.value' <<<"$J")
    KEY=$({{tfs}} output -raw access_key)
    mkdir -p data/runs
    az storage file download-batch --account-name "$ACCT" --account-key "$KEY" \
        --source "$SHARE/archive" --destination data/runs --no-progress >/dev/null 2>&1 || {
        echo "  nothing published yet"; exit 0; }
    echo "  fetched into data/runs"
    uv run poker-solver-run ledger --rebuild --limit 10

# Print a leg's published log. Unlike `job-log`, this survives the node.
#
# Batch keeps task stdout/stderr ON THE NODE, and the pool scales to zero within
# minutes of a task ending -- so `job-log` returns NodeNotFound for exactly the
# failed legs you most need to read. run_leg.sh copies its log to the share on
# every publish; this reads that copy.
[doc("Print a leg's log from the share (survives node teardown). Args: task")]
leg-log task:
    #!/usr/bin/env bash
    set -euo pipefail
    J=$({{tfs}} output -json)
    ACCT=$(jq -r '.storage_account.value' <<<"$J")
    SHARE=$(jq -r '.share_name.value' <<<"$J")
    KEY=$({{tfs}} output -raw access_key)
    TMP=$(mktemp); trap 'rm -f "$TMP"' EXIT
    az storage file download --account-name "$ACCT" --account-key "$KEY" \
        --share-name "$SHARE" --path "logs/{{task}}.log" --dest "$TMP" \
        --no-progress -o none 2>/dev/null || { echo "  no published log for {{task}}"; exit 0; }
    tr '\r' '\n' < "$TMP" | grep -v "Training batches:" | tail -80

# Every leg log on the share, newest last.
[doc("List published leg logs.")]
leg-logs:
    #!/usr/bin/env bash
    set -euo pipefail
    J=$({{tfs}} output -json)
    ACCT=$(jq -r '.storage_account.value' <<<"$J")
    SHARE=$(jq -r '.share_name.value' <<<"$J")
    KEY=$({{tfs}} output -raw access_key)
    az storage file list --account-name "$ACCT" --account-key "$KEY" \
        --share-name "$SHARE" --path logs -o tsv --query "[].name" 2>/dev/null || echo "  none"
