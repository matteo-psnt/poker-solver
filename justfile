# Terraform lifecycle, the emergency stop, and shorthands for the Python CLI.
#
#   Terraform owns what EXISTS   (infra/*.tf)        -> just create
#   poker-solver-run owns what HAPPENS (submissions) -> just submit / jobs / fetch
#
# Dispatch used to live here as ~450 lines of `az` invocations. It is now
# `src/interfaces/cloud/`, where a leg spec is a typed object a test can look
# at rather than a shell string -- which is also how the hex-encoding of config
# overrides disappeared: the SDK takes name/value pairs, so nothing has to
# survive a `KEY=VALUE` parser. What remains below is what genuinely belongs in
# a task runner.
#
# The pool holds ZERO nodes at rest. You submit legs and walk away; nodes appear
# while work is queued and disappear when it drains.
#
# The durable share (infra/store) is a SEPARATE Terraform state in its own
# resource group: `just destroy` tears down compute and cannot touch the
# experiment record.

tf := "terraform -chdir=infra"
tfs := "terraform -chdir=infra/store"

_default:
    @just --list --unsorted

# --------------------------------------------------------------------------- #
# lifecycle (Terraform)
# --------------------------------------------------------------------------- #

# Show what Terraform would change, without changing it.
plan:
    {{tf}} init -input=false
    {{tf}} plan

# Create the durable share. Separate state, `prevent_destroy` -- run once, ever.
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
# emergency
# --------------------------------------------------------------------------- #

# STOP EVERYTHING. Terminates every job and forces the pool to zero nodes.
#
# The one command to reach for when something is wrong and you do not yet know
# what. Deliberately blunt: it kills running tasks rather than waiting for them,
# because the situation where you need this is the one where waiting is the
# problem. Anything a leg had published up to its last retained rung survives on
# the share, and `just submit <config> <to> --run <id>` picks it back up.
#
# THE ONLY RECIPE THAT DELIBERATELY DOES NOT USE THE PYTHON CLI, and the only
# one that reads no Terraform state. It must work when the checkout is broken,
# the venv is missing, or you are on a phone in Azure Cloud Shell -- so it takes
# its coordinates as arguments and shells `az` directly. It previously claimed
# that property while calling `just _login` and `terraform output -raw pool_id`,
# neither of which exists in Cloud Shell.
#
#   just panic poker-batch-rg pokerbatchus31321 train
[doc("STOP EVERYTHING. Args: resource-group batch-account pool-id")]
panic rg account pool:
    #!/usr/bin/env bash
    set -euo pipefail
    az batch account login -g "{{rg}}" -n "{{account}}"
    # `stop`, not `terminate`: there is no `az batch job terminate`, and the CLI
    # answers an unknown verb by printing help and exiting 1 -- which a `|| true`
    # then swallows, so panic reports success while stopping nothing.
    for job in $(az batch job list --query "[?state!='completed'].id" -o tsv); do
        echo "  stopping job $job"
        az batch job stop --job-id "$job" --terminate-reason "panic" || true
    done
    # Replace the target outright rather than only disabling autoscale:
    # disabling it leaves targetDedicatedNodes wherever it was.
    az batch pool autoscale disable --pool-id "{{pool}}" 2>/dev/null || true
    az batch pool resize --pool-id "{{pool}}" --target-dedicated-nodes 0 \
        --node-deallocation-option terminate
    echo "  pool {{pool}} resizing to 0. Re-arm autoscale with: just create"

# Alert before Azure charges can ever reach the credit card.
#
# There is no hard spending cap on this subscription and there cannot be one --
# the Azure spending limit is unavailable for Azure Plan pricing. A MasterCard is
# attached to the billing profile and cannot be detached via any API. So the card
# is guarded by watching the two routes to it: the credit running out/expiring,
# and charges that were never credit-eligible (Marketplace, support plans, Entra
# P1/P2) which bill the card TODAY while the balance sits untouched.
#
# THE MONTHLY BUDGET CANNOT SEE THE SECOND ROUTE. A budget measures burn, not
# eligibility, and stays quiet while a Marketplace charge goes to the card. That
# gap is the whole reason this exists.
#
# Exit codes are the interface -- 0 clear, 1 alert, 3 COULD NOT EVALUATE. A cron
# wrapper must treat 3 as a failure: a watchdog that dies silently on an expired
# token looks exactly like an all-clear.
#
# Stdlib only and no repo imports, so it runs from Azure Cloud Shell on a phone
# exactly like `just panic` does.
#
#   0 6 * * *  cd ~/Projects/poker-solver && python3 infra/credit_watch.py \
#                || echo "azure credit watch: exit $?" | mail -s "azure" you@example.com
#
# Raise --daily-burn whenever max_nodes or pool_vm_size goes up; it is the
# denominator of the runway number and is not derived automatically.
[doc("Alert before Azure charges can reach the credit card. 0 clear / 1 alert / 3 unevaluable.")]
credit-check *flags:
    @python3 infra/credit_watch.py {{flags}}

# --------------------------------------------------------------------------- #
# shorthands for `poker-solver-run`
# --------------------------------------------------------------------------- #
#
# Aliases, not logic. Every one of these is reachable directly as
# `uv run poker-solver-run <cmd>`; they exist so `just --list` still answers
# "what can I do here?" and so muscle memory keeps working. Anything that needs
# a flag not listed here should be run against the CLI directly.

# Start or continue a run, to an ABSOLUTE iteration target.
#
#   just submit quick_test 3000
#   just submit production 25000000 --experiment exp-7 --arm control
#   just submit "" 50000000 --run run-20260728_011716-ca70cf
[doc("Start/continue a run to an ABSOLUTE target. Args: config to [flags...]")]
submit config to *flags:
    uv run poker-solver-run submit --config "{{config}}" --to "{{to}}" {{flags}}

# Score a published run on the pool, one task per rung.
[doc("Score a published run. Args: run [flags...]")]
score run *flags:
    uv run poker-solver-run score --run "{{run}}" {{flags}}

# Every queued/running task on the pool.
jobs *flags:
    uv run poker-solver-run jobs {{flags}}

# Pool node counts, and the real cause of any allocation failure.
pool-status:
    uv run poker-solver-run pool-status

# Evaluate the deployed autoscale formula, errors included.
autoscale-check:
    uv run poker-solver-run autoscale-check

# A leg's log, from the share by default (survives node teardown).
[doc("Read a leg's log. Args: [flags...] e.g. --task <id> or --list")]
leg-log *flags:
    uv run poker-solver-run logs {{flags}}

# Bring published runs back. JSON only unless --full is passed.
fetch *flags:
    uv run poker-solver-run fetch {{flags}}

# Upload card abstractions to the share (~773 MB, one time).
push-data *flags:
    uv run poker-solver-run push-data {{flags}}

# Publish an immutable snapshot of the tree; echoes its id.
push-code:
    @uv run poker-solver-run push-code

# Verify a published static ladder, marking the rungs that load.
[doc("Verify a published ladder. Args: run config")]
repair-ladder run config:
    uv run poker-solver-run repair-ladder --run "{{run}}" --config "{{config}}"
