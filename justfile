# Terraform lifecycle, the emergency stop, and shorthands for the Python CLI.
#
#   Terraform owns what EXISTS   (infra/*.tf)     -> just create
#   poker-solver owns what HAPPENS (submissions)  -> just submit / just cli status
#
# Dispatch used to live here as ~450 lines of `az` invocations. It is now
# `src/interfaces/cloud/`, where a task spec is a typed object a test can look
# at rather than a shell string -- which is also how the hex-encoding of config
# overrides disappeared: the SDK takes name/value pairs, so nothing has to
# survive a `KEY=VALUE` parser. The node-side wrapper went the same way, to
# `src/shared/cloudtask/node/`. What remains below is what genuinely belongs in a task
# runner: Terraform lifecycle, the emergency stop, and passthrough aliases.
#
# The pool holds ZERO nodes at rest. You submit tasks and walk away; nodes appear
# while work is queued and disappear when it drains.
#
# The durable share (infra/store) is a SEPARATE Terraform state in its own
# resource group: `just destroy` tears down compute and cannot touch the
# experiment record.

tf := "terraform -chdir=infra"
tfs := "terraform -chdir=infra/store"
tfv := "terraform -chdir=infra/serve"

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
    @echo "  next:  just cli push-data && just submit quick_test 3000"

# Delete the Batch account and pool. The share and every published run survive.
destroy:
    {{tf}} destroy

# Create the box that serves a trained run for reading. Its own state, so
# `destroy` above cannot reach it -- and it is NOT part of the pool, because the
# pool is for work that finishes and a server never does.
serve-create:
    {{tfv}} init -input=false
    {{tfv}} apply
    @echo ""
    @echo "  next:  just serve-tunnel, then just serve-ssh to start a run"

# Show what the serving box would change, without changing it.
serve-plan:
    {{tfv}} init -input=false
    {{tfv}} plan

# The forward that reaches the blueprint server. It binds loopback on the box,
# so this is the only route to it -- run it in a second terminal and leave it.
serve-tunnel:
    @sh -c "$({{tfv}} output -raw tunnel)"

# A shell on the serving box, for starting the server against a run.
serve-ssh:
    @sh -c "$({{tfv}} output -raw ssh)"

# Delete the serving box. Nothing on it is a source of truth; it holds copies.
serve-destroy:
    {{tfv}} destroy

# --------------------------------------------------------------------------- #
# emergency
# --------------------------------------------------------------------------- #

# STOP EVERYTHING. Terminates every job and forces the pool to zero nodes.
#
# The one command to reach for when something is wrong and you do not yet know
# what. Deliberately blunt: it kills running tasks rather than waiting for them,
# because the situation where you need this is the one where waiting is the
# problem. Anything a task had published up to its last retained rung survives on
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
# shorthands for `poker-solver`
# --------------------------------------------------------------------------- #
#
# ONLY recipes that RESHAPE the call survive here. `submit`, `score` and
# `score` turns positionals into flags, which is a real saving; `cli` is
# the escape hatch for everything else.
#
# The nine pure passthroughs that used to sit here (`status`, `jobs`, `tasks`,
# `logs`, `cancel`, `pool-status`, `autoscale-check`, `push-code`, `push-data`,
# `ledger`) are gone. They retyped a command without changing it, and they
# answered "what can I do here?" with 13 of 26 -- a hand-maintained subset that
# `poker-solver --help` already answers in full and cannot drift from.
# `just fetch` outlived the command it called by weeks for exactly that reason;
# `tests/interfaces/commands/test_justfile_aliases.py` still guards what is left.

# Any subcommand, with any flags. The escape hatch, so the list above does not
# have to grow a recipe per flag: `just cli ledger --experiment exp-7`.
[doc("Run any poker-solver subcommand. Args: cmd [flags...]")]
cli *args:
    uv run poker-solver {{args}}

# Start or continue a run, to an ABSOLUTE iteration target.
#
# `--config` is required even when continuing: it builds the tree and the
# solver, and the checkpoint stores neither.
#
#   just submit quick_test 3000
#   just submit production 25000000 --experiment exp-7 --arm control
#   just submit production 50000000 --run run-20260728_011716-ca70cf
[doc("Start/continue a run to an ABSOLUTE target. Args: config to [flags...]")]
submit config to *flags:
    uv run poker-solver submit --config "{{config}}" --to "{{to}}" {{flags}}

# Score a published run on the pool, one task per rung.
[doc("Score a published run. Args: run [flags...]")]
score run *flags:
    uv run poker-solver score --run "{{run}}" {{flags}}

# --- console ---------------------------------------------------------------- #
# The web console. Its toolchain is npm and lives entirely under `console/`;
# nothing here touches the Python environment.
#
# Only the recipes that DO something npm cannot are here. Installing is
# `npm --prefix console ci` and checking is `npm --prefix console run ci`;
# wrapping either in a recipe saves no typing and gives the same command a
# second name to drift from.

# Build the console into console/dist, which `serve` looks for.
console-build:
    npm --prefix console run build

# Build, then serve on http://127.0.0.1:8765.
console: console-build
    uv run poker-solver serve

# The server is SUPERVISED, not merely backgrounded. `serve &` plus a trap threw
# away its exit status and interleaved its stderr into vite's, so a dead server
# was observable only as a vite `ECONNREFUSED` on /api -- with no way to tell an
# OOM kill (137) from a stray Ctrl-C reaching the shared process group (130)
# from a traceback from an `[Errno 48] Address already in use` two seconds after
# startup. Same lesson `src/shared/cloudtask/node/process.py` learned on the cloud side: a
# death the log cannot record needs its own account.
#
# (The blank line below is load-bearing: `just --list` takes the LAST contiguous
# comment block as a recipe's summary, so without it the list reads "death the
# log cannot record needs its own account".)

# Dev: Vite on :5173 with hot reload, proxying /api to a real server on :8765.
console-dev:
    #!/usr/bin/env bash
    set -uo pipefail

    port=8765
    log=/tmp/poker-console-server.log
    : >"$log"
    echo "[console-dev] server log: $log"

    # A subshell, because only a parent can `wait` on the server and read its
    # status. It keeps the pid so its own TERM handler can pass the signal on --
    # without that, the EXIT trap below kills the supervisor and leaves the
    # server holding :8765.
    (
      trap 'kill "${child:-}" 2>/dev/null; exit 0' TERM INT
      consecutive=0
      while :; do
        started=$SECONDS
        uv run poker-solver serve --port "$port" >>"$log" 2>&1 &
        child=$!
        rc=0; wait "$child" || rc=$?
        lived=$(( SECONDS - started ))
        printf '\n[console-dev] server exited rc=%d after %ds\n' "$rc" "$lived" >&2
        tail -n 20 "$log" | sed 's/^/[server] /' >&2
        # A server that cannot bind dies in the same second every time, so
        # respawning forever would only scroll the reason off the screen.
        if [ "$lived" -lt 10 ]; then consecutive=$(( consecutive + 1 )); else consecutive=0; fi
        if [ "$consecutive" -ge 3 ]; then
          printf '[console-dev] died %d times in a row -- stopping. Full log: %s\n' "$consecutive" "$log" >&2
          # This process group is the `just` job and nothing else, so it takes
          # vite down too and hands the terminal back, rather than leaving a
          # console that can only render proxy errors.
          kill 0
        fi
        printf '[console-dev] restarting in 2s\n' >&2
        sleep 2
      done
    ) &
    supervisor=$!
    trap 'kill "$supervisor" 2>/dev/null || true' EXIT
    npm --prefix console run dev
