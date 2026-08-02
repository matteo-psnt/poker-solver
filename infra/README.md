# Fire-and-forget training on Azure Batch

Submit experiment legs and walk away. The pool holds **zero nodes at rest** —
nodes appear while work is queued and disappear when it drains. There is nothing
to start, nothing to remember to stop, and no idle compute bill.

**Terraform owns what exists** (Batch account, pool, guardrails) — `infra/*.tf`.
**`just` owns what happens** (submissions) — `justfile` at the repo root. Jobs and
tasks are deliberately *not* in Terraform: `azurerm_batch_job` exposes no useful
properties and tasks have no resource at all. A submission is a runtime act.

## Two states, on purpose

| | `infra/` | `infra/store/` |
|---|---|---|
| Holds | Batch account, pool, guardrails | storage account + file share |
| Lifetime | disposable — `just destroy` | **durable**, `prevent_destroy` |
| Resource group | `poker-batch-rg` | `poker-solver-store-rg` |

Separate root modules mean separate state, so tearing down compute **cannot**
reach the experiment record.

## Measured constraints — do not "simplify" these away

Three of these cost real time to discover. All are verified against this
subscription, not assumed.

1. **Batch's default (BatchService) mode gives 6 dedicated cores** — less than one
   node. The account must be **UserSubscription** mode, which allocates VMs into
   *this* subscription and draws on its 65-vCPU quota. Proven: a pool took the
   quota 0 → 8 → 0. This also keeps everything on the sponsorship credit.
2. **`Dalds_v6` is not Batch-supported** in swedencentral; `Dals_v6` is. Same AMD
   Genoa, but **no local temp disk** — hence the attached `data_disks` volume the
   start task formats and mounts at `/mnt/work`.
3. **`als_v6` is Gen2-only.** A Gen1 image fails as `AllocationFailed / cannot boot
   Hypervisor Generation '1'`. The image SKU must stay `*-gen2`.
4. **The SKU policy must list the pool's family.** `allowed_vm_skus` denies at
   request time, so a stale `Dalds_v6` entry would make every allocation fail with
   an opaque error that looks nothing like a policy problem.

## One-time setup

The Batch service principal needs **Contributor at subscription scope** so
UserSubscription pools can create VMs. This is deliberately **not** Terraform-
managed: it is a one-time tenant-level act, and putting a subscription-scope role
assignment in the same state as the compute would let a `destroy` revoke it.

```bash
az provider register -n Microsoft.Batch
az role assignment create --assignee-object-id 2736183d-125f-4bd0-8cc8-4f1189c65986 \
  --assignee-principal-type ServicePrincipal --role Contributor \
  --scope /subscriptions/<sub>
```

## Setup

Needs `terraform`, `az` and `just` locally, plus `az login` for the credential.

```bash
az login
just store-create   # the durable share — once, ever
just create         # Batch account, pool, guardrails
just push-data      # card abstractions to the share (~773 MB, one time)
```

## Daily use

Dispatch is `poker-solver-run`, a Python CLI over the Batch SDK
(`src/interfaces/cloud/`). The `just` recipes below are one-line aliases kept
for discoverability; anything needing a flag they do not forward should be run
against the CLI directly.

```bash
poker-solver-run submit --config quick_test --to 3000                  # smoke
poker-solver-run submit --config production --to 25000000 \
    --experiment exp-7 --arm control                                   # an arm
poker-solver-run submit --config production --to 25000000 \
    --experiment exp-7 --arm variant:pruning --set solver__pruning=true
poker-solver-run submit --run run-20260728_011716-ca70cf --to 50000000 # continue

poker-solver-run jobs                 # live tasks (--all for finished jobs)
poker-solver-run pool-status          # nodes + the REAL cause of any allocation failure
poker-solver-run logs --task <task>   # the published leg log; --list to enumerate
poker-solver-run logs --task <task> --source node --job <job>   # live, node-side
poker-solver-run cancel --job <job> --task <task>
poker-solver-run score --run <id> --at 10000000,20000000 -- --br-flops 8
poker-solver-run fetch                # JSON record back + `ledger --rebuild`
poker-solver-run fetch --run <id> --full   # ...including checkpoint data

poker-solver-run report --experiment exp-7
poker-solver-run curve --run <id>
poker-solver-run promote --run <winner> --rationale "..."
```

Two things worth knowing at the seams:

- **`score` passthrough needs a `--` separator.** `-- --br-flops 8`, not
  `--br-flops 8`: argparse rejects a bare unknown option as an argument of
  `score` itself rather than handing it to the passthrough.
- **`fetch` is metadata-only by default.** Analysis commands read nothing but
  small JSON, and the checkpoints beside them are ~540 MB each. `--full` obeys
  `CHECKPOINT.json`: a snapshot directory the manifest does not name is
  unfinished by construction and is skipped, which is what stops a killed
  task's orphans from being pulled down and then cached forever.

`to` is an **absolute** iteration target. That is what makes Batch's automatic
retry safe: a retried task re-reads a newer checkpoint and converges on the same
endpoint instead of compounding an increment.

## How a leg survives being killed

`infra/run_leg.sh` publishes to the share **every time a retained checkpoint rung
appears**, and again on any exit — success, failure, or cancellation.

The node's disk is ephemeral. Publishing only at the end would mean an OOM or a
`maxWallClockTime` kill destroys a multi-hour leg entirely, which is the same
failure as `REBUILD.md` Decision 5 in a new form. With rung publishing, a killed
task loses at most one rung interval and the next attempt resumes from the last
published rung.

## How you find out *why* a leg died

`just legs`.

A run's `.run.json` records what a *living* process did. It structurally cannot
record how an attempt died: a container killed by the OOM killer, by
`maxWallClockTime`, or by losing its node is gone before it can write anything.
Batch sees those deaths — but retains them for far less time than the run lives.

So the record is written from both sides, into `<share>/legs/`:

- **The node's own account.** `run_leg.sh` writes `<task>.<attempt>.start.json`
  at entry and `<task>.<attempt>.exit.json` from the EXIT trap. This covers every
  death the shell survives, and is the only side that can tell a *hang*
  (`RUN_TIMEOUT`, exit 124) from an *OOM* (exit 137) from a *cancel* (SIGTERM).
  Batch reports all three as `failure`.
- **Batch's account.** `just legs` asks about legs still stuck at `started` —
  precisely the ones whose trap never ran — and writes `<task>.observed.json`.

Numbered by attempt because a Batch retry reuses the task id and Batch describes
only the latest attempt; without the number the retry would erase the failed
attempt that caused it. Separate start/exit files because `write_text` truncates,
so a kill mid-write would otherwise make the leg vanish from the listing
entirely — in exactly the SIGKILL window this exists for.

`src/shared/leg_log.py` imports **only the standard library**, on purpose:
`run_leg.sh` calls it with the node's system `python3` before `uv sync` has run.
A test enforces that.

`just legs --skip-reconcile` reads the share without querying Batch, and
`just leg-log <task> errors` filters a published log to WARN/ERROR.

## What must never go on the share

**Active run directories.** A checkpoint is ~2,000 small files and the read path
mmaps them; SMB turns every page fault into a network round-trip and offers no
atomic replace. Runs live on the node's `/mnt/work` data disk and are *published*
to the share. The card abstraction is likewise copied share→local at node start.

There is a second reason: a run directory has exactly one writer for its whole
life, and that invariant is what keeps the checkpoint layer safe — its prune step
deletes by glob, so two writers in one run dir destroy each other's snapshots. One
task owning one run preserves it.

## What actually protects you from a bill

Be clear-eyed: **there is no hard spending cap.** The subscription reports
`spendingLimit: Off` and there is no supported way to turn it on for a
Sponsorship offer. Azure budgets are *alerts*, not caps, and cost data lags
several hours.

Every control bounds either the RATE of spend or the DURATION of one piece of
work. Read them that way — most of them do not stop anything by themselves.

**Rate ceiling (strongest):**

1. **`max_nodes`.** Nothing can burn faster than `max_nodes` x the per-node rate.
   At the default 2 x D8als_v6 that is ~$0.80/hr, ~$19/day. This is the control
   that makes the worst case finite, and it is why the default is deliberately
   low until a real leg has run end to end.
2. **Policy denials — preventive, zero lag.** A VM outside the SKU or region
   whitelist is rejected at request time.

**Duration bounds:**

3. **`--max-wall-clock-time P1D` per task.** A hung task self-terminates after a
   day instead of holding its node forever.
4. **`--job-max-wall-clock-time P2D`, `--job-max-task-retry-count 0`.** Bounds a
   whole day's submissions, and stops a deterministically-failing task from
   re-burning a node. Both set explicitly rather than relying on service
   defaults — a billing control that depends on a default is one upgrade away
   from not existing.

**Acts without you:**

5. **The stall clause in the autoscale formula.** If average CPU stays below
   `stall_cpu_floor` for `stall_window_minutes` while nodes are allocated, the
   pool force-deallocates to zero. A deadlocked trainer sits near 0% CPU; a
   healthy one pegs its cores.

   Two details make this work rather than merely look good. `$NodeDeallocationOption`
   is `terminate` **only** in the stalled branch and `taskcompletion` otherwise —
   with `taskcompletion` alone the clause would be inert, because it waits for a
   task that by definition never finishes. And the formula requires >50% sample
   coverage before acting, so a freshly-created pool with no CPU history cannot
   scale itself down before it starts.

   **Verify it before trusting it:** `just autoscale-check` evaluates the live
   formula server-side and prints `cpuAvg` and `stalled`. The clause assumes
   `$CPUPercent` is a 0-1 fraction; if it is not, the threshold never fires and
   the backstop is silently absent.

**Alerts only — these stop nothing:**

6. **Budget alerts** at 50/75/90/100% actual plus 100% forecast. Forecast is the
   one that warns you while there is still time to act. At ~$19/day the default
   $250 budget is about two weeks of a total runaway.

**When something is wrong:** `just panic <rg> <account> <pool>` terminates every job and forces the
pool to zero, killing running tasks rather than waiting. Whatever a leg published
up to its last retained rung survives, and
`poker-solver-run submit --run <id> --to <n>` picks it up.

It takes its coordinates as arguments deliberately. `panic` is the one recipe
that uses neither the Python CLI nor Terraform state, so it still works when the
checkout is broken, the venv is missing, or you are on a phone in Azure Cloud
Shell — the same property `just credit-check` has. It previously *claimed* that
while calling `just _login` and `terraform output -raw pool_id`, neither of
which exists in Cloud Shell.

The only *absolute* control is the billing account having no payment method —
and that is **not** the case here: the account is a `MicrosoftCustomerAgreement`
(Individual), which normally has a card attached. So **if the sponsorship credit
is exhausted or expires, real charges follow.** Confirm the balance and end date
at <https://www.microsoftazuresponsorships.com/> before relying on this.


## Region and SKU availability

Quota is not availability. This subscription has 65 vCPUs of quota and still
cannot launch most SKUs in most regions. Establishing what works took three
failed deployments:

| Attempt | Result |
|---|---|
| westeurope | `RequestDisallowedByAzure` — region not accepting new customers |
| northeurope, `D4s_v3` | `SkuNotAvailable` — capacity restrictions, all zones |
| swedencentral, `D4s_v3` | `SkuNotAvailable` — same |
| swedencentral, `D8alds_v6` | works |

The `Fsv2` and `Dsv3` families are `NotAvailableForSubscription` in every region
checked. `Dalds_v6` (AMD Genoa) is available and comparably priced.

**The diagnostic trap:** plain `az vm list-skus` *omits* restricted SKUs instead
of flagging them, so an empty result looks identical to "available" — which I
initially misread. Always pass `--all` and inspect `restrictions`:

```bash
az vm list-skus -l <region> --size <sku> --all --query "[].restrictions[].reasonCode" -o tsv
```

Everything this subscription can launch in a region:

```bash
az vm list-skus -l swedencentral --resource-type virtualMachines --all -o json \
  | python3 -c "import json,sys; print('\n'.join(s['name'] for s in json.load(sys.stdin) if not s.get('restrictions')))"
```

Capacity restrictions change without notice, so if `just create` starts failing
with `SkuNotAvailable`, re-run that query rather than assuming the account broke.

## Troubleshooting

**`Permission denied (publickey)` despite the key existing.** With several keys
in `~/.ssh` or an agent, ssh offers them one at a time and the server closes the
connection after `MaxAuthTries`. The justfile pins the identity
(`IdentitiesOnly=yes -i`). Override with `POKER_SSH_KEY=...`.

**rsync fails with a usage dump.** macOS ships `openrsync`, which does not
support `--info` (rsync 3.1+) and will not create missing *parent* directories.
The justfile avoids both; if you rsync by hand, `mkdir -p` the remote path first.

**Policy denies a SKU you just whitelisted.** Assignment changes take a few
minutes to propagate. Wait and retry before assuming the whitelist is wrong.

**Bare `az vm create` prints a Python traceback** ending in "The content for this
response was already consumed". That is an `az` CLI bug in its error handler; the
real Azure error is *above* the traceback. Terraform surfaces these properly,
which is part of why provisioning moved there.

## State

Terraform state is local (`infra/terraform.tfstate`, `infra/store/terraform.tfstate`) and gitignored — it can
contain resource detail you would not want committed. Solo use makes a remote
backend unnecessary; if this ever becomes shared, move state to an Azure Storage
backend before a second person runs `apply`.

`infra/.terraform.lock.hcl` *is* committed, like `uv.lock`: it pins the `azurerm`
provider version so a fresh `terraform init` elsewhere resolves the same one.

## Teardown

```bash
just destroy   # Batch account + pool. The share and every published run SURVIVE
```

`just destroy` removes only compute. The share is a separate state with
`prevent_destroy`, so published runs and the eval record survive it.
