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

Needs `terraform`, `az`, `just` and `jq` locally.

```bash
az login
just store-create   # the durable share — once, ever
just create         # Batch account, pool, guardrails
just push-data      # card abstractions to the share (~773 MB, one time)
```

## Daily use

```bash
just submit quick_test 3000                                   # smoke
just submit production 25000000 exp-7 control                 # an arm
just submit production 25000000 exp-7 variant:pruning solver__pruning=true

just jobs           # task states
just pool-status    # node counts + the REAL cause of any allocation failure
just job-log poker-20260728 <task>       # stdout   (add `err` for stderr)
just fetch          # published runs back, then `ledger --rebuild`

uv run poker-solver-run report --experiment exp-7
uv run poker-solver-run curve --run <id>
uv run poker-solver-run promote --run <winner> --rationale "..."
```

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

Be clear-eyed: **there is no hard spending cap, and there cannot be one.** The
subscription reports `spendingLimit: Off`. The blocker is the *pricing*, not the
sponsorship: the Azure spending limit "isn't available for subscriptions with
commitment plans or with pay-as-you-go pricing", which is what a Microsoft Azure
Plan under MCA is. It is not a setting anyone forgot to turn on. Azure budgets
are *alerts*, not caps, and cost data lags several hours.

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

**When something is wrong:** `just panic` terminates every job and forces the
pool to zero, killing running tasks rather than waiting. Whatever a leg published
up to its last retained rung survives, and `just resume <run> <to>` picks it up.

**Alerts that watch the card specifically:**

7. **`just credit-check`** (`infra/credit_watch.py`). Budgets measure burn; this
   measures whether the *card* is reachable. Two routes, both watched: credit
   depletion/expiry, and charges that were never credit-eligible. Exit codes are
   the interface — 0 clear, 1 alert, **3 could not evaluate**, which a caller must
   treat as failure rather than as an all-clear.

### The payment method, and why it cannot be removed here

The only *absolute* control is the billing account having no payment method. That
is **not** achievable on this account, and the reasons are worth recording so it
is not re-litigated:

- MasterCard `...6136` is attached to billing profile `EPHL-5ZRZ-BG7-PGB`, which
  bills every resource here.
- **There is no API operation to detach a payment-method link from a billing
  profile.** The Billing API's only delete is `Delete By User`, which removes a
  method *owned by the caller* — and both `paymentMethods` at user scope and at
  billing-account scope return `[]`. Detaching is portal-only.
- The portal blocks deleting a card that is the *default* method for a profile,
  which this one is.

Note it would be a bad trade even if it worked: an unpayable card does not cap
spend, it converts spend into unpaid debt. The subscription is disabled, VMs
deallocate, **data is deleted 90 days after service ends**, and the balance is
still owed. Prepaid and virtual cards are rejected outright as payment
instruments, so that variant does not start either.

So **if the credit is exhausted or expires, real charges follow.** Confirm the
balance and end date with `just credit-check` — *not* at
microsoftazuresponsorships.com, which is the classic-sponsorship portal and shows
nothing for this account. This grant is an **MCA credit lot** (`Azure for
startups credit`, $10,000, expiring 2028-07-26) read from the Consumption `lots`
and `credits/balanceSummary` APIs, which are authoritative here.

One trap in reading it: `currentBalance` is the last *closed* balance and reads
as the full grant until an invoice issues. The live number is `estimatedBalance`.


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
