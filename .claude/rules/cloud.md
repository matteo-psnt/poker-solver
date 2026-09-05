---
paths:
  - "src/interfaces/cloud/**"
  - "src/shared/cloudtask/**"
  - "src/shared/task_history.py"
  - "infra/**"
  - "tests/interfaces/cloud/**"
  - "tests/shared/cloudtask/**"
---

# Azure dispatch, the node wrapper, and the share

`src/interfaces/cloud/` is split by what the code TALKS TO: `tasks/` (Batch),
`store/` (the share and the blob containers), `cost/` (Cost Management), with `config.py` and
`serve_box.py` above them. It lives under `interfaces` so nothing in
`pipeline`/`engine`/`core` can reach Azure.

- **Auth is `AzureCliCredential`, never `DefaultAzureCredential`.** The default
  chain probes the link-local IMDS address, which on a laptop hangs instead of
  refusing: >120 s per call versus 1.3 s, and it looks like a Batch outage.
- **Read cost is a maintained property**, pinned as call counts in
  `tests/interfaces/cloud/test_read_cost.py` because latency is invisible in a
  test and enormous in practice. Never list tasks for a job you will discard;
  issue independent round trips together; never sync `keys-*` tables.
- **`src/shared/cloudtask/` is STDLIB ONLY.** The node runs it before
  `uv sync`. The interpreter is 3.13 (the start task installs it), so there is
  no old-language floor — only the import floor, enforced fail-closed by
  `test_imports.py` and `test_node_interpreter.py`. It lives under `shared`
  because `pipeline` reads the task record and may not import `interfaces`.
  `task_history.py` is deliberately outside the package: it is the reading
  half and runs only on a laptop.
- **A Batch task's state is classified ONCE, in `src/shared/task_states.py`.**
  `Phase` and `Outcome` ride on the payload; nothing downstream parses an
  Azure enum string. `OCCUPIES_A_NODE` excludes `queued` and `IN_FLIGHT`
  includes it — cost accounting must use the former, or queue time is billed
  as node time (it was, for 455 of 718 node-hours).
- **Exit 124 and 137 are different causes.** 124 is the guard's deadline (a
  hang), 137 is SIGKILL from outside (the OOM killer). A wrong terminal cause
  is permanent: it suppresses reconciliation. `poker-solver tasks` is where a
  death is explained; the run log cannot record one.
- **Profiling a running task**: `poker-solver profile --task <id>` drops a
  request file on the share, the node serves it, and a speedscope document
  comes back. Training tasks only; it cannot fail a task. A profile that never
  arrives explains itself only in the node log (`logs --task <id> | grep
  profile`). numba JIT frames are bare addresses; numpy's resolve.
- **Never point `runs_dir` at the share.** Active runs live on the node's
  `/mnt/work` data disk and are *published* from there. The wrapper sets
  `POKER_SOLVER_CACHE=/mnt/work/cache` so the river's 2.6M boards are not
  re-canonicalised (~1 min) on every task.
- **Rungs live in the `checkpoints` CONTAINER; the share holds the record.**
  A published rung is one Blob object, `<run>/static-<iter>.ckpt.zst`, and what
  stays on the share is the manifest, `.run.json`, the loose result files and a
  completion MARKER per rung. The marker is still the run's own claim that a
  rung is complete, so a share run directory is now markers with no snapshot
  beside them.
- **A manifest names `static-N.zarr`, the container holds `static-N.ckpt.zst`,
  and `shared.records.object_name` is the only thing that maps between them.**
  Manifests are never repointed: rewriting them would mutate the durable share
  and destroy the share fallback for exactly the runs whose only other copy is
  the container. Both spellings are permanent; the mapping is idempotent and
  spelled once. 1,081 rungs were uploaded under a name no reader asked for
  before this existed.
- **A manifest OVER-CLAIMS by design.** `prune-checkpoints` drops a snapshot
  without rewriting the ladder that advertises it, so `retained` names rungs
  that were deleted weeks ago -- 1,030 of them measured against 3 genuinely
  lost. Gate a FETCH on the manifest; gate a DELETION on what a store holds.
- **Anything that asks "is this rung published" must ask both stores.** The
  node does it through `blobstore` over its SAS, the dispatcher through
  `cloud.store.blob` with the account key. Building `static-<rung>.zarr` by
  hand and asking the share for a directory is the bug this keeps producing:
  it hit evaluation fetch, warm-start, dispatch verification and prune.
- **`blob.delete_rung` is the only delete against the container**, reached from
  `prune-checkpoints` alone. No task SAS carries `delete`, so nothing running
  on a node can remove a rung even by accident.
- **`infra/store/` is a separate Terraform state** holding the durable share,
  so `just destroy` cannot reach the experiment record. Jobs and tasks are
  created at runtime by Python, never in HCL.
- **Always spell it `terraform -chdir=<dir> …`, never `cd infra && terraform`.**
  The approved forms are per-directory (`-chdir=infra`, `-chdir=infra/serve`,
  `-chdir=infra/store`), so a `cd` form matches nothing and is refused — that
  cost 15 blocked applies across past sessions, every one of them a change the
  user had already agreed to. `apply` on `infra/store` is denied outright: it
  holds the durable share.
- **Two shell things remain shell**: `just panic` (must work from a phone in
  Cloud Shell) and `main.tf`'s `start_task` (runs before any code snapshot
  exists).
- Constraints that look arbitrary but are measured (UserSubscription mode,
  `Dals_v6` not `Dalds_v6`, Gen2-only images, the SKU policy) are in
  `infra/README.md`. Read it before changing pool config.
