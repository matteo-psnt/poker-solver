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
`store/` (the SMB share), `cost/` (Cost Management), with `config.py` and
`serve_box.py` above them. It lives under `interfaces` so nothing in
`pipeline`/`engine`/`core` can reach Azure.

- **Auth is `AzureCliCredential`, never `DefaultAzureCredential`.** The default
  chain probes the link-local IMDS address, which on a laptop hangs rather than
  refusing (measured: >120s vs 1.3s).
- **Read cost is a maintained property.** `tests/interfaces/cloud/test_read_cost.py`
  pins it as call counts, because latency is invisible in a test and enormous in
  practice. Three rules: never list tasks for a job you will discard; issue
  independent round trips together; never sync `keys-*` key tables (the deleted
  dynamic backend's — 37 of the 38 MB a share read pulled).
- **`src/shared/cloudtask/` is STDLIB ONLY.** The node runs it BEFORE
  `uv sync`, on the pinned image's system `python3` — **3.10**, not 3.13.
  Enforced twice: `test_node_interpreter.py` imports the node's whole closure on
  that interpreter, and `test_imports.py` is fail-closed (nothing outside
  `records`/`jsonio`/`cache` may be reached). It lives under `shared` because
  `pipeline` reads the task record and `pipeline → interfaces` is forbidden.
- **`task_history.py` is deliberately OUTSIDE that package** — it is the reading
  half, runs only on a laptop, and the fail-closed guard walks every file in
  `cloudtask/`, which would hold ~260 lines to a stdlib floor for no reason.
- **Exit 124 and 137 are DIFFERENT causes** — 124 is the guard's deadline (a
  hang), 137 is SIGKILL from outside (the OOM killer). A wrong terminal cause is
  permanent: it suppresses reconciliation. `poker-solver tasks` is how you find
  out why a task died, because the run log cannot record a death (the container
  is gone first).
- **Never point `runs_dir` at the share.** Active runs live on the node's
  `/mnt/work` data disk and are *published* to the share.
- **`infra/store/` is a separate Terraform state** holding the durable share, so
  `just destroy` cannot reach the experiment record. Jobs and tasks are created
  at runtime by Python, never in HCL.
- **Two shell things REMAIN shell**: `just panic` (must work from a phone in
  Cloud Shell, with no venv and no Terraform state) and `main.tf`'s `start_task`
  (disk discovery, `mkfs`, mount — it runs before any code snapshot exists).
- Constraints that look arbitrary but are measured (UserSubscription mode,
  `Dals_v6` not `Dalds_v6`, Gen2-only images, the SKU policy) are documented in
  `infra/README.md`. Read it before changing pool config.
- The node wrapper sets `POKER_SOLVER_CACHE=/mnt/work/cache`, because a Batch
  task's `HOME` is its own working directory and the default would
  re-canonicalise the river's 2.6M boards (~1 min) on every task.
