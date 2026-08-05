"""The node side of a cloud task: what runs on the Batch node, not on a laptop.

Everything in this package is invoked by ``infra/run_task.py``, which a Batch
task runs directly after extracting the pinned code snapshot. That timing is the
whole design constraint: it starts BEFORE ``uv sync``, so it is **stdlib only**,
enforced by ``tests/shared/cloudtask/node/test_node_interpreter.py``, which
imports the package on a bare interpreter at the version ``infra/main.tf``
installs. The failure being guarded against is invisible: the task dies before
it can say why.

Split by what can go wrong, in dependency order::

    paths      where things live on a node -- pure addressing
    process    running a child under a deadline, and keeping what it said
    plan       the RUN_* environment -> the argv -- pure, no IO
    archive    the share <-> the node's disk
    progress   how far along, and how much of that this task can claim
    handlers   what each KIND of task does
    lifecycle  one task start to finish, and the account of how it ended

``infra/main.tf``'s start task is the one node-side thing still shell: it
formats and mounts the data disk before any code exists to run.
"""
