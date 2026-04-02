"""The node side of a cloud leg: what runs on the Batch node, not on a laptop.

Everything in this package is invoked by ``infra/run_leg.py``, which a Batch
task runs directly after extracting the pinned code snapshot. Two constraints
follow from *when* that happens, and both are enforced by tests in
``tests/shared/node/test_node_interpreter.py``:

**Stdlib only.** The wrapper starts before ``uv sync``, so a third-party import
anywhere in this package makes the whole leg unrunnable -- including the leg
record that would have explained why.

**Python 3.10.** ``infra/main.tf`` pins ``batch.node.ubuntu 22.04``, whose
system ``python3`` is 3.10, not the 3.12+ this project is developed against.
Ruff's ``target-version`` is ``py312``, so ``UP017`` and friends are disabled
for this package in ``pyproject.toml`` -- without that, the formatter rewrites
these modules into something the node cannot import, and every test but one
still passes.

The split is by what can go wrong:

``archive``
    Copying between the node's disk and the SMB share. Seven distinct
    production failures are encoded here as rules; each one cost a run.
``plan``
    The leg's environment turned into an argv. Pure, so a test can look at the
    command line without a node.
``runner``
    The process lifecycle around it: the timeout guard, the tee, the mid-run
    publisher, and the exit accounting.
"""
