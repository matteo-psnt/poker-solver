"""The node side of a cloud leg: what runs on the Batch node, not on a laptop.

Everything in this package is invoked by ``infra/run_leg.py``, which a Batch
task runs directly after extracting the pinned code snapshot. Two constraints
follow from *when* that happens, and both are enforced by tests in
``tests/shared/node/test_node_interpreter.py``:

**Stdlib only**, because the wrapper starts before ``uv sync``, and **Python
3.10**, because ``infra/main.tf`` pins ``batch.node.ubuntu 22.04``. Neither
failure is visible: the leg dies before it can say why. Ruff targets py312, so
``pyproject.toml`` disables ``UP017`` here -- without that the formatter
rewrites these modules into something the node cannot import.

Split by what can go wrong: ``archive`` (share <-> disk), ``plan`` (environment
-> argv, pure), ``runner`` (process lifecycle).
"""
