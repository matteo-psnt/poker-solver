"""A cloud task: what it IS, how a node runs it, and how it is accounted for.

* :mod:`~src.shared.cloudtask.kinds` -- what KIND of work a task is, and
  everything that differs because of it. Imported from BOTH ends of the wire:
  ``interfaces.cloud.spec`` builds a task with it and ``cloudtask.node.plan``
  runs one with it, which is what stops a node accepting what the submitter
  would have refused.
* :mod:`~src.shared.cloudtask.task_log` -- the durable per-task account, joined
  from what the node said and what Batch observed.
* :mod:`~src.shared.cloudtask.node` -- what happens ON the node. Imported by
  nothing in ``src``; its only entry point is ``infra/run_task.py``.

Two constraints pin this under ``shared`` rather than ``interfaces``.
``pipeline`` reads the task log, and ``pipeline -> interfaces`` is forbidden.
And the node wrapper starts BEFORE ``uv sync``, so under ``interfaces`` every
Batch task would be one stray import in ``interfaces/__init__.py`` away from
dying at bootstrap, with no test that could see it coming.

Stdlib only, all of it, plus ``shared.records`` and ``shared.cache`` -- a
contract ``tests/shared/cloudtask/test_imports.py`` walks the closure to
enforce (importlinter cannot express it), because a node that cannot import
the wrapper dies before the wrapper can say so.
"""
