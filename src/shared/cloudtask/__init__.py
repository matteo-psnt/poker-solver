"""A cloud task: what it IS, how a node runs it, and how it is accounted for.

One subsystem, gathered. These modules used to sit at the top level of
``src/shared`` beside ``config``, ``jsonio`` and ``numeric`` -- utilities every
layer uses -- which read as though a Batch wrapper were a general-purpose
helper. It is not: nothing in ``core``, ``engine`` or the solver half of
``pipeline`` has any business with it, and ``node/`` is imported by nothing in
``src`` at all. Its only entry point is ``infra/run_task.py``, on a node.

* :mod:`~src.shared.cloudtask.kinds` -- what KIND of work a task is, and
  everything that differs because of it. Imported from BOTH ends of the wire:
  ``interfaces.cloud.spec`` builds a task with it and ``cloudtask.node.plan``
  runs one with it, which is what stops a node accepting what the submitter
  would have refused.
* :mod:`~src.shared.cloudtask.task_log` -- the durable per-task account, joined
  from what the node said and what Batch observed. Read by the command layer and
  by ``pipeline.services.experiments``, which is why this package sits under
  ``shared`` rather than under ``interfaces``: ``pipeline -> interfaces`` is
  forbidden, and rightly.
* :mod:`~src.shared.cloudtask.node` -- what happens ON the node.

Why ``shared`` and not ``interfaces``
-------------------------------------
Two constraints pin it here. ``pipeline`` reads the task log, and the layering
contract forbids ``pipeline -> interfaces``. And the node wrapper starts BEFORE
``uv sync``, so it can import neither the Azure SDK nor anything else
third-party -- placing it under ``interfaces`` would leave every Batch task one
stray import in ``interfaces/__init__.py`` away from dying at bootstrap, with no
test that could see it coming.

Stdlib only, all of it, plus ``shared.records`` and ``shared.cache``. That is a
contract in ``.importlinter``, not a convention: the failure it prevents is
invisible, because a node that cannot import the wrapper dies before the wrapper
can say so.
"""
