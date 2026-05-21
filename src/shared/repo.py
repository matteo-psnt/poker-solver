"""Where this checkout is, derived once instead of counted twenty-six times.

Every caller that needed the repo root wrote `Path(__file__).resolve().parents[N]`,
and N is a fact about the CALLING FILE'S DEPTH -- not about the checkout. So the
expression is correct only until the file moves, and moving a file is not
something anyone expects to change where the repository is.

It has already gone wrong twice, both found the same day:

- `cloud/dispatch.py` used `parents[3]` to derive the tree it seals into a code
  snapshot. Moved one directory deeper, that resolves to `src/interfaces`, and
  a submission would have tarred *that* and stamped it as the run's provenance.
  Nothing about the failure looks like a path bug: you get a snapshot, and it is
  of the wrong thing.
- `tests/shared/cloudtask/test_task_log.py` had `parents[2]`, which is `tests/`,
  and passed it to `sys.path.insert` in a subprocess meant to prove the node
  package imports on a bare interpreter. The subprocess imported fine -- from
  the ambient CWD, because `python -c` puts it on `sys.path`. The guarantee the
  test advertised was being supplied by where pytest happened to be run.

Both failures are silent, and neither is visible in the diff that causes it.

Found by NAME instead. `src` is this package's root and cannot be renamed
without renaming every import in the tree, so walking up to it is stable under
any move of any file inside it. `parents` is ordered nearest-first, so a
checkout that itself sits in a directory called `src` still resolves to the
inner one.
"""

from __future__ import annotations

from pathlib import Path

SRC = next(parent for parent in Path(__file__).resolve().parents if parent.name == "src")

ROOT = SRC.parent
