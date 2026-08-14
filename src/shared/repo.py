"""Where this checkout is, derived once instead of counted twenty-six times.

`Path(__file__).resolve().parents[N]` encodes the CALLING FILE'S DEPTH, which is
not a fact about the checkout -- so it is correct only until the file moves, and
moving a file is not something anyone expects to change where the repository is.
Both failures it has caused were silent and neither was visible in the diff:
a code snapshot sealed from `src/interfaces` and stamped as a run's provenance,
and a node-import guard whose guarantee was actually being supplied by wherever
pytest happened to run.

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
