"""Where regenerable artifacts live, which is never inside the working tree.

`data/` used to be the answer, and the trouble with that is not the disk: a
directory inside the checkout reads as part of the project. It survived several
prunes, it came back after each one, and it left the repo looking like it held
state when everything in it was reproducible from code. The rule is now
literal — **the working tree holds no runtime artifact at all** — and the only
way to keep a rule like that is to give the caches somewhere else to go.

Resolution order:

``POKER_SOLVER_CACHE``
    An explicit override. The node wrapper sets it to ``/mnt/work/cache`` so a
    Batch node keeps its cache on the data disk, shared by every task that runs
    there — previously achieved by symlinking ``$CODE/data`` at the same disk,
    which only worked because the path was relative to the code tree.
``XDG_CACHE_HOME``
    The conventional location, honoured where it is set.
otherwise
    ``~/.cache/poker-solver``.

A side benefit of leaving the tree: this project keeps several git worktrees,
and they now share one cache instead of each recomputing the river's 2.6M
boards (~1 min) the first time it is asked.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_OVERRIDE = "POKER_SOLVER_CACHE"


def cache_root() -> Path:
    """The base directory for every regenerable artifact.

    Not created here. A caller that writes creates its own subdirectory, so
    merely importing this module never puts a directory on disk -- which is the
    behaviour that made ``data/`` reappear after each deletion.
    """
    override = os.environ.get(ENV_OVERRIDE)
    if override:
        return Path(override)
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "poker-solver"


def cache_dir(name: str) -> Path:
    """A named cache under :func:`cache_root`, e.g. ``canonical_boards``."""
    return cache_root() / name
