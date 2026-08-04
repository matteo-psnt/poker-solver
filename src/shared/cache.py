"""Where regenerable artifacts live, which is never inside the working tree.

`data/` used to be the answer. It was deleted twice and came back both times,
because the caches named paths inside it -- so the rule only holds if they have
somewhere else to go.

Resolution order: ``POKER_SOLVER_CACHE`` (the node wrapper sets it to
``/mnt/work/cache``, since a task's HOME is wiped with the task), then
``XDG_CACHE_HOME``, then ``~/.cache/poker-solver``.

Leaving the tree also means this project's several git worktrees share one
cache instead of each recomputing the river's 2.6M boards (~1 min).
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
