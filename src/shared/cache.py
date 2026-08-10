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

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

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


"""A small expiring JSON cache, for answers that are slow to fetch and cheap to
be slightly stale
-------------------------------------------------------------------------------
This is the counterpart to :mod:`src.shared.records`, and the split is the whole
point. ``records`` owns ARTIFACTS: the durable account of what happened, one
convention, atomic, versioned, on the share. This owns CACHES: regenerable, of
no value if lost, deletable at any moment. A guard test in
``tests/shared/test_records.py`` fails if any other module hand-rolls a
``write_text(json.dumps(...))``, precisely so those two stay the only two ways
JSON is written -- the drift it replaced was six writers with six sets of
decisions.

It exists because an in-process memo does nothing for the CLI, which is a fresh
process every time. `poker-solver cost` run three times made three Cost
Management queries and earned a 429 that outlasted the burst by minutes.
"""


def cached_json(name: str, key: str, ttl: float) -> dict[str, Any] | None:
    """A cached value, or ``None`` if absent, expired, or unreadable.

    Every failure is a miss. A cache that can break the thing it accelerates is
    worse than no cache, and these sit in front of the least reliable parts of a
    command.
    """
    try:
        stored = json.loads(_cache_file(name, key).read_text())
        if time.time() - float(stored["at"]) > ttl:
            return None
        value = stored["value"]
    except Exception:  # noqa: BLE001 -- an unreadable cache is simply a miss
        return None
    return value if isinstance(value, dict) else None


def store_json(name: str, key: str, value: dict[str, Any]) -> None:
    """Store a value for a later process. Best-effort: a read-only cache root
    must not break a command that already has its answer."""
    try:
        path = _cache_file(name, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Written beside and renamed, so a concurrent reader sees one whole file
        # or the previous one -- never a half-written record. Two commands at
        # once is ordinary; the pid keeps their scratch files apart.
        scratch = path.with_suffix(f".{os.getpid()}.tmp")
        scratch.write_text(json.dumps({"at": time.time(), "value": value}))
        scratch.replace(path)
    except OSError:
        return


def _cache_file(name: str, key: str) -> Path:
    """One file per key, hashed: a key here is a machine-readable tuple, often
    long, and nothing reads these filenames."""
    return cache_dir(name) / f"{hashlib.sha256(key.encode()).hexdigest()[:16]}.json"
