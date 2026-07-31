"""Resident-set size of a process, for memory telemetry.

Training has stalled with every worker ALIVE and none producing -- the shape of
memory pressure rather than a crash -- and there was no way to tell after the
fact how much memory the run was actually using. RSS per worker turns "8 workers
works and 16 does not" from folklore into a number.

Deliberately dependency-free (no psutil): this is read on the batch path, and
`/proc` is a two-syscall read on the only platform training runs on.
"""

from __future__ import annotations

import os
import sys

_PAGE_SIZE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096


def rss_mb(pid: int | None = None) -> int | None:
    """Resident set size in MiB, or None where it cannot be read.

    None rather than 0 on failure: a zero would average into telemetry as a real
    reading, and "we could not measure" is a different fact from "it used none".

    Note for readers comparing worker totals against the node: RSS counts shared
    pages in EVERY process that maps them, so summing it across workers
    double-counts the mmapped bucket matrices and the shared training arrays.
    Use it to compare workers against each other and to track growth over time,
    not to compute a node total.
    """
    target = os.getpid() if pid is None else pid
    if sys.platform == "darwin":
        # macOS has no /proc; ps is slow but this path is only ever hit in local
        # development, never in the batch loop on a node.
        return None
    try:
        with open(f"/proc/{target}/statm") as handle:
            resident_pages = int(handle.read().split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return resident_pages * _PAGE_SIZE // (1024 * 1024)
