"""Representation versioning for training runs.

A run's *representation version* identifies the on-disk checkpoint layout plus the
infoset-encoding format it was produced under. It is the anchor for detecting —
loudly, at build time — when a code change would silently render existing runs
unloadable (the failure mode that orphaned pre-``src.bucketing`` artifacts).

Scope note: the full migration applier/registry is deliberately deferred until a
real format change needs one (there is nothing yet to migrate). What exists today
is the stamp plus the enforcement tripwire — a committed golden run that must keep
loading under current code, so a format break fails the test suite instead of
silently orphaning runs. See the iteration-and-migration design memo.
"""

from __future__ import annotations

import json
from pathlib import Path

# Bump when a change alters the on-disk run/checkpoint format or the infoset
# encoding such that an existing run can no longer be loaded/continued as-is.
# A bump is only safe alongside a migration or a documented barrier; the
# golden-run test is what forces that discipline.
REPRESENTATION_VERSION = 1

RUN_METADATA_FILE = ".run.json"


def run_representation_version(run_dir: Path) -> int:
    """Representation version stamped in a run's metadata; 0 for pre-versioning runs."""
    meta_path = Path(run_dir) / RUN_METADATA_FILE
    if not meta_path.exists():
        raise FileNotFoundError(f"No run metadata at {meta_path}")
    with open(meta_path) as f:
        data = json.load(f)
    return int(data.get("representation_version", 0))
