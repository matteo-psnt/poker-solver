"""Getting a published run onto local disk, so the server can load it.

The half of `just serve-deploy` that has to do with WHICH RUN is served. The
rest of that script is deployment and none of it changes when you only want to
look at a different run; separating them turns a three-minute SSH round trip
into an in-process load.

**FETCHING IS GONE, and this only answers from local disk now.** It used to
copy one rung off the SMB share; the share holds nothing since the move to blob
(`infra/serve/deploy.sh`, measured 09-10: `/mnt/shared` mounts EMPTY). A reader
kept against it does not fail, it answers "no published run" about a run that is
published -- the confident-and-wrong answer that deploy script exists to refuse.

So `/api/load` can switch to a run the box already holds and says plainly what
is missing otherwise. Restoring the fetch means pulling `<run>/` out of the
`checkpoints` container the way `deploy.sh` does, which is real work and is NOT
what this module does today.

ONE checkpoint, not the run directory -- the shape that survives. A published
run holds its whole ladder at ~850 MB and ~5,500 files per rung, so a directory
copy moves ~127 GB in 400,000 files to load one of them. The manifest names the
rung; `at_iteration` picks it.

The abstraction is separate and is not staged here either: ~773 MB shared by
every run trained against it, so the box normally holds the right one already,
and `build_card_abstraction` raises against the local directory when it does not.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

#: The manifest naming the head checkpoint and the rungs kept beside it. Its
#: presence is also what marks a directory as a run this solver can load.
MANIFEST = "STATIC_CHECKPOINT.json"

#: The run's own event log, which `RunTracker.load` reads for the config.
RUN_LOG = "run.jsonl"


class StagingError(RuntimeError):
    """A run cannot be put on local disk. Always a sentence for a person."""


def stage_run(
    run: str,
    *,
    runs_dir: Path,
    at_iteration: int | None = None,
) -> Path:
    """Make ``run`` loadable under ``runs_dir`` and return its directory.

    Staged means "the manifest, the run log and ONE checkpoint are here". That
    is a question about local disk and nothing else -- see the module docstring
    for why the fetch went away with the share.
    """
    local = runs_dir / run
    manifest = _read(local / MANIFEST)
    if manifest is None:
        raise StagingError(
            f"'{run}' is not on this box. Runs live in the `checkpoints` container "
            "now, and nothing here fetches from it -- stage it with "
            "`just serve-deploy <run>`, which pulls the rung and restarts the server."
        )

    wanted = _checkpoint(manifest, at_iteration)
    if wanted is None:
        rungs = ", ".join(str(rung) for rung in _iterations(manifest)) or "none"
        raise StagingError(
            f"'{run}' has no checkpoint at iteration {at_iteration}. It has: {rungs}."
        )
    if not _complete(local, wanted):
        raise StagingError(
            f"'{run}' has {wanted} listed but not complete on disk. Re-stage it with "
            "`just serve-deploy <run>`; a half-copied rung fails deep in the loader."
        )
    return local


def _read(path: Path) -> dict[str, Any] | None:
    """The manifest, or None when it is absent or unreadable."""
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _iterations(manifest: dict[str, Any]) -> list[int]:
    """Every rung the manifest knows about, head included."""
    rungs = [
        int(entry["iteration"]) for entry in manifest.get("retained", []) if "iteration" in entry
    ]
    head = manifest.get("iteration")
    if head is not None and int(head) not in rungs:
        rungs.append(int(head))
    return sorted(rungs)


def _checkpoint(manifest: dict[str, Any], at_iteration: int | None) -> str | None:
    """The zarr directory to stage: the head, or the rung asked for."""
    if at_iteration is None:
        zarr = manifest.get("zarr")
        return str(zarr) if zarr else None
    if manifest.get("iteration") == at_iteration and manifest.get("zarr"):
        return str(manifest["zarr"])
    for entry in manifest.get("retained", []):
        if entry.get("iteration") == at_iteration and entry.get("zarr"):
            return str(entry["zarr"])
    return None


def _complete(run_dir: Path, zarr: str) -> bool:
    """Whether ``zarr`` is here AND was fully written.

    The `.complete-` sentinel is the writer's own signal, and checking it rather
    than mere existence is what stops a copy interrupted half way -- a box that
    deallocated mid-stage, say -- from being read as a staged run on the next
    attempt.
    """
    return (run_dir / zarr).is_dir() and (run_dir / f".complete-{zarr}").is_file()
