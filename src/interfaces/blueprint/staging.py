"""Getting a published run onto local disk, so the server can load it.

This is the half of `just serve-deploy` that actually has to do with which run
is served. The rest of that script -- rsyncing the code, `uv sync`, rewriting
the unit's environment, `systemctl restart` -- is deployment, and none of it
changes when you only want to look at a different run. Separating them is what
turns a three-minute SSH round trip into an in-process load.

Why copy at all
---------------
The share is SMB. A checkpoint is ~5,500 small files that the read path mmaps,
so serving one straight off the share turns every page fault into a network
round trip. `deploy.sh` copies for that reason and so does this.

Why ONE checkpoint
------------------
A published run holds its whole ladder -- `static-1000000.zarr` through
`static-150000000.zarr`, ~850 MB and ~5,500 files each. Copying the run
directory therefore moved ~127 GB in 400,000 files to load ONE of them.
Measured on the box at 950 files/min, that is about six hours to answer a
question the server needs 850 MB to answer.

So the manifest is read first and exactly one checkpoint is staged: the head, or
the rung `at_iteration` names. That is ~5 minutes the first time a run is
touched and seconds every time after, and it is what makes `at` worth having --
you stage a rung, not a ladder.

The abstraction is NOT copied here either. It is ~773 MB, it is shared by every
run trained against it, and the box already holds the one its deployed run
needed -- so the common case is that the target run wants the same one and there
is nothing to do. When it wants a different one, `build_card_abstraction` raises
against the local directory and the server reports that rather than silently
pulling most of a gigabyte over SMB while a caller waits on an HTTP request.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

#: Where the durable store is mounted on the blueprint box, read-only. Matches
#: `infra/serve/main.tf`'s `mnt-shared.mount`, which the unit already waits for.
DEFAULT_SHARE = Path("/mnt/shared")

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
    share: Path = DEFAULT_SHARE,
    at_iteration: int | None = None,
) -> Path:
    """Make ``run`` loadable under ``runs_dir`` and return its directory.

    Staged means "the manifest, the run log and ONE checkpoint are here" -- not
    "the published directory has been mirrored". A run already carrying the
    checkpoint being asked for is served from disk without touching the share at
    all, which is both the fast path and the only path that works on a laptop
    with nothing mounted.
    """
    local = runs_dir / run
    manifest = _read(local / MANIFEST)

    # Already here, with the rung being asked for? Nothing to do.
    if manifest is not None:
        wanted = _checkpoint(manifest, at_iteration)
        if wanted and _complete(local, wanted):
            return local

    published = share / "archive" / run
    if not published.is_dir():
        # Named separately from "not on the share", because a box with no share
        # mounted and a genuinely unknown run are different problems.
        if not share.is_dir():
            raise StagingError(
                f"'{run}' is not on local disk and the share is not mounted at {share}, "
                "so there is nowhere to fetch it from."
            )
        raise StagingError(f"No published run '{run}' under {share / 'archive'}.")

    published_manifest = _read(published / MANIFEST)
    if published_manifest is None:
        raise StagingError(
            f"'{run}' has no {MANIFEST}, so it is not a run this solver can load. "
            "Checkpoints from the retired dynamic backend are unreadable at HEAD "
            "by design."
        )

    zarr = _checkpoint(published_manifest, at_iteration)
    if zarr is None:
        rungs = ", ".join(str(rung) for rung in _iterations(published_manifest)) or "none"
        raise StagingError(
            f"'{run}' has no checkpoint at iteration {at_iteration}. It has: {rungs}."
        )

    local.mkdir(parents=True, exist_ok=True)
    try:
        # The two small files first: a manifest present beside an absent
        # checkpoint would make the fast path above claim a run is staged when
        # the expensive half never arrived.
        _copy_tree(published / zarr, local / zarr)
        _marker(published, local, zarr)
        _copy_file(published / RUN_LOG, local / RUN_LOG)
        _copy_file(published / MANIFEST, local / MANIFEST)
    except OSError as error:
        raise StagingError(f"Could not copy '{run}' from the share: {error}") from error

    if not _complete(local, zarr):
        raise StagingError(f"'{run}' staged, but {zarr} did not arrive complete.")
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


def _marker(source: Path, target: Path, zarr: str) -> None:
    """Copy the completion sentinel, if the publisher wrote one."""
    sentinel = source / f".complete-{zarr}"
    if sentinel.is_file():
        _copy_file(sentinel, target / f".complete-{zarr}")


def _copy_file(source: Path, target: Path) -> None:
    """One file, skipped when the local copy already matches."""
    if not source.is_file():
        return
    if target.is_file():
        here, there = target.stat(), source.stat()
        if here.st_size == there.st_size and here.st_mtime >= there.st_mtime:
            return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _copy_tree(source: Path, target: Path) -> None:
    """`cp -ru`, which is what `deploy.sh` has always used, and for good reason.

    A blind `copytree` re-reads every file on every attempt. Over SMB, where
    per-file overhead dominates, that is the difference between a resumed copy
    finishing in seconds and one starting over -- and an interrupted stage is the
    normal case, being what a caller who lost patience and clicked again leaves
    behind. Size-and-mtime rather than content is sound because a published run
    is immutable once archived.
    """
    for entry in source.rglob("*"):
        destination = target / entry.relative_to(source)
        if entry.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        else:
            _copy_file(entry, destination)
