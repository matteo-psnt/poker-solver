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
round trip. `deploy.sh` copies for that reason and so does this; the copy is
`cp -ru`-shaped (skip what is already there, newer wins), which makes a repeat
switch back to a run you have already looked at nearly free.

The abstraction is NOT copied here. It is ~773 MB, it is shared by every run
trained against it, and the box already holds the one its deployed run needed --
so the common case is that the target run wants the same one and there is
nothing to do. When it wants a different one, `build_card_abstraction` raises
against the local directory and the server reports that rather than silently
pulling most of a gigabyte over SMB while a caller waits on an HTTP request.
"""

from __future__ import annotations

import shutil
from pathlib import Path

#: Where the durable store is mounted on the blueprint box, read-only. Matches
#: `infra/serve/main.tf`'s `mnt-shared.mount`, which the unit already waits for.
DEFAULT_SHARE = Path("/mnt/shared")


class StagingError(RuntimeError):
    """A run cannot be put on local disk. Always a sentence for a person."""


def stage_run(run: str, *, runs_dir: Path, share: Path = DEFAULT_SHARE) -> Path:
    """Make ``run`` available under ``runs_dir`` and return its directory.

    Already-local wins outright: a run that is on the disk is served from there
    without consulting the share at all, which is both the fast path and the
    only path that works when the share is not mounted (a laptop).
    """
    local = runs_dir / run
    if (local / "STATIC_CHECKPOINT.json").is_file():
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

    local.mkdir(parents=True, exist_ok=True)
    try:
        # `dirs_exist_ok` so a partial earlier copy is completed rather than
        # refused, which is what makes a retry after a timeout do the right thing.
        shutil.copytree(published, local, dirs_exist_ok=True)
    except OSError as error:
        raise StagingError(f"Could not copy '{run}' from the share: {error}") from error

    if not (local / "STATIC_CHECKPOINT.json").is_file():
        raise StagingError(
            f"'{run}' copied, but it has no STATIC_CHECKPOINT.json — it is not a "
            "run this solver can load. Checkpoints from the retired dynamic "
            "backend are unreadable at HEAD by design."
        )
    return local
