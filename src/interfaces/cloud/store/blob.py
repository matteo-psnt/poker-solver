"""Blob at dispatch: the checkpoint container's SAS, and the sealed code tree.

THE LAPTOP SIDE of `shared.cloudtask.node.blobstore`. The SDK lives here and
cannot live there: `archive` is imported by the wrapper before `uv sync`, on
the interpreter the start task installs, so the node speaks REST through
`urllib` and needs a URL that already carries its own authorisation. The code
tarball goes further -- the wrapper is INSIDE it, so the task command line
fetches it with `curl` and a URL is the only credential that can carry.

Minted per dispatch rather than stored anywhere. A SAS expires by design, so
there is nothing to rotate and nothing to leak from a config file -- and the
task it is sealed into outlives it only if the task outlives `SAS_LIFETIME`,
which is why that is measured against the longest job this pool runs rather
than the longest task.
"""

from __future__ import annotations

import tarfile
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from src.shared.cloudtask.kinds import TaskName

CONTAINER = "checkpoints"

# The ops that PUT a rung. Everything else fetches one, and a fetch has no
# business holding a credential that can overwrite what it read. Kept beside the
# minting rather than at the call site so there is one list of who may write.
#
# A task that writes checkpoints must be named here, and
# `test_every_kind_is_classified` fails if a new one is not. The migration was
# once omitted, handed a read-only SAS, and every upload came back 403
# `AuthorizationPermissionMismatch` -- the scoping worked as designed and the
# list was wrong.
WRITES_CHECKPOINTS = frozenset({TaskName.TRAIN, TaskName.TRAIN_PCS})

# Longer than any job, not any task. A ladder score fans out 30 rungs behind one
# dispatch and the last of them can start hours after the first; a SAS that
# expired between them would fail a task that had done nothing wrong, and the
# retry would fail identically.
SAS_LIFETIME = timedelta(days=7)

# Backdated, because the node's clock is not this machine's. A SAS whose window
# opens "now" is refused by a node running a few seconds behind, and the failure
# reads as a 403 that looks exactly like a bad signature.
CLOCK_SKEW = timedelta(minutes=15)


def container_sas(account: str, key: str, *, write: bool) -> str:
    """A URL for the checkpoint container, carrying its own authorisation.

    `write` is the discriminator between a training task and a reader: a score
    or an evaluate FETCHES rungs and must not be able to overwrite one. Both
    need `list`, because a fetch of "the current rung" resolves a name first.
    """
    from azure.storage.blob import (  # noqa: PLC0415 -- Azure only when dispatching
        AccountSasPermissions,
        ResourceTypes,
        generate_account_sas,
    )

    now = datetime.now(UTC)
    token = generate_account_sas(
        account_name=account,
        account_key=key,
        resource_types=ResourceTypes(container=True, object=True),
        permission=AccountSasPermissions(read=True, list=True, write=write, create=write),
        start=now - CLOCK_SKEW,
        expiry=now + SAS_LIFETIME,
    )
    return f"https://{account}.blob.core.windows.net/{CONTAINER}?{token}"


SNAPSHOT_EXCLUDES = frozenset(
    {
        ".git",
        "data",
        ".venv",
        "__pycache__",
        "node_modules",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".terraform",
        ".claude",
        ".uv-cache",
        ".import_linter_cache",
        ".idea",
        ".vscode",
        ".DS_Store",
    }
)


def snapshot_name(now: datetime) -> str:
    """The id of one immutable code snapshot; the blob is `<id>.tar.gz`."""
    return f"code-{now:%Y%m%d_%H%M%S}"


def _snapshot_filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    """Drop excluded directories, and strip ownership from what remains.

    Ownership is cleared because the node extracts as an unprivileged task
    user: a tarball carrying the laptop's uid/gid is one more thing for tar to
    fail to restore. macOS xattrs never enter the archive -- ``tarfile`` does
    not write them.
    """
    parts = Path(info.name).parts
    if any(part in SNAPSHOT_EXCLUDES for part in parts):
        return None
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    return info


def build_code_snapshot(root: Path, destination: Path) -> None:
    """Seal the working tree into one gzipped tarball.

    One object rather than a tree: a half-uploaded tarball is simply absent,
    where a partially-populated tree is something a node might run.
    """
    with tarfile.open(destination, "w:gz") as archive:
        for entry in sorted(root.iterdir()):
            archive.add(entry, arcname=entry.name, filter=_snapshot_filter)


def publish_code_snapshot(account: str, key: str, container: str, root: Path, now: datetime) -> str:
    """Build and upload an immutable snapshot of ``root``; return its id.

    `overwrite=False` because a snapshot is sealed: the id names bytes, and a
    second dispatch inside the same second must fail rather than replace what
    a task already queued against.
    """
    from azure.storage.blob import BlobClient  # noqa: PLC0415 -- Azure only when dispatching

    name = snapshot_name(now)
    with tempfile.TemporaryDirectory() as workspace:
        tarball = Path(workspace) / f"{name}.tar.gz"
        build_code_snapshot(root, tarball)
        client = BlobClient(
            account_url=f"https://{account}.blob.core.windows.net",
            container_name=container,
            blob_name=tarball.name,
            credential=key,
        )
        with tarball.open("rb") as handle:
            client.upload_blob(handle, overwrite=False)
    return name


def code_snapshot_sas(account: str, key: str, container: str, snapshot: str) -> str:
    """A read-only URL for ONE snapshot blob, carrying its own authorisation.

    Narrower than the checkpoint SAS on purpose: a service SAS on the single
    object, read and nothing else, because the command line that holds it
    fetches one tarball and never lists, writes or reads a sibling. Same
    window and backdating as `container_sas`, for the same reasons.
    """
    from azure.storage.blob import (  # noqa: PLC0415 -- Azure only when dispatching
        BlobSasPermissions,
        generate_blob_sas,
    )

    now = datetime.now(UTC)
    blob_name = f"{snapshot}.tar.gz"
    token = generate_blob_sas(
        account_name=account,
        container_name=container,
        blob_name=blob_name,
        account_key=key,
        permission=BlobSasPermissions(read=True),
        start=now - CLOCK_SKEW,
        expiry=now + SAS_LIFETIME,
    )
    return f"https://{account}.blob.core.windows.net/{container}/{blob_name}?{token}"


def _client(config: Any, run_id: str, object_name: str) -> Any:
    """A client for one rung, from the LAPTOP.

    The node reaches the container through `blobstore` over a SAS; a dispatcher
    has the account key and the SDK, so it goes direct. Both halves exist
    because since rungs live in the container, questions about a rung -- does it
    exist, how big is it, delete it -- are asked on both sides of a dispatch.
    """
    from azure.storage.blob import BlobServiceClient  # noqa: PLC0415 -- Azure only here

    service = BlobServiceClient(
        account_url=f"https://{config.storage_account}.blob.core.windows.net",
        credential=config.share_key,
    )
    return service.get_blob_client(CONTAINER, f"{run_id}/{object_name}")


def rung_size(config: Any, run_id: str, object_name: str) -> int:
    """Bytes the container holds for one rung; 0 when it holds none."""
    from azure.core.exceptions import ResourceNotFoundError  # noqa: PLC0415 -- Azure only here

    try:
        return int(_client(config, run_id, object_name).get_blob_properties().size or 0)
    except ResourceNotFoundError:
        return 0


def holds_rung(config: Any, run_id: str, object_name: str) -> bool:
    """Whether the container holds one rung."""
    return rung_size(config, run_id, object_name) > 0


def delete_rung(config: Any, run_id: str, object_name: str) -> bool:
    """Remove one rung from the container. False when it was not there.

    The ONLY delete this project performs against the container, and it is
    reached from `prune-checkpoints` alone -- no task SAS carries `delete`, so
    nothing running on a node can do this even by accident.
    """
    from azure.core.exceptions import ResourceNotFoundError  # noqa: PLC0415 -- Azure only here

    try:
        _client(config, run_id, object_name).delete_blob()
    except ResourceNotFoundError:
        return False
    return True
