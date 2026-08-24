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
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from src.shared import records
from src.shared.cloudtask.kinds import TaskName
from src.shared.cloudtask.node import archive

CONTAINER = "checkpoints"

# Latency-bound, not bandwidth-bound: a manifest is a few KB and the store is
# in another country, so the downloads overlap.
_PARALLEL_DOWNLOADS = 32
ABSTRACTIONS = "abstractions"

DIAGNOSTICS = "diagnostics"

# The ops that PUT anything. Everything else fetches, and a fetch has no
# business holding a credential that can overwrite what it read. PRECOMPUTE is
# here because it publishes the card abstraction it builds. Kept beside the
# minting rather than at the call site so there is one list of who may write.
#
# A task that writes checkpoints must be named here, and
# `test_every_kind_is_classified` fails if a new one is not. The migration was
# once omitted, handed a read-only SAS, and every upload came back 403
# `AuthorizationPermissionMismatch` -- the scoping worked as designed and the
# list was wrong.
WRITES_BLOBS = frozenset({TaskName.TRAIN, TaskName.TRAIN_PCS, TaskName.PRECOMPUTE})

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

    `write` is the discriminator between a task that publishes and one that
    reads: a score or an evaluate FETCHES and must not be able to overwrite what
    it read. Both need `list`, because a fetch of "the current rung" resolves a
    name first.

    ONE ACCOUNT SAS, named per container by its URL. `generate_account_sas`
    authorises the account, so the same token reaches the abstractions
    container through `abstractions_uri` -- which is what lets a task carry one
    credential rather than one per store.
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


def diagnostics_sas(account: str, key: str) -> str:
    """A WRITABLE credential for the diagnostics container, for every task.

    Separate from the checkpoint SAS because that one is an ACCOUNT token, so
    its `write=False` -- which is what stops a score overwriting the rung it
    read -- also stripped write on diagnostics. The effect was that no
    evaluate, score or duel could publish its log tail: the PUT came back 403
    and the only account of the failure was the one thing that could not be
    written. Every score failure was invisible through `poker-solver logs`.

    Scoped to ONE CONTAINER, so widening it back does not touch the rungs. No
    `delete`: a task removes nothing, here or anywhere.
    """
    from azure.storage.blob import (  # noqa: PLC0415 -- Azure only when dispatching
        ContainerSasPermissions,
        generate_container_sas,
    )

    now = datetime.now(UTC)
    token = generate_container_sas(
        account_name=account,
        container_name=DIAGNOSTICS,
        account_key=key,
        permission=ContainerSasPermissions(read=True, list=True, write=True, create=True),
        start=now - CLOCK_SKEW,
        expiry=now + SAS_LIFETIME,
    )
    return f"https://{account}.blob.core.windows.net/{DIAGNOSTICS}?{token}"


def abstractions_uri(checkpoint_sas: str) -> str:
    """The abstractions container, from the checkpoint container's SAS."""
    from src.shared.cloudtask.node import blobstore  # noqa: PLC0415 -- shared with the node

    return blobstore.sibling_container(checkpoint_sas, ABSTRACTIONS)


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
        # Credential files. A snapshot seals the WORKING TREE, not the index, so
        # an untracked secrets file in the root would ride up and sit in `code/`
        # readable by every node. `chipzen.toml` is here because it is the
        # Chipzen SDK's own config, which carries a bot token verbatim.
        "chipzen.toml",
        ".chipzen",
    }
)

# Whole families of credential file, matched by prefix rather than by name --
# `.env`, `.env.local`, `.env.production`. An exact-name set cannot express this,
# and the variant that gets forgotten is the one that leaks. A comment, not a
# string: a string here is not a docstring and would bind to SNAPSHOT_EXCLUDES.
SNAPSHOT_EXCLUDE_PREFIXES = (".env",)


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
    if any(part.startswith(SNAPSHOT_EXCLUDE_PREFIXES) for part in parts):
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


def read_task_log(config: Any, task_id: str) -> str | None:
    """One task's published log tail, or None when it has none.

    A task publishes this while it RUNS -- the node-side stream dies with the
    node, so this copy is the only one a reader can reach. Diagnostics rather
    than record: it expires on the container's lifecycle policy, and the
    account that outlives a task is its `legs` row.
    """
    from azure.core.exceptions import ResourceNotFoundError  # noqa: PLC0415 -- Azure only here
    from azure.storage.blob import BlobServiceClient  # noqa: PLC0415 -- see above

    service = BlobServiceClient(
        account_url=f"https://{config.storage_account}.blob.core.windows.net",
        credential=config.share_key,
    )
    try:
        blob = service.get_blob_client(DIAGNOSTICS, f"{task_id}.log")
        return blob.download_blob().readall().decode("utf-8", "replace")
    except ResourceNotFoundError:
        return None


def task_log_names(config: Any) -> list[str]:
    """Every published task log, sorted.

    Published logs matter more than node-side `stdout.txt`: Batch keeps task
    output on the node, and the pool scales to zero within minutes of a task
    ending, so the node copy is gone for exactly the failed tasks most worth
    reading.
    """
    from azure.storage.blob import BlobServiceClient  # noqa: PLC0415 -- Azure only here

    service = BlobServiceClient(
        account_url=f"https://{config.storage_account}.blob.core.windows.net",
        credential=config.share_key,
    )
    return sorted(x.name for x in service.get_container_client(DIAGNOSTICS).list_blobs())


def write_diagnostic(config: Any, name: str, body: str) -> None:
    """Put one small text object in the diagnostics container."""
    _diagnostics(config).upload_blob(name, body.encode("utf-8"), overwrite=True)


def diagnostic_names(config: Any, suffix: str) -> list[str]:
    """Every diagnostics object whose name ends with `suffix`, sorted."""
    return sorted(x.name for x in _diagnostics(config).list_blobs() if x.name.endswith(suffix))


def download_diagnostic(config: Any, name: str, destination: Path) -> None:
    """Save one diagnostics object beside the operator."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(_diagnostics(config).download_blob(name).readall())


def _diagnostics(config: Any) -> Any:
    from azure.storage.blob import BlobServiceClient  # noqa: PLC0415 -- Azure only here

    return BlobServiceClient(
        account_url=f"https://{config.storage_account}.blob.core.windows.net",
        credential=config.share_key,
    ).get_container_client(DIAGNOSTICS)


def published_abstractions(config: Any) -> list[str]:
    """Every abstraction the container holds, by DIRECTORY name, sorted.

    The collision guard `submit-precompute` runs before it allocates a node.
    It asked the share until the share stopped holding abstractions, and then
    returned an empty list every time -- a guard that cannot refuse, in front
    of the one invariant here that matters: bucket ASSIGNMENT is not pinned by
    the abstraction hash, so republishing over a name silently rebuckets every
    run already trained against it.
    """
    from azure.storage.blob import BlobServiceClient  # noqa: PLC0415 -- Azure only here

    container = BlobServiceClient(
        account_url=f"https://{config.storage_account}.blob.core.windows.net",
        credential=config.share_key,
    ).get_container_client(ABSTRACTIONS)
    return sorted(
        entry.name.removesuffix(archive.ABSTRACTION_SUFFIX)
        for entry in container.list_blobs()
        if entry.name.endswith(archive.ABSTRACTION_SUFFIX)
    )


def published_record(config: Any) -> dict[str, dict[str, Any]]:
    """Every published run, as `{run_id: {"rungs": {...}, "manifest": bytes}}`.

    ONE listing plus one small download per manifest. The manifests are the
    whole of what a reader materialises now -- the eval documents and run logs
    that made this incremental live in Postgres, and 345 files of a few KB do
    not need an etag cache to stay fast.
    """
    from azure.storage.blob import BlobServiceClient  # noqa: PLC0415 -- Azure only here

    container = BlobServiceClient(
        account_url=f"https://{config.storage_account}.blob.core.windows.net",
        credential=config.share_key,
    ).get_container_client(CONTAINER)

    found: dict[str, dict[str, Any]] = {}
    manifests: list[str] = []
    for entry in container.list_blobs():
        run, _, name = entry.name.partition("/")
        if not name:
            continue
        slot = found.setdefault(run, {"rungs": set(), "manifest": None})
        if name == records.STATIC_CHECKPOINT:
            manifests.append(run)
        else:
            slot["rungs"].add(name)

    def _pull(run: str) -> tuple[str, bytes]:
        name = f"{run}/{records.STATIC_CHECKPOINT}"
        return run, container.download_blob(name).readall()

    with ThreadPoolExecutor(max_workers=_PARALLEL_DOWNLOADS) as pool:
        for run, body in pool.map(_pull, manifests):
            found[run]["manifest"] = body
    return found


def published_rungs(config: Any) -> dict[str, set[str]]:
    """Every rung the container holds, as `{run_id: {object name}}`.

    ONE listing for the whole container rather than one per run: the readers
    ask about every published run at once, and a per-run call is a round trip
    each against a store in another country.

    This is what replaced the share's completion markers. A rung is one
    atomically-committed object, so its PRESENCE is the completeness a marker
    used to assert about a directory that could be half-copied.
    """
    from azure.storage.blob import BlobServiceClient  # noqa: PLC0415 -- Azure only here

    service = BlobServiceClient(
        account_url=f"https://{config.storage_account}.blob.core.windows.net",
        credential=config.share_key,
    )
    found: dict[str, set[str]] = {}
    for entry in service.get_container_client(CONTAINER).list_blobs():
        run, _, name = entry.name.partition("/")
        if name and name != records.STATIC_CHECKPOINT:
            found.setdefault(run, set()).add(name)
    return found


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
