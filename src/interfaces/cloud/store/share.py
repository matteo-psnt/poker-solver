"""The durable share: published runs, code snapshots, abstractions, task logs.

The share is the experiment record. It lives in its own Terraform state and its
own resource group precisely so tearing down compute cannot reach it, and every
long-lived artifact a node produces is *published* here rather than kept on the
node's ephemeral disk.

Two constraints shape everything in this module, and both are service
behaviour rather than SDK awkwardness:

* **Azure Files does not create parent directories.** Writing to a path whose
  parent is absent fails with ``ParentNotFound``. Every write therefore walks
  the path and creates each level first.
* **A killed task leaves partially-copied snapshot directories behind.** The
  manifest is the definition of what is complete, so a download must be driven
  by what ``CHECKPOINT.json`` names, never by what the directory happens to
  contain.
"""

from __future__ import annotations

import contextlib
import json
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from azure.storage.fileshare import ShareDirectoryClient, ShareServiceClient

from src.shared import records

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from datetime import datetime

    from src.interfaces.cloud.config import CloudConfig

ARCHIVE_DIR = "archive"
CODE_DIR = "code"
LOGS_DIR = "logs"
ABSTRACTION_DIR = "combo_abstraction"

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


@dataclass(frozen=True)
class ShareEntry:
    """One name under a share directory, and whether it is itself a directory."""

    name: str
    is_directory: bool
    size: int | None
    # Only when listed with ``etags=True``: the version key an incremental
    # sync compares, since a rewrite rarely changes a small record's size.
    etag: str | None = None


def share_client(config: CloudConfig) -> ShareServiceClient:
    """Build a share client from the store's account key.

    The account key rather than a token credential: Azure Files' OAuth path
    needs a data-plane RBAC role assignment that this subscription does not
    grant, and the key is already the credential the pool itself mounts with.
    """
    return ShareServiceClient(
        account_url=f"https://{config.storage_account}.file.core.windows.net",
        credential=config.share_key,
    )


def directory(service: ShareServiceClient, share: str, path: str) -> ShareDirectoryClient:
    """Client for one directory on the share."""
    return service.get_share_client(share).get_directory_client(path)


def list_entries(
    service: ShareServiceClient, share: str, path: str, *, etags: bool = False
) -> list[ShareEntry]:
    """List one directory, returning an empty list when it does not exist.

    Absent and empty are deliberately not distinguished: every caller here is
    answering "what is published?", and a share where nothing has been
    published yet is not an error state.

    ``etags`` asks the service for each file's etag in the same listing --
    measured at no extra cost over 2,252 entries (0.78s against 1.6s plain).
    """
    try:
        client = directory(service, share, path)
        listing = (
            client.list_directories_and_files(include=["Etag"])
            if etags
            else client.list_directories_and_files()
        )
        return [
            ShareEntry(
                name=str(item["name"]),
                is_directory=bool(item.get("is_directory")),
                size=item.get("size"),
                etag=item.get("etag"),
            )
            for item in listing
        ]
    except ResourceNotFoundError:
        return []


def read_text(service: ShareServiceClient, share: str, path: str) -> str | None:
    """Read one file from the share, or ``None`` when it is not there."""
    try:
        downloader = service.get_share_client(share).get_file_client(path).download_file()
    except ResourceNotFoundError:
        return None
    return bytes(downloader.readall()).decode("utf-8", errors="replace")


def write_text(service: ShareServiceClient, share: str, path: str, body: str) -> None:
    """Write one file to the share, creating its parent directory.

    Azure Files will not create parents implicitly -- a write beneath a missing
    directory fails with ``ParentNotFound`` rather than making it.
    """
    share_client = service.get_share_client(share)
    parent = path.rsplit("/", 1)[0] if "/" in path else ""
    if parent:
        with contextlib.suppress(ResourceExistsError):
            share_client.get_directory_client(parent).create_directory()
    share_client.get_file_client(path).upload_file(body.encode("utf-8"))


def delete_file(service: ShareServiceClient, share: str, path: str) -> bool:
    """Remove one file. ``True`` if it was there, ``False`` if it already was not.

    The share is the experiment record and this is the only function here that
    destroys any of it, so it is deliberately narrow: one file, named in full,
    never a directory and never a pattern. Anything wanting to remove many
    things does so one name at a time, having decided each one.

    An absent file is not an error. A delete that has already happened and a
    delete that never needed to are the same outcome, and a caller retrying
    after a partial sweep should not have to tell them apart.
    """
    try:
        service.get_share_client(share).get_file_client(path).delete_file()
    except ResourceNotFoundError:
        return False
    return True


def delete_directory(service: ShareServiceClient, share: str, path: str) -> bool:
    """Remove one EMPTY directory. ``False`` if it was absent or still occupied.

    Azure Files does not remove a directory when its files go, so deleting a
    snapshot's contents leaves the directory behind -- and an empty directory is
    not free: the parent listing still enumerates it, which is the cost a
    metadata walk pays per run.

    Emptiness is CHECKED rather than the service's refusal caught: the SDK's
    general error type is classified once in `errors.attempt` and a guard fails
    if anything under `interfaces/` names it again -- including, as it turns
    out, a docstring explaining that it does not. The listing costs one round
    trip against a delete that would have cost one anyway.

    This never decides that a subtree should go -- only tidies up after that
    decision was carried out one file at a time.
    """
    if list_entries(service, share, path):
        return False
    try:
        directory(service, share, path).delete_directory()
    except ResourceNotFoundError:
        return False
    return True


def walk_files(
    service: ShareServiceClient,
    share: str,
    path: str,
    *,
    skip_dir: Callable[[str], bool] | None = None,
) -> Iterator[tuple[str, str | None]]:
    """Yield ``(path, etag)`` for every file beneath ``path``, depth first.

    Used by the metadata sync, which needs to see the whole published tree in
    order to pick the small JSON out of it.

    ``skip_dir`` prunes a subtree without descending into it. A run's ``.zarr``
    snapshots hold thousands of chunk files each, and listing them is a round
    trip per directory -- filtering them out AFTER the walk still paid for the
    walk, which is where the time went: 167s to pull 146 small JSON files.

    The etag rides along because it costs nothing here and is the whole basis of
    an incremental refresh: a published record never changes once written, so an
    unchanged etag means the caller already has the bytes.
    """
    for entry in list_entries(service, share, path, etags=True):
        child = f"{path}/{entry.name}"
        if entry.is_directory:
            if skip_dir is not None and skip_dir(entry.name):
                continue
            yield from walk_files(service, share, child, skip_dir=skip_dir)
        else:
            yield child, entry.etag


def task_log_names(service: ShareServiceClient, share: str) -> list[str]:
    """Every published task log, oldest first.

    Published logs matter more than node-side ``stdout.txt``: Batch keeps task
    output on the node, and the pool scales to zero within minutes of a task
    ending, so the node copy is gone for exactly the failed tasks most worth
    reading.
    """
    return sorted(entry.name for entry in list_entries(service, share, LOGS_DIR))


def read_task_log(service: ShareServiceClient, share: str, task_id: str) -> str | None:
    """Read one published task log by task id."""
    return read_text(service, share, f"{LOGS_DIR}/{task_id}.log")


def download_file(service: ShareServiceClient, share: str, path: str, destination: Path) -> None:
    """Download one file, creating its local parent directories.

    Written to a sibling and renamed into place, so a reader sharing the tree
    never opens a half-downloaded file -- and a destination that is a hard link
    into an older tree is replaced, not written through.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    downloader = service.get_share_client(share).get_file_client(path).download_file()
    partial = destination.with_name(f".{destination.name}.part")
    with partial.open("wb") as handle:
        downloader.readinto(handle)
    partial.replace(destination)


def manifest_members(service: ShareServiceClient, share: str, run_path: str) -> set[str] | None:
    """Every snapshot directory the manifest names, or ``None`` if it is absent.

    THE MANIFEST IS THE DEFINITION OF COMPLETE. A killed task leaves
    partially-copied snapshot directories, and the publish writes the manifest LAST
    so it never names one. Listing the directory instead pulls an orphan down, which
    reads as ``mmap length is greater than file size`` and which a later incremental
    copy skips as already-present, making it permanent.

    The whole LADDER, not just the current snapshot: every retained rung is a
    legitimate evaluation target, which is what makes a within-run exploitability
    curve computable at all.

    The manifest NAME is imported rather than spelled here, because a hardcoded one
    fails OPEN -- the guard would silently degrade to "fetch everything".
    """
    raw = read_text(service, share, f"{run_path}/{records.STATIC_CHECKPOINT}")
    if raw is None:
        return None
    manifest = json.loads(raw)
    members = {str(manifest["zarr"])} if manifest.get("zarr") else set()
    for entry in manifest.get("retained", []):
        if entry.get("zarr"):
            members.add(str(entry["zarr"]))
    return members


def is_snapshot_dir(name: str) -> bool:
    """A directory holding checkpoint data, never worth descending into.

    Two kinds. ``*.zarr`` is the static backend's snapshot. ``keys-<iter>`` is
    the DYNAMIC backend's key table, and it is pure dead weight: that backend
    was deleted, its checkpoints are permanently unreadable at HEAD, and
    nothing in ``src/`` opens a ``vocab.json``. It was still being fetched
    because it is JSON and every metadata sync matches on the suffix --
    **37.17 MB of the 37.8 MB** a ``--source share`` read pulled, to answer a
    question that needed 0.06 MB of eval documents.

    Skipped, not deleted: leaving the bytes on the share costs nothing and
    removing them is not this function's decision.
    """
    return name.endswith(".zarr") or name.startswith("keys-")


def is_snapshot_path(relative: str) -> bool:
    """Whether a published path is checkpoint data rather than record.

    Keyed on a snapshot COMPONENT, NOT on nesting depth. Depth is wrong and
    was: ``<run>/evals/record-*.json`` is three components deep and would be
    classified as checkpoint data, which silently excluded the eval records
    from every fetch -- the exact files ``ledger --rebuild`` globs, and the
    reason the command exists.
    """
    return any(is_snapshot_dir(part) for part in Path(relative).parts)


def is_metadata(name: str) -> bool:
    """Whether a published file is small enough to be worth a default sync.

    Every analysis command -- ``ledger``, ``curve``,
    ``promote`` -- reads only JSON. The checkpoints beside them are ~540 MB of
    zarr chunks that nothing local opens.
    """
    return name.endswith((".json", ".jsonl"))


def ensure_directory(service: ShareServiceClient, share: str, path: str) -> None:
    """Create ``path`` and every parent above it.

    Azure Files does not create parents implicitly -- writing under an absent
    directory fails with ``ParentNotFound`` -- so each level is created in
    turn. Already-present levels are the normal case, not an error.
    """
    share_client = service.get_share_client(share)
    walked = ""
    for part in [segment for segment in path.split("/") if segment]:
        walked = f"{walked}/{part}" if walked else part
        try:
            share_client.get_directory_client(walked).create_directory()
        except ResourceExistsError:
            continue


def upload_file(service: ShareServiceClient, share: str, path: str, source: Path) -> None:
    """Upload one local file to ``path``, creating its parent directories."""
    parent = path.rsplit("/", 1)[0] if "/" in path else ""
    if parent:
        ensure_directory(service, share, parent)
    with source.open("rb") as handle:
        service.get_share_client(share).get_file_client(path).upload_file(handle)


def snapshot_name(now: datetime) -> str:
    """The id of one immutable code snapshot."""
    return f"code-{now:%Y%m%d_%H%M%S}"


def _snapshot_filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    """Drop excluded directories, and strip ownership from what remains.

    Ownership is cleared because the node extracts as an unprivileged task
    user: a tarball carrying the laptop's uid/gid is one more thing for tar to
    fail to restore. macOS xattrs and resource forks never enter the archive in
    the first place -- ``tarfile`` does not write them, which is what the shell
    version needed ``COPYFILE_DISABLE=1 --no-xattrs`` to achieve.
    """
    parts = Path(info.name).parts
    if any(part in SNAPSHOT_EXCLUDES for part in parts):
        return None
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    return info


def build_code_snapshot(root: Path, destination: Path) -> None:
    """Seal the working tree into one gzipped tarball.

    ONE TARBALL, not a directory tree: Azure Files would otherwise need every
    nested path pre-created and would cost a round trip per file. A sealed
    archive is also atomic in the way that matters -- a half-uploaded tarball
    is simply absent, rather than a partially-populated tree a node might run.
    """
    with tarfile.open(destination, "w:gz") as archive:
        for entry in sorted(root.iterdir()):
            archive.add(entry, arcname=entry.name, filter=_snapshot_filter)


def publish_code_snapshot(
    service: ShareServiceClient, share: str, root: Path, now: datetime
) -> str:
    """Build and upload an immutable snapshot of the tree; return its id.

    Pinned per submission on purpose: a push while a job is running must not
    change what that job is executing.
    """
    name = snapshot_name(now)
    with tempfile.TemporaryDirectory() as workspace:
        tarball = Path(workspace) / f"{name}.tar.gz"
        build_code_snapshot(root, tarball)
        upload_file(service, share, f"{CODE_DIR}/{name}.tar.gz", tarball)
    return name
