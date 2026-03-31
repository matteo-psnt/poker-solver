"""Answering a question about the record without keeping a copy of it.

Every reading command -- ``runinfo``, ``progress``, ``curve``, ``report``,
``compare``, ``ledger`` -- used to require a local ``data/runs`` populated by
``fetch``. The data was in the cloud; the questions were only answerable on the
machine that had last synced. Two boxes could hold different answers, and a
fresh checkout could hold none.

``--source share`` materialises the published JSON into a temporary directory,
answers the question there, and throws it away. Materialising rather than
reading in place is deliberate: the record is a few hundred KB of small JSON, so
pulling it costs one round trip per file, while reading it in place would make
every command an SMB client and every reader aware of two filesystems. The
readers stay ordinary local-path code; only where the path comes from changes.

Checkpoints are never materialised. They are ~540 MB of zarr chunks that no
reading command opens, and the rule that the manifest defines what is complete
stays in ``fetch``, which is the command that genuinely wants them.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from azure.storage.fileshare import ShareServiceClient

from src.interfaces.cloud import share
from src.interfaces.cloud.config import CloudConfig

BASELINE_NAME = "baseline.json"


def pull_metadata(
    service: ShareServiceClient,
    share_name: str,
    destination: Path,
    *,
    run: str | None = None,
) -> int:
    """Download every published JSON record into ``destination``. Returns the count."""
    published = [
        entry.name
        for entry in share.list_entries(service, share_name, share.ARCHIVE_DIR)
        if entry.is_directory
    ]
    if run is not None:
        if run not in published:
            raise SystemExit(
                f"'{run}' is not published. Published runs: {', '.join(published) or '(none)'}"
            )
        published = [run]

    pulled = 0
    for name in published:
        for remote in share.walk_files(service, share_name, f"{share.ARCHIVE_DIR}/{name}"):
            relative = remote[len(f"{share.ARCHIVE_DIR}/") :]
            if share.is_snapshot_path(relative) or not share.is_metadata(Path(relative).name):
                continue
            share.download_file(service, share_name, remote, destination / relative)
            pulled += 1
    return pulled


def read_baseline(service: ShareServiceClient, share_name: str) -> str | None:
    """The published baseline document, or None when none has been promoted."""
    return share.read_text(service, share_name, BASELINE_NAME)


def write_baseline(service: ShareServiceClient, share_name: str, body: str) -> None:
    """Publish the baseline pointer.

    Small, single-writer, and the conclusion of every experiment: which run the
    next one forks from. It was the only artifact that never left the machine
    that wrote it, so a reinstall lost it and two boxes could silently disagree.
    """
    share.write_text(service, share_name, BASELINE_NAME, body)


@contextmanager
def share_records(*, run: str | None = None) -> Iterator[Path]:
    """A local runs directory holding the published record, for the duration.

    Yields a path the ordinary local readers can use, then removes it. Nothing
    is left behind: this is a question being answered, not a sync.
    """
    config = CloudConfig.load()
    service = share.share_client(config)
    with tempfile.TemporaryDirectory(prefix="poker-share-") as tmp:
        root = Path(tmp)
        pull_metadata(service, config.share_name, root, run=run)
        baseline = read_baseline(service, config.share_name)
        if baseline is not None:
            (root / BASELINE_NAME).write_text(baseline)
        yield root
