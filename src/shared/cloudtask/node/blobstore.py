"""One rung, one object: Blob over plain HTTPS, on the stdlib alone.

`archive` is imported by the wrapper BEFORE `uv sync`, on the interpreter the
pool's start task installs -- so nothing here may import the Azure SDK, at
module level or lazily. The alternative was a second `--target` install into a
start task that now fails a node outright, to move bytes that need one PUT and
one GET.

So the credential is a CONTAINER SAS, minted at dispatch where the SDK does
exist and sealed into the task beside the record DSN. The node never signs
anything: a SAS URL already carries endpoint, scope and authorisation, and
`urllib` can speak the rest of the Blob REST API without help.

A RUNG IS ONE OBJECT. On the share it is ~4,200 zarr chunk files that take
minutes to copy; as a tar it is a single request, and existence becomes
completeness -- there is no window in which half a rung is readable, which is
the state `require_complete` exists to refuse.
"""

from __future__ import annotations

import tarfile
import urllib.error
import urllib.parse
import urllib.request
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path
    from typing import IO

# The version that allows a single-request PUT above 256 MiB. A rung is ~1 GiB,
# so without this every publish would need block-list assembly -- three round
# trips and a failure mode where the blocks land and the list does not.
API_VERSION = "2021-08-06"

# Generous: a rung is ~1 GiB and a cold node's uplink is not the pool's best.
# Shorter than the task guard, so a stall surfaces here rather than as a kill.
TIMEOUT_SECONDS = 900


def rung_uri(container_sas: str, run_id: str, snapshot: str) -> str:
    """The object a rung lives at, as a full SAS URL ready to request.

    The SAS is `https://<account>.blob.../<container>?<query>`; the blob name is
    spliced BEFORE the query, which is the one place this is easy to get wrong.
    """
    base, _, query = container_sas.partition("?")
    name = urllib.parse.quote(f"{run_id}/{snapshot}.tar")
    return f"{base.rstrip('/')}/{name}" + (f"?{query}" if query else "")


def _request(url: str, method: str, data: IO[bytes] | None = None) -> urllib.request.Request:
    request = urllib.request.Request(url, method=method, data=data)
    request.add_header("x-ms-version", API_VERSION)
    return request


def exists(container_sas: str, run_id: str, snapshot: str) -> bool:
    """Whether the rung is there. A HEAD, so it costs no bytes.

    This is what replaces `(source / name).is_dir()` on the mount, and it is a
    stronger question than the directory was: a directory can exist half-copied.
    """
    try:
        with urllib.request.urlopen(
            _request(rung_uri(container_sas, run_id, snapshot), "HEAD"),
            timeout=TIMEOUT_SECONDS,
        ):
            return True
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return False
        raise


def put_rung(container_sas: str, run_id: str, snapshot: str, source: Path) -> int:
    """Tar the rung's directory into ONE object. Returns bytes uploaded.

    Streamed through a temporary file rather than memory: a rung is ~1 GiB and
    the node holds the training arrays at the same time.
    """
    import tempfile  # noqa: PLC0415 -- stdlib, deferred only to keep import cost off the wrapper

    with tempfile.TemporaryDirectory(prefix="rung-") as tmp:
        from pathlib import Path as _Path  # noqa: PLC0415 -- see above

        bundle = _Path(tmp) / f"{snapshot}.tar"
        with tarfile.open(bundle, "w") as archive:
            # `arcname` is the snapshot itself, so unpacking reproduces the
            # directory the loader already expects and nothing downstream has to
            # know a tar was involved.
            archive.add(source, arcname=snapshot)
        size = bundle.stat().st_size
        with bundle.open("rb") as handle:
            request = _request(rung_uri(container_sas, run_id, snapshot), "PUT", handle)
            request.add_header("x-ms-blob-type", "BlockBlob")
            request.add_header("Content-Length", str(size))
            with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS):
                pass
    return size


def get_rung(container_sas: str, run_id: str, snapshot: str, destination: Path) -> bool:
    """Fetch one rung and unpack it under `destination`. False when absent."""
    import shutil  # noqa: PLC0415 -- stdlib, deferred to keep the wrapper's import light
    import tempfile  # noqa: PLC0415 -- see above

    url = rung_uri(container_sas, run_id, snapshot)
    try:
        with (
            urllib.request.urlopen(_request(url, "GET"), timeout=TIMEOUT_SECONDS) as response,
            tempfile.TemporaryDirectory(prefix="rung-") as tmp,
        ):
            from pathlib import Path as _Path  # noqa: PLC0415 -- see above

            bundle = _Path(tmp) / f"{snapshot}.tar"
            with bundle.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            destination.mkdir(parents=True, exist_ok=True)
            with tarfile.open(bundle) as archive:
                # `filter="data"` refuses absolute paths, `..` and device files.
                # The tar is ours, but a checkpoint loader is not the place to
                # find out that something else wrote one.
                archive.extractall(destination, filter="data")
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return False
        raise
    return True
