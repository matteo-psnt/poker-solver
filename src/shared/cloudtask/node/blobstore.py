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

A RUNG IS ONE OBJECT, and since the format changed it is one FILE -- the
`.ckpt.zst` the trainer writes, uploaded verbatim. The object name is the file
name, so nothing has to invent or parse a second naming convention.

Existence becomes completeness: there is no window in which half a rung is
readable, which is the state `require_complete` exists to refuse.
"""

from __future__ import annotations

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

    The object name IS the snapshot's file name -- `run-x/static-100.ckpt.zst`
    -- so there is one naming convention rather than a stored name and a
    derived one that can drift.

    The SAS is `https://<account>.blob.../<container>?<query>`; the blob name is
    spliced BEFORE the query, which is the one place this is easy to get wrong.
    """
    base, _, query = container_sas.partition("?")
    name = urllib.parse.quote(f"{run_id}/{snapshot}")
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
    """Upload one rung's FILE as one object. Returns bytes uploaded.

    Streamed from the file rather than read into memory: a production rung is
    ~540 MB and the node is holding a trainer's tables at the same time.
    """
    size = source.stat().st_size
    with source.open("rb") as handle:
        request = _request(rung_uri(container_sas, run_id, snapshot), "PUT", handle)
        request.add_header("x-ms-blob-type", "BlockBlob")
        request.add_header("Content-Length", str(size))
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS):
            pass
    return size


def get_rung(container_sas: str, run_id: str, snapshot: str, destination: Path) -> bool:
    """Fetch one rung into `destination/<snapshot>`. False when absent.

    Streamed to disk for the same reason the upload is streamed from it.
    """
    import shutil  # noqa: PLC0415 -- stdlib, deferred to keep the wrapper's import light

    url = rung_uri(container_sas, run_id, snapshot)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / snapshot
    try:
        with (
            urllib.request.urlopen(_request(url, "GET"), timeout=TIMEOUT_SECONDS) as response,
            target.open("wb") as handle,
        ):
            shutil.copyfileobj(response, handle)
    except urllib.error.HTTPError as error:
        target.unlink(missing_ok=True)
        if error.code == 404:
            return False
        raise
    return True


def read_head(container_sas: str, run_id: str, snapshot: str, length: int) -> bytes | None:
    """The first `length` bytes of a rung, or None when it is not there.

    A HEAD proves an object exists; it cannot prove the object is a snapshot.
    An upload that died mid-stream still answers 200, and the difference only
    shows up when something tries to load it -- which for a migrated rung would
    be after the share copy was deleted. A few hundred bytes read the format's
    own header instead, so the check that gates a deletion actually opens what
    it is about to make the only copy.
    """
    request = _request(rung_uri(container_sas, run_id, snapshot), "GET")
    request.add_header("x-ms-range", f"bytes=0-{length - 1}")
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise
