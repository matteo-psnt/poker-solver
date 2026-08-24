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


def object_uri(container_sas: str, name: str) -> str:
    """Where one object lives, as a full SAS URL ready to request.

    The object name IS the file's name -- `run-x/static-100.ckpt.zst` for a
    rung, `buckets-....tar.zst` for an abstraction -- so there is one naming
    convention rather than a stored name and a derived one that can drift.

    The SAS is `https://<account>.blob.../<container>?<query>`; the blob name is
    spliced BEFORE the query, which is the one place this is easy to get wrong.
    """
    base, _, query = container_sas.partition("?")
    return f"{base.rstrip('/')}/{urllib.parse.quote(name)}" + (f"?{query}" if query else "")


def sibling_container(container_sas: str, container: str) -> str:
    """The same account SAS, addressing a different container.

    `container_sas` carries an ACCOUNT token and only the path names a
    container, so a task holds ONE credential and reaches both the rungs and
    the abstractions through it. Defined here rather than beside the minting
    because the node needs it and cannot import `interfaces`.
    """
    base, _, query = container_sas.partition("?")
    root = base.rstrip("/").rsplit("/", 1)[0]
    return f"{root}/{container}" + (f"?{query}" if query else "")


def _request(
    url: str, method: str, data: IO[bytes] | bytes | None = None
) -> urllib.request.Request:
    request = urllib.request.Request(url, method=method, data=data)
    request.add_header("x-ms-version", API_VERSION)
    return request


def list_container(container_sas: str, prefix: str = "") -> list[str]:
    """Every blob name in the container, following continuation markers.

    XML because that is what the REST API answers; the SDK that would hide it
    is exactly what the node cannot import. `<Name>` is the only element read,
    so a schema that grows around it does not matter.

    `prefix` narrows it SERVER-SIDE, which is the difference between asking
    about one run and paging through every rung of every run to find it.
    """
    import xml.etree.ElementTree as ET  # noqa: PLC0415 -- stdlib, only when listing

    base, _, query = container_sas.partition("?")
    names: list[str] = []
    marker = ""
    while True:
        url = f"{base.rstrip('/')}?restype=container&comp=list&{query}"
        if prefix:
            url += f"&prefix={urllib.parse.quote(prefix)}"
        if marker:
            url += f"&marker={urllib.parse.quote(marker)}"
        with urllib.request.urlopen(_request(url, "GET"), timeout=TIMEOUT_SECONDS) as response:
            root = ET.fromstring(response.read())
        names += [node.text or "" for node in root.iter("Name")]
        marker = (root.findtext("NextMarker") or "").strip()
        if not marker:
            return names


def exists(container_sas: str, name: str) -> bool:
    """Whether the rung is there. A HEAD, so it costs no bytes.

    This is what replaces `(source / name).is_dir()` on the mount, and it is a
    stronger question than the directory was: a directory can exist half-copied.
    """
    try:
        with urllib.request.urlopen(
            _request(object_uri(container_sas, name), "HEAD"),
            timeout=TIMEOUT_SECONDS,
        ):
            return True
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return False
        raise


def put_object(container_sas: str, name: str, source: Path) -> int:
    """Upload one rung's FILE as one object. Returns bytes uploaded.

    Streamed from the file rather than read into memory: a production rung is
    ~540 MB and the node is holding a trainer's tables at the same time.
    """
    size = source.stat().st_size
    with source.open("rb") as handle:
        request = _request(object_uri(container_sas, name), "PUT", handle)
        request.add_header("x-ms-blob-type", "BlockBlob")
        request.add_header("Content-Length", str(size))
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS):
            pass
    return size


def get_object(container_sas: str, name: str, destination: Path) -> bool:
    """Fetch one rung into `destination/<snapshot>`. False when absent.

    Streamed to disk for the same reason the upload is streamed from it.
    """
    import shutil  # noqa: PLC0415 -- stdlib, deferred to keep the wrapper's import light

    url = object_uri(container_sas, name)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / name.rsplit("/", 1)[-1]
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


def put_bytes(container_sas: str, name: str, body: bytes) -> int:
    """Upload a small object from memory. Returns bytes written.

    For a manifest, which is a few KB and already in hand. A rung goes through
    `put_object`, which streams from a file so a node holding a trainer's
    tables does not also hold 540 MB of snapshot.
    """
    request = _request(object_uri(container_sas, name), "PUT", data=body)
    request.add_header("x-ms-blob-type", "BlockBlob")
    request.add_header("Content-Length", str(len(body)))
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS):
        pass
    return len(body)


def read_object(container_sas: str, name: str) -> bytes | None:
    """One whole object's bytes, or None when it is not there.

    For the small things -- a manifest is a few KB. A rung goes through
    `get_object`, which streams to disk rather than into memory.
    """
    try:
        with urllib.request.urlopen(
            _request(object_uri(container_sas, name), "GET"), timeout=TIMEOUT_SECONDS
        ) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise


def read_head(container_sas: str, name: str, length: int) -> bytes | None:
    """The first `length` bytes of a rung, or None when it is not there.

    A HEAD proves an object exists; it cannot prove the object is a snapshot.
    An upload that died mid-stream still answers 200, and the difference only
    shows up when something tries to load it -- which for a migrated rung would
    be after the share copy was deleted. A few hundred bytes read the format's
    own header instead, so the check that gates a deletion actually opens what
    it is about to make the only copy.
    """
    request = _request(object_uri(container_sas, name), "GET")
    request.add_header("x-ms-range", f"bytes=0-{length - 1}")
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise
