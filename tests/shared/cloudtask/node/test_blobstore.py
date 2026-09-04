"""One rung, one object -- and the node speaks Blob without the SDK.

The constraint that shaped this: `archive` is imported by the wrapper BEFORE
`uv sync`, so nothing it reaches may import the Azure SDK. A container SAS is
minted at dispatch, where the SDK exists, and the node does plain HTTPS.
"""

from __future__ import annotations

import email.message
import io
import urllib.error

import pytest

from src.shared.cloudtask.node import blobstore


class _Response:
    """`urlopen`'s answer: a context manager wrapping a readable body.

    A plain class rather than a `BytesIO` subclass -- overriding `__exit__` on
    an IO type fights the signature the stdlib declares for it.
    """

    def __init__(self, body: bytes = b"") -> None:
        self._body = io.BytesIO(body)

    def __enter__(self) -> io.BytesIO:
        return self._body

    def __exit__(self, *_exc: object) -> None:
        return None

    def read(self, *args: int) -> bytes:
        return self._body.read(*args)


SAS = "https://acct.blob.core.windows.net/checkpoints?sv=2021&sig=abc%3D"


class TestTheObjectItAddresses:
    def test_the_blob_name_goes_before_the_query(self):
        """The one place this is easy to get wrong. Appended after the query,
        every request would address the container with a junk parameter and
        authorise against a signature covering a different path."""
        url = blobstore.rung_uri(SAS, "run-a", "static-100.ckpt.zst")
        assert url == (
            "https://acct.blob.core.windows.net/checkpoints/"
            "run-a/static-100.ckpt.zst?sv=2021&sig=abc%3D"
        )

    def test_the_object_name_is_the_file_name(self):
        """One naming convention, not a stored name and a derived one that can
        drift: the object is called what the snapshot is called."""
        assert (
            blobstore.rung_uri(SAS, "r", "static-5.ckpt.zst")
            .partition("?")[0]
            .endswith("/r/static-5.ckpt.zst")
        )

    def test_a_sas_without_a_query_still_addresses(self):
        assert blobstore.rung_uri(
            "https://acct.blob.core.windows.net/checkpoints", "r", "s.ckpt.zst"
        ).endswith("/r/s.ckpt.zst")

    def test_the_name_is_escaped(self):
        """A run id is generated, but the record has held one with a bare
        number and one with characters nobody planned for."""
        assert "%20" in blobstore.rung_uri(SAS, "run a", "s.ckpt.zst")


class TestARungIsOneObject:
    def test_it_round_trips_as_one_file(self, tmp_path, monkeypatch):
        """The rung is the `.ckpt.zst` the trainer wrote, uploaded verbatim --
        no repacking, so what comes back is byte-for-byte what went up."""
        rung = tmp_path / "static-100.ckpt.zst"
        rung.write_bytes(b"\x28\xb5\x2f\xfd" + bytes(range(256)) * 40)
        sent: dict[str, bytes] = {}

        def _urlopen(request, timeout=None):
            if request.method == "PUT":
                sent["body"] = request.data.read()
                return _Response(b"")
            return _Response(sent["body"])

        monkeypatch.setattr(blobstore.urllib.request, "urlopen", _urlopen)
        size = blobstore.put_rung(SAS, "run-a", "static-100.ckpt.zst", rung)
        assert size == rung.stat().st_size

        back = tmp_path / "fetched"
        assert blobstore.get_rung(SAS, "run-a", "static-100.ckpt.zst", back) is True
        assert (back / "static-100.ckpt.zst").read_bytes() == rung.read_bytes()

    def test_the_upload_declares_its_length(self, tmp_path, monkeypatch):
        """Azure Blob refuses a chunked body on Put Blob, and urllib sends one
        for a file object unless Content-Length is set."""
        rung = tmp_path / "static-1.ckpt.zst"
        rung.write_bytes(b"x" * 5000)
        seen: dict[str, str] = {}

        def _urlopen(request, timeout=None):
            seen.update({k.lower(): v for k, v in request.header_items()})
            request.data.read()
            return _Response(b"")

        monkeypatch.setattr(blobstore.urllib.request, "urlopen", _urlopen)
        blobstore.put_rung(SAS, "run-a", "static-1.ckpt.zst", rung)
        assert seen["content-length"] == "5000"
        assert seen["x-ms-blob-type"] == "BlockBlob"


class TestAbsenceIsNotAnError:
    def _http_error(self, code):
        return urllib.error.HTTPError("u", code, "no", email.message.Message(), None)

    def test_a_missing_rung_reads_as_absent(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            blobstore.urllib.request,
            "urlopen",
            lambda *_a, **_k: (_ for _ in ()).throw(self._http_error(404)),
        )
        assert blobstore.exists(SAS, "run-a", "s.ckpt.zst") is False
        assert blobstore.get_rung(SAS, "run-a", "s.ckpt.zst", tmp_path) is False

    def test_any_other_failure_raises(self, monkeypatch, tmp_path):
        """A 403 is an expired SAS and a 500 is the service. Reading either as
        "no such rung" would let a resume start from zero over a live ladder --
        the failure the whole publish path is arranged to prevent."""
        for code in (403, 500):
            monkeypatch.setattr(
                blobstore.urllib.request,
                "urlopen",
                lambda *_a, code=code, **_k: (_ for _ in ()).throw(self._http_error(code)),
            )
            with pytest.raises(urllib.error.HTTPError):
                blobstore.exists(SAS, "run-a", "s.ckpt.zst")
            with pytest.raises(urllib.error.HTTPError):
                blobstore.get_rung(SAS, "run-a", "s.ckpt.zst", tmp_path)


def test_it_asks_for_an_api_version_that_allows_a_big_single_put():
    """Below 2019-12-12 a single PUT caps at 256 MiB and a ~1 GiB rung needs
    block-list assembly -- three round trips, and a failure mode where the
    blocks land and the list does not."""
    assert blobstore.API_VERSION >= "2019-12-12"
