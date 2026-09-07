"""Who may overwrite a checkpoint, and for how long.

The SAS is minted at dispatch because the node cannot: `archive` is imported
before `uv sync`, so it speaks REST and needs a URL that carries its own
authorisation.
"""

from __future__ import annotations

import pytest

from src.interfaces.cloud.store import blob
from src.shared.cloudtask.kinds import TaskName

ACCOUNT, KEY = "acct", "a2V5" * 8


class TestOnlyATrainerMayWrite:
    """A score, an evaluate and a duel all FETCH a rung. Handing those a
    writable credential puts the power to overwrite a checkpoint in every task
    that merely reads one -- the difference between losing a score and losing
    the run it was scoring.
    """

    def test_training_writes(self):
        assert TaskName.TRAIN in blob.WRITES_CHECKPOINTS
        assert TaskName.TRAIN_PCS in blob.WRITES_CHECKPOINTS

    @pytest.mark.parametrize("op", [TaskName.EVALUATE, TaskName.PRECOMPUTE])
    def test_everything_else_only_reads(self, op):
        assert op not in blob.WRITES_CHECKPOINTS

    def test_every_kind_is_classified(self):
        """A kind that is neither named a writer nor deliberately a reader is a
        kind whose credential nobody decided. It silently gets read-only, which
        is safe for a fetch and a 403 for anything that publishes."""
        from src.shared.cloudtask import kinds

        readers = {TaskName.EVALUATE, TaskName.PRECOMPUTE}
        unclassified = set(kinds.KINDS) - {str(op) for op in blob.WRITES_CHECKPOINTS | readers}
        assert not unclassified, f"no checkpoint-access decision for: {sorted(unclassified)}"

    def test_a_reader_sas_carries_no_write_permission(self):
        token = blob.container_sas(ACCOUNT, KEY, write=False)
        # `sp=` is the permission set. `r`/`l` are read and list; `w`/`c` would
        # be write and create.
        permissions = _query(token)["sp"]
        assert "r" in permissions
        assert "w" not in permissions
        assert "c" not in permissions

    def test_a_writer_sas_can_create(self):
        permissions = _query(blob.container_sas(ACCOUNT, KEY, write=True))["sp"]
        assert "w" in permissions
        assert "c" in permissions

    @pytest.mark.parametrize("write", [True, False])
    def test_no_task_can_delete_a_checkpoint(self, write):
        """Not even a trainer. Overwriting its own rung is republishing, which
        is ordinary; removing one is `prune-checkpoints`' job and belongs to an
        operator reading a dry run, not to a task that crashed oddly.

        Verified live: cleaning up the round-trip probe needed a credential
        minted for the purpose, because neither task SAS could do it.
        """
        assert "d" not in _query(blob.container_sas(ACCOUNT, KEY, write=write))["sp"]

    def test_both_may_list(self):
        """Fetching "the current rung" resolves a name before it reads bytes."""
        for write in (True, False):
            assert "l" in _query(blob.container_sas(ACCOUNT, KEY, write=write))["sp"]


class TestTheWindow:
    def test_it_starts_in_the_past(self):
        """The node's clock is not this machine's. A window opening `now` is
        refused by a node running seconds behind, and the 403 that comes back
        looks exactly like a bad signature."""
        assert blob.CLOCK_SKEW.total_seconds() > 0
        assert _query(blob.container_sas(ACCOUNT, KEY, write=True))["st"] < _now()

    def test_it_outlives_a_job_not_a_task(self):
        """A ladder score fans out 30 rungs behind one dispatch and the last can
        start hours after the first. A SAS that expired between them would fail
        a task that did nothing wrong, and the retry would fail identically."""
        assert blob.SAS_LIFETIME.days >= 7


class TestTheUrlItBuilds:
    def test_it_addresses_the_container_and_carries_the_token(self):
        url = blob.container_sas(ACCOUNT, KEY, write=False)
        assert url.startswith(f"https://acct.blob.core.windows.net/{blob.CONTAINER}?")

    def test_it_is_what_blobstore_expects(self):
        """The two halves are written apart and must agree: `rung_uri` splices
        the blob name in BEFORE the query."""
        from src.shared.cloudtask.node import blobstore

        url = blobstore.rung_uri(
            blob.container_sas(ACCOUNT, KEY, write=True), "run-a", "s.ckpt.zst"
        )
        head, _, query = url.partition("?")
        assert head.endswith("/checkpoints/run-a/s.ckpt.zst")
        assert "sig=" in query


class TestTheCodeSnapshotUrl:
    """The task command line holds this in its environment and `curl`s it.
    One blob, read only: the line fetches a tarball and never lists, writes or
    reads a sibling, so the credential says exactly that."""

    def test_it_is_a_read_only_sas_on_the_single_blob(self):
        query = _query(blob.code_snapshot_sas(ACCOUNT, KEY, "code", "code-20260904_120000"))
        assert query["sp"] == "r"
        # `sr=b`: a SERVICE SAS on one blob, not an account SAS.
        assert query["sr"] == "b"
        assert query["st"] < _now()

    def test_it_addresses_the_tarball_the_dispatch_uploaded(self):
        url = blob.code_snapshot_sas(ACCOUNT, KEY, "code", "code-20260904_120000")
        head, _, query = url.partition("?")
        assert head == "https://acct.blob.core.windows.net/code/code-20260904_120000.tar.gz"
        assert "sig=" in query


class TestTheSealedTree:
    def test_the_tarball_carries_no_git_and_no_owner(self, tmp_path):
        """`.git` is dropped so the node has no history to misreport, and
        ownership is cleared because the node extracts as an unprivileged
        user: a laptop uid in the archive is one more thing for tar to fail
        to restore."""
        import tarfile

        root = tmp_path / "tree"
        (root / ".git").mkdir(parents=True)
        (root / ".git" / "HEAD").write_text("ref")
        (root / "src").mkdir()
        (root / "src" / "a.py").write_text("x = 1\n")
        tarball = tmp_path / "snap.tar.gz"

        blob.build_code_snapshot(root, tarball)

        with tarfile.open(tarball) as archive:
            names = archive.getnames()
            assert "src/a.py" in names
            assert not any(name.startswith(".git") for name in names)
            assert all(m.uid == 0 and m.gid == 0 and m.uname == "" for m in archive.getmembers())


def _query(url: str) -> dict[str, str]:
    import urllib.parse

    return dict(urllib.parse.parse_qsl(url.partition("?")[2]))


def _now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
