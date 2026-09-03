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

    @pytest.mark.parametrize("op", [TaskName.EVALUATE, TaskName.PRECOMPUTE, TaskName.NET_PROBE])
    def test_everything_else_only_reads(self, op):
        assert op not in blob.WRITES_CHECKPOINTS

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

        url = blobstore.rung_uri(blob.container_sas(ACCOUNT, KEY, write=True), "run-a", "s.zarr")
        head, _, query = url.partition("?")
        assert head.endswith("/checkpoints/run-a/s.zarr.tar")
        assert "sig=" in query


def _query(url: str) -> dict[str, str]:
    import urllib.parse

    return dict(urllib.parse.parse_qsl(url.partition("?")[2]))


def _now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
