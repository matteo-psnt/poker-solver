"""A pruned rung must be refused in the terminal, not on a node.

Pins a MEASURED failure. Pruning removes a snapshot without rewriting the
manifest that advertises it, so `runinfo` offers rungs a node cannot find:
`control-fixed-300M` lists 57 ladder rungs and holds 3. Each such rung cost a
snapshot upload, a node allocation and a `uv sync` before dying on "the manifest
names static-N.zarr but it is not on the share" -- ~26 tasks in the 2026-08-23/24
window.

The check reads the CONTAINER, which is the store a node fetches from. Reading
the share instead is how this gate stopped working at all: once no rung landed
there its listing went empty, and `score --run` refused runs whose whole ladder
was present.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.interfaces.errors import CommandError

RUN = "run-train-production-to300M-control-fixed-300M-175417-9211"


@pytest.fixture
def container_holding(monkeypatch):
    """Point the container at a given set of objects for one run."""

    def _install(names):
        from src.interfaces.cloud.config import CloudConfig
        from src.interfaces.cloud.store import blob, workspace

        monkeypatch.setattr(
            CloudConfig,
            "load",
            classmethod(
                lambda cls: SimpleNamespace(storage_account="a", share_name="s", share_key="k")
            ),
        )
        # BY PREFIX: the gate asks about ONE run, and used to page through
        # every other run's rungs to find it.
        monkeypatch.setattr(
            blob, "rungs_for", lambda config, run_id: set(names) if run_id == RUN else set()
        )
        return workspace

    return _install


def _held(*iterations: int) -> list[str]:
    """The objects a published ladder is: one committed blob per rung, and no
    marker beside it -- presence IS the completeness a marker used to assert
    about a directory that could be half-copied."""
    return [f"static-{iteration}.ckpt.zst" for iteration in iterations]


def test_a_rung_the_container_holds_is_accepted(container_holding):
    workspace = container_holding(_held(100_000_000, 200_000_000, 300_000_000))
    workspace.verify_published_rungs(RUN, ["100000000", "300000000"])


def test_a_pruned_rung_is_refused_here_not_on_a_node(container_holding):
    workspace = container_holding(_held(100_000_000, 200_000_000, 300_000_000))
    with pytest.raises(CommandError, match="no published checkpoint"):
        workspace.verify_published_rungs(RUN, ["150000000"])


def test_the_refusal_names_what_the_container_actually_holds(container_holding):
    """A bare refusal sends the reader back to `runinfo`, which is the thing
    that lied -- so the message has to carry the store's own answer."""
    workspace = container_holding(_held(100_000_000, 300_000_000))
    with pytest.raises(CommandError, match="100000000, 300000000"):
        workspace.verify_published_rungs(RUN, ["5000000"])


def test_the_latest_checkpoint_is_not_checked(container_holding):
    """An empty rung means "whatever is current", which the node resolves and
    the ladder cannot name in advance -- checking it here would refuse every
    `score --run X` with no `--at`."""
    workspace = container_holding(_held(100_000_000))
    workspace.verify_published_rungs(RUN, [""])


def test_every_missing_rung_is_reported_at_once(container_holding):
    """One listing, one answer: reporting the first would make a 57-rung
    request a 57-attempt discovery."""
    workspace = container_holding(_held(100_000_000))
    with pytest.raises(CommandError, match="5000000, 10000000"):
        workspace.verify_published_rungs(RUN, ["5000000", "10000000", "100000000"])


def test_a_run_the_container_has_never_heard_of_is_refused(container_holding):
    """Not a crash on a missing key: an unpublished run is the same answer as a
    run holding no rungs, and the reader needs to be told which rungs exist."""
    workspace = container_holding(_held(100_000_000))
    with pytest.raises(CommandError, match="no published checkpoint"):
        workspace.verify_published_rungs("run-that-never-trained", ["100000000"])
