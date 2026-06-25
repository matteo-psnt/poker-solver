"""A pruned rung must be refused in the terminal, not on a node.

Pins a MEASURED failure. Pruning removes a snapshot without rewriting the
manifest that advertises it, so `runinfo` offers rungs `fetch_for_evaluation`
cannot find: `control-fixed-300M` lists 57 ladder rungs and holds 3. Each such
rung cost a snapshot upload, a node allocation and a `uv sync` before dying on
"the manifest names static-N.zarr but it is not on the share" -- ~26 tasks in
the 2026-08-23/24 window.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.interfaces.errors import CommandError

RUN = "run-train-production-to300M-control-fixed-300M-175417-9211"


@pytest.fixture
def share_holding(monkeypatch):
    """Point the share at a given set of entry names for one run directory."""

    def _install(names):
        from src.interfaces.cloud.config import CloudConfig
        from src.interfaces.cloud.store import share, workspace

        monkeypatch.setattr(
            CloudConfig,
            "load",
            classmethod(
                lambda cls: SimpleNamespace(storage_account="a", share_name="s", share_key="k")
            ),
        )
        monkeypatch.setattr(share, "share_client", lambda config: object())
        monkeypatch.setattr(
            share,
            "list_entries",
            lambda service, share_name, path: [
                SimpleNamespace(name=name, is_directory=not name.startswith(".")) for name in names
            ],
        )
        return workspace

    return _install


def _complete(*iterations: int) -> list[str]:
    names: list[str] = []
    for iteration in iterations:
        names.append(f"static-{iteration}.zarr")
        names.append(f".complete-static-{iteration}.zarr")
    return names


def test_a_rung_the_share_holds_is_accepted(share_holding):
    workspace = share_holding(_complete(100_000_000, 200_000_000, 300_000_000))
    workspace.verify_published_rungs(RUN, ["100000000", "300000000"])


def test_a_pruned_rung_is_refused_here_not_on_a_node(share_holding):
    workspace = share_holding(_complete(100_000_000, 200_000_000, 300_000_000))
    with pytest.raises(CommandError, match="no published, complete checkpoint"):
        workspace.verify_published_rungs(RUN, ["150000000"])


def test_the_refusal_names_what_the_share_actually_holds(share_holding):
    """A bare refusal sends the reader back to `runinfo`, which is the thing that
    lied -- so the message has to carry the share's own answer."""
    workspace = share_holding(_complete(100_000_000, 300_000_000))
    with pytest.raises(CommandError, match="100000000, 300000000"):
        workspace.verify_published_rungs(RUN, ["5000000"])


def test_a_snapshot_without_its_completion_marker_is_not_usable(share_holding):
    """An interrupted publish leaves the directory and no marker; the node's
    `require_complete` refuses it, so dispatch must refuse it too."""
    workspace = share_holding(["static-100000000.zarr"])
    with pytest.raises(CommandError, match="no published, complete checkpoint"):
        workspace.verify_published_rungs(RUN, ["100000000"])


def test_the_latest_checkpoint_is_not_checked(share_holding):
    """An empty rung means "whatever is current", which the node resolves and the
    ladder cannot name in advance -- checking it here would refuse every
    `score --run X` with no `--at`."""
    workspace = share_holding(_complete(100_000_000))
    workspace.verify_published_rungs(RUN, [""])


def test_every_missing_rung_is_reported_at_once(share_holding):
    """One round trip, one answer: reporting the first would make a 57-rung
    request a 57-attempt discovery."""
    workspace = share_holding(_complete(100_000_000))
    with pytest.raises(CommandError, match="5000000, 10000000"):
        workspace.verify_published_rungs(RUN, ["5000000", "10000000", "100000000"])
