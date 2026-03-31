"""Reading a dead leg's log must not be the moment the tooling breaks.

`--source node` reads files that live on the node, and the pool scales to zero
within minutes of a task ending -- so for exactly the failed legs most worth
reading, the node is already gone and Batch answers "The specified node does not
exist." That surfaced as a forty-line ``azure.core`` traceback, at the moment
someone was trying to find out why a leg died, and it buried the fact that the
answer is one flag away on the share.
"""

from __future__ import annotations

import argparse

import pytest
from azure.core.exceptions import ResourceNotFoundError

from src.interfaces.cli.commands import logs
from src.interfaces.errors import CommandError


def _args(**overrides) -> argparse.Namespace:
    return logs.COMMAND.arguments(
        **{"task": "prod-220218-7098", "source": "node", "job": "poker-1", **overrides}
    )


@pytest.fixture
def dead_node(monkeypatch):
    monkeypatch.setattr(logs.CloudConfig, "load", staticmethod(lambda: object()))
    monkeypatch.setattr(logs.batch, "client", lambda _config: object())

    def _gone(*_args, **_kwargs):
        raise ResourceNotFoundError(
            message="Operation returned an invalid status 'The specified node does not exist.'"
        )

    monkeypatch.setattr(logs.batch, "task_file", _gone)


class TestTheNodeIsGone:
    def test_it_is_a_readable_refusal_not_a_traceback(self, dead_node):
        with pytest.raises(CommandError) as caught:
            logs.run(_args())
        assert "does not exist" in str(caught.value)

    def test_it_names_the_command_that_does_work(self, dead_node):
        """The share copy exists precisely for this case -- the publish-on-exit
        trap in run_leg.sh writes it because the node cannot be relied on."""
        with pytest.raises(CommandError) as caught:
            logs.run(_args())
        assert "logs --task prod-220218-7098" in str(caught.value)


def test_source_node_without_a_job_is_refused_before_any_call():
    """Node-side files are addressed by (job, task); this cannot be guessed."""
    with pytest.raises(CommandError, match="needs --job"):
        logs.run(_args(job=None))
