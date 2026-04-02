"""Every submission carries the submitting machine's git provenance.

A Batch node has no `.git` -- `share.SNAPSHOT_EXCLUDES` drops it from the code
tarball -- so `gitinfo`'s `git rev-parse` has nothing to answer from there. The
consequence went unnoticed for as long as training has been in the cloud:
`train_git_commit` and `eval_git_commit` were NULL on every cloud row. Measured
2026-08-04 on a probe leg, where `ledger` printed an empty commit column.

The stamp goes on in `stage_and_queue` and not in any caller, because that is
what the module exists for -- `submit`, `score` and `repair-ladder` differ only
in the legs they build, and a property every submission must have cannot be a
step three callers have to remember.
"""

from __future__ import annotations

import pytest

from src.interfaces.cloud import dispatch, spec
from src.shared import gitinfo


@pytest.fixture(autouse=True)
def _cold_gitinfo():
    """Both readers are `lru_cache`d and other modules call them during a suite
    run; without clearing, these assert against a warmed value."""
    gitinfo.get_git_commit.cache_clear()
    gitinfo.is_git_dirty.cache_clear()
    yield
    gitinfo.get_git_commit.cache_clear()
    gitinfo.is_git_dirty.cache_clear()


def _leg(**kwargs) -> spec.LegSpec:
    return spec.LegSpec(code_snapshot="snap", config="production", to=1000, **kwargs)


class TestStamping:
    def test_a_leg_gains_the_commit_it_was_submitted_from(self, monkeypatch):
        monkeypatch.setenv(gitinfo.COMMIT_ENV, "c" * 40)
        monkeypatch.setenv(gitinfo.DIRTY_ENV, "0")

        stamped = dispatch._stamped(_leg())

        assert stamped.git_commit == "c" * 40
        assert stamped.git_dirty == "0"

    def test_a_dirty_tree_is_recorded_as_dirty(self, monkeypatch):
        """The snapshot is built from the WORKING TREE, not from HEAD, so on a
        dirty tree the commit is only the nearest ancestor. Saying so is the
        difference between "probably this code" and "this code, verified
        clean"."""
        monkeypatch.setenv(gitinfo.COMMIT_ENV, "c" * 40)
        monkeypatch.setenv(gitinfo.DIRTY_ENV, "1")

        assert dispatch._stamped(_leg()).git_dirty == "1"

    def test_it_reaches_the_node_environment(self, monkeypatch):
        monkeypatch.setenv(gitinfo.COMMIT_ENV, "c" * 40)
        monkeypatch.setenv(gitinfo.DIRTY_ENV, "0")

        env = dispatch._stamped(_leg()).environment()

        assert env["RUN_GIT_COMMIT"] == "c" * 40
        assert env["RUN_GIT_DIRTY"] == "0"

    def test_an_unstampable_machine_still_submits(self, monkeypatch):
        """No git, no checkout -- the leg runs and records a null commit, which
        is exactly the status quo it is replacing. Refusing here would make a
        provenance nicety able to block work."""
        monkeypatch.delenv(gitinfo.COMMIT_ENV, raising=False)
        monkeypatch.delenv(gitinfo.DIRTY_ENV, raising=False)
        monkeypatch.setattr(gitinfo, "_run_git", lambda *a: None)

        stamped = dispatch._stamped(_leg())

        assert stamped.git_commit == ""
        assert stamped.git_dirty == ""
        stamped.validate()

    def test_nothing_else_about_the_leg_changes(self, monkeypatch):
        """`replace` on a frozen dataclass, so a typo in a field name raises
        rather than silently dropping an override."""
        monkeypatch.setenv(gitinfo.COMMIT_ENV, "c" * 40)
        original = _leg(sets=("solver__dcfr=1.5",), experiment="exp-7", arm="control")

        stamped = dispatch._stamped(original)

        assert stamped.sets == ("solver__dcfr=1.5",)
        assert stamped.experiment == "exp-7"
        assert stamped.arm == "control"
        assert stamped.to == original.to


class TestTheNodeEnd:
    """`src.shared.node.plan` reads what `spec` writes."""

    def test_the_node_parses_what_the_submitter_stamped(self, monkeypatch):
        from src.shared.node import plan as node_plan

        monkeypatch.setenv(gitinfo.COMMIT_ENV, "c" * 40)
        monkeypatch.setenv(gitinfo.DIRTY_ENV, "0")
        env = dispatch._stamped(_leg()).environment()

        parsed = node_plan.parse_environment(env)

        assert parsed.git_commit == "c" * 40
        assert parsed.git_dirty == "0"

    @pytest.mark.parametrize(
        ("dirty", "expected"),
        [("1", "DIRTY tree"), ("0", "clean tree"), ("", "tree state unknown")],
    )
    def test_the_wrapper_can_say_what_code_it_runs(self, dirty, expected):
        """The leg log is the only place the answer appears while the leg is
        still alive."""
        from src.shared.node import plan as node_plan

        leg = node_plan.LegPlan(
            op=node_plan.TRAIN, config="p", to=1, git_commit="c" * 40, git_dirty=dirty
        )
        assert expected in leg.provenance
        assert leg.provenance.startswith("cccccccccccc")

    def test_an_unstamped_leg_says_so_rather_than_claiming_a_commit(self):
        from src.shared.node import plan as node_plan

        leg = node_plan.LegPlan(op=node_plan.TRAIN, config="p", to=1)
        assert "unknown" in leg.provenance
