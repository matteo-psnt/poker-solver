"""An arm IS a worktree here, so the branch is already the arm's name.

Experiments are developed in parallel git worktrees — one idea per branch — and
retyping that name as `--arm` is a step that adds nothing and can be forgotten.
Forgetting it is not a small mistake: an untagged run is unaffiliated,
`ledger --experiment` never sees it, and the omission shows up as a missing row
rather than as an error.
"""

from __future__ import annotations

import argparse

import pytest

from src.interfaces.commands import submit
from src.shared import gitinfo


@pytest.fixture(autouse=True)
def _cold_gitinfo():
    gitinfo.get_git_branch.cache_clear()
    yield
    gitinfo.get_git_branch.cache_clear()


def _args(**over) -> argparse.Namespace:
    return argparse.Namespace(**{"arm": "", "experiment": "", **over})


class TestArmDefaultsToTheBranch:
    def test_an_experiment_arm_takes_the_branch(self, monkeypatch):
        monkeypatch.setenv(gitinfo.BRANCH_ENV, "worktree-hybrid-kernels")

        assert submit._arm(_args(experiment="exp-7")) == "worktree-hybrid-kernels"

    def test_an_explicit_arm_always_wins(self, monkeypatch):
        """The default exists to save typing, never to override it."""
        monkeypatch.setenv(gitinfo.BRANCH_ENV, "worktree-hybrid-kernels")

        assert submit._arm(_args(experiment="exp-7", arm="control")) == "control"

    def test_an_ordinary_run_is_left_untagged(self, monkeypatch):
        """Outside an experiment an arm label means nothing.

        Stamping every run with its branch would turn a field that says "this is
        part of a comparison" into one that is always set, and `ledger` groups on
        exactly that field.
        """
        monkeypatch.setenv(gitinfo.BRANCH_ENV, "worktree-hybrid-kernels")

        assert submit._arm(_args()) == ""

    def test_a_detached_checkout_changes_nothing(self, monkeypatch):
        """No branch to borrow, so the run is submitted exactly as it was before."""
        monkeypatch.delenv(gitinfo.BRANCH_ENV, raising=False)
        monkeypatch.setattr(gitinfo, "_run_git", lambda *args: "HEAD")

        assert submit._arm(_args(experiment="exp-7")) == ""
