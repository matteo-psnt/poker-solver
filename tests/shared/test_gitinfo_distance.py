"""Tests for git commit-distance used to age training runs.

These build a throwaway repository with a known shape rather than measuring
THIS one. The previous version asserted ``commits_ahead_of(HEAD~1) == 1``
against the live checkout, which silently assumed a linear history and broke
the moment a merge landed: for a merge commit ``HEAD~1`` is only the FIRST
parent, so ``HEAD~1..HEAD`` legitimately includes the whole second-parent
lineage. That is correct behaviour from ``commits_ahead_of`` and a wrong
expectation in the test — the kind of assertion that reports a defect where
none exists, on a day unrelated to the code it covers.
"""

import subprocess

import pytest

from src.shared import gitinfo
from src.shared.gitinfo import commits_ahead_of


def _git(repo, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def _init(repo):
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")


def _commit(repo, message: str) -> str:
    (repo / "file.txt").write_text(message)
    _git(repo, "add", "file.txt")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def linear_repo(tmp_path, monkeypatch):
    """Three commits in a straight line: c0 -> c1 -> c2 (HEAD)."""
    repo = tmp_path / "linear"
    _init(repo)
    shas = [_commit(repo, f"c{i}") for i in range(3)]
    monkeypatch.setattr(gitinfo, "_REPO_ROOT", repo)
    return shas


@pytest.fixture
def merged_repo(tmp_path, monkeypatch):
    """A merge: main gains one commit, a side branch two, then they join."""
    repo = tmp_path / "merged"
    _init(repo)
    base = _commit(repo, "base")

    _git(repo, "checkout", "-q", "-b", "side")
    (repo / "side.txt").write_text("a")
    _git(repo, "add", "side.txt")
    _git(repo, "commit", "-q", "-m", "side-1")
    (repo / "side.txt").write_text("b")
    _git(repo, "add", "side.txt")
    _git(repo, "commit", "-q", "-m", "side-2")

    _git(repo, "checkout", "-q", "main")
    main_only = _commit(repo, "main-1")
    _git(repo, "merge", "-q", "--no-ff", "side", "-m", "merge side")

    monkeypatch.setattr(gitinfo, "_REPO_ROOT", repo)
    return base, main_only


class TestLinearHistory:
    def test_head_is_zero_commits_ago(self, linear_repo):
        assert commits_ahead_of(linear_repo[-1]) == 0

    def test_parent_is_one_commit_ago(self, linear_repo):
        assert commits_ahead_of(linear_repo[-2]) == 1

    def test_distance_grows_walking_back(self, linear_repo):
        assert commits_ahead_of(linear_repo[0]) == 2


class TestMergedHistory:
    def test_first_parent_counts_the_whole_merged_lineage(self, merged_repo):
        """The documented behaviour, pinned so nobody "fixes" it back.

        From main's last pre-merge commit, HEAD is ahead by the two side commits
        plus the merge itself. Reading ``HEAD~1`` as "one commit ago" is exactly
        what made the old test wrong.
        """
        _, main_only = merged_repo
        assert commits_ahead_of(main_only) == 3

    def test_common_ancestor_counts_every_reachable_commit(self, merged_repo):
        base, _ = merged_repo
        assert commits_ahead_of(base) == 4


class TestUnknownCommits:
    def test_unknown_or_missing_commit_is_none(self, linear_repo):
        assert commits_ahead_of(None) is None
        assert commits_ahead_of("") is None
        # A well-formed but nonexistent sha resolves to None, not a bogus distance.
        assert commits_ahead_of("0" * 40) is None
