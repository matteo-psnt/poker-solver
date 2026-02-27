"""Tests for git commit-distance used to age training runs."""

import pytest

from src.shared.gitinfo import _run_git, commits_ahead_of, get_git_commit


def test_head_is_zero_commits_ago():
    head = get_git_commit()
    if head is None:
        pytest.skip("not a git checkout")
    assert commits_ahead_of(head) == 0


def test_parent_is_one_commit_ago():
    parent = _run_git("rev-parse", "HEAD~1")
    if not parent:
        pytest.skip("no parent commit")
    assert commits_ahead_of(parent) == 1


def test_unknown_or_missing_commit_is_none():
    assert commits_ahead_of(None) is None
    assert commits_ahead_of("") is None
    # A well-formed but nonexistent sha resolves to None, not a bogus distance.
    assert commits_ahead_of("0" * 40) is None
