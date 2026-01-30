"""Tests for git provenance helpers."""

import re

from src.shared import gitinfo


def test_get_git_commit_is_sha_or_none():
    commit = gitinfo.get_git_commit()
    assert commit is None or re.fullmatch(r"[0-9a-f]{40}", commit)


def test_is_git_dirty_is_bool_or_none():
    assert gitinfo.is_git_dirty() in (True, False, None)


def test_run_git_returns_none_on_failure():
    # An invalid subcommand exits non-zero; helper must swallow it and return None.
    assert gitinfo._run_git("definitely-not-a-git-command") is None
