"""Tests for git provenance helpers."""

import re

import pytest

from src.shared import gitinfo


def test_get_git_commit_is_sha_or_none():
    commit = gitinfo.get_git_commit()
    assert commit is None or re.fullmatch(r"[0-9a-f]{40}", commit)


def test_is_git_dirty_is_bool_or_none():
    assert gitinfo.is_git_dirty() in (True, False, None)


def test_run_git_returns_none_on_failure():
    # An invalid subcommand exits non-zero; helper must swallow it and return None.
    assert gitinfo._run_git("definitely-not-a-git-command") is None


class TestTheStampedProvenance:
    """A cloud node has no `.git`: the code snapshot excludes it.

    Until the submitter started stamping its own HEAD into the task environment,
    `git rev-parse` on a node had nothing to answer from -- so EVERY
    cloud-trained run and EVERY cloud-run evaluation recorded a null commit,
    for as long as training has been in the cloud. Measured on a probe task
    2026-08-04: `ledger` showed the commit column empty.
    """

    @pytest.fixture(autouse=True)
    def _cold(self):
        """Both readers are `lru_cache`d, and `records.py`/`metadata.py` call
        them during an ordinary suite run. Without clearing, these tests assert
        against whatever warmed the cache first -- passing alone and lying in
        the suite -- and then poison it for whatever runs next."""
        gitinfo.get_git_commit.cache_clear()
        gitinfo.is_git_dirty.cache_clear()
        yield
        gitinfo.get_git_commit.cache_clear()
        gitinfo.is_git_dirty.cache_clear()

    def test_a_stamped_commit_is_used(self, monkeypatch):
        monkeypatch.setenv(gitinfo.COMMIT_ENV, "a" * 40)
        assert gitinfo.get_git_commit() == "a" * 40

    def test_it_wins_over_the_local_checkout(self, monkeypatch):
        """Not a fallback. The tree on a node is an extracted tarball, so git's
        upward search can only describe some OTHER repository that happens to
        be an ancestor on that filesystem."""
        monkeypatch.setattr(gitinfo, "_run_git", lambda *a: "b" * 40)
        monkeypatch.setenv(gitinfo.COMMIT_ENV, "a" * 40)
        assert gitinfo.get_git_commit() == "a" * 40

    def test_an_empty_stamp_falls_back_to_git(self):
        """Every RUN_* key is emitted even when empty, so an unstamped task sets
        this to "" rather than leaving it absent."""
        with pytest.MonkeyPatch.context() as patch:
            patch.setenv(gitinfo.COMMIT_ENV, "")
            patch.setattr(gitinfo, "_run_git", lambda *a: "b" * 40)
            assert gitinfo.get_git_commit() == "b" * 40

    @pytest.mark.parametrize(("stamped", "expected"), [("1", True), ("0", False)])
    def test_dirty_is_three_state_not_two(self, monkeypatch, stamped, expected):
        """ "0" is a real answer. Collapsing it into "unknown" would throw away
        the distinction that makes a bare hash worth recording -- "probably this
        code" versus "this code, verified clean"."""
        monkeypatch.setenv(gitinfo.DIRTY_ENV, stamped)
        assert gitinfo.is_git_dirty() is expected

    def test_an_unstamped_dirty_flag_is_unknown_not_clean(self, monkeypatch):
        monkeypatch.setenv(gitinfo.DIRTY_ENV, "")
        monkeypatch.setattr(gitinfo, "_run_git", lambda *a: None)
        assert gitinfo.is_git_dirty() is None

    @pytest.mark.parametrize(("value", "encoded"), [(True, "1"), (False, "0"), (None, "")])
    def test_the_encoding_round_trips(self, monkeypatch, value, encoded):
        assert gitinfo.encode_dirty(value) == encoded
        monkeypatch.setenv(gitinfo.DIRTY_ENV, encoded)
        monkeypatch.setattr(gitinfo, "_run_git", lambda *a: None)
        assert gitinfo.is_git_dirty() is value

    def test_nothing_anywhere_reads_as_none(self, monkeypatch):
        monkeypatch.delenv(gitinfo.COMMIT_ENV, raising=False)
        monkeypatch.setattr(gitinfo, "_run_git", lambda *a: None)
        assert gitinfo.get_git_commit() is None
