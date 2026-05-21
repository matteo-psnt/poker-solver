"""Regenerable artifacts live outside the working tree.

`data/` was deleted for good on 2026-08-04. Keeping it deleted is not a
one-time act: the caches recreated it every time anything enumerated boards,
which is why it survived two prunes and came back after both.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.shared import cache, repo

WORKING_TREE = repo.ROOT


class TestResolution:
    def test_an_explicit_override_wins(self, monkeypatch):
        """The node wrapper sets this to /mnt/work/cache: a Batch task's HOME is
        its own working directory, wiped with the task, so the default would
        re-canonicalise the river's 2.6M boards (~1 min) on every task."""
        monkeypatch.setenv(cache.ENV_OVERRIDE, "/mnt/work/cache")
        assert cache.cache_root() == Path("/mnt/work/cache")

    def test_xdg_is_honoured(self, monkeypatch, tmp_path):
        monkeypatch.delenv(cache.ENV_OVERRIDE, raising=False)
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        assert cache.cache_root() == tmp_path / "poker-solver"

    def test_it_falls_back_to_the_home_cache(self, monkeypatch):
        monkeypatch.delenv(cache.ENV_OVERRIDE, raising=False)
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: Path("/home/someone")))
        assert cache.cache_root() == Path("/home/someone/.cache/poker-solver")

    def test_importing_creates_nothing(self, monkeypatch, tmp_path):
        """The behaviour that made `data/` reappear after each deletion: a
        module-level path that something eagerly mkdir'd."""
        monkeypatch.setenv(cache.ENV_OVERRIDE, str(tmp_path / "nope"))
        cache.cache_dir("boards")
        assert not (tmp_path / "nope").exists()


class TestNothingCachesIntoTheWorkingTree:
    """The rule, stated as a test rather than as a comment in a doc."""

    @pytest.mark.parametrize(
        "module_path",
        [
            "src.pipeline.abstraction.postflop.board_enumeration",
            "src.pipeline.abstraction.preflop.opponent_clusters",
        ],
    )
    def test_default_cache_dirs_are_outside_the_repo(self, module_path):
        import importlib

        module = importlib.import_module(module_path)
        resolved = Path(module.DEFAULT_CACHE_DIR).resolve()
        inside = resolved == WORKING_TREE or WORKING_TREE in resolved.parents
        assert not inside, (
            f"{module_path}.DEFAULT_CACHE_DIR is {resolved}, inside the working tree. "
            f"That is what kept recreating data/."
        )

    def test_no_module_still_names_a_data_cache(self):
        """A literal `data/cache/...` anywhere is the old shape coming back."""
        offenders = [
            path
            for path in (WORKING_TREE / "src").rglob("*.py")
            if 'Path("data/cache' in path.read_text()
        ]
        assert not offenders, f"still caching into the working tree: {offenders}"


class TestExpiringJsonCache:
    """The counterpart to the records substrate: throwaway, expiring, and safe
    to lose. It exists because an in-process memo does nothing for a CLI, which
    is a fresh process every time -- `poker-solver cost` run three times made
    three Cost Management queries and earned a 429.
    """

    def test_a_stored_value_comes_back(self, tmp_path, monkeypatch):
        monkeypatch.setenv(cache.ENV_OVERRIDE, str(tmp_path))
        cache.store_json("billing", "k", {"total": 316.71})
        assert cache.cached_json("billing", "k", ttl=60.0) == {"total": 316.71}

    def test_keys_do_not_collide(self, tmp_path, monkeypatch):
        monkeypatch.setenv(cache.ENV_OVERRIDE, str(tmp_path))
        cache.store_json("billing", "a", {"v": 1})
        cache.store_json("billing", "b", {"v": 2})
        assert cache.cached_json("billing", "a", ttl=60.0) == {"v": 1}
        assert cache.cached_json("billing", "b", ttl=60.0) == {"v": 2}

    def test_an_expired_value_is_a_miss(self, tmp_path, monkeypatch):
        monkeypatch.setenv(cache.ENV_OVERRIDE, str(tmp_path))
        cache.store_json("billing", "k", {"v": 1})
        assert cache.cached_json("billing", "k", ttl=-1.0) is None

    def test_an_absent_value_is_a_miss_not_an_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv(cache.ENV_OVERRIDE, str(tmp_path))
        assert cache.cached_json("billing", "never-stored", ttl=60.0) is None

    def test_a_corrupt_file_is_a_miss_not_a_crash(self, tmp_path, monkeypatch):
        """A cache that can break the thing it accelerates is worse than none."""
        monkeypatch.setenv(cache.ENV_OVERRIDE, str(tmp_path))
        cache.store_json("billing", "k", {"v": 1})
        for path in (tmp_path / "billing").glob("*.json"):
            path.write_text("{ not json")
        assert cache.cached_json("billing", "k", ttl=60.0) is None

    def test_an_unwritable_root_does_not_raise(self, monkeypatch):
        """Best-effort in both directions: the caller already has its answer."""
        monkeypatch.setenv(cache.ENV_OVERRIDE, "/nonexistent-root/x")
        cache.store_json("billing", "k", {"v": 1})

    def test_no_scratch_files_are_left_behind(self, tmp_path, monkeypatch):
        """Written beside and renamed, so a concurrent reader never sees half a
        file. The rename must also not leave the scratch copy on disk."""
        monkeypatch.setenv(cache.ENV_OVERRIDE, str(tmp_path))
        cache.store_json("billing", "k", {"v": 1})
        assert not list((tmp_path / "billing").glob("*.tmp"))
        assert len(list((tmp_path / "billing").glob("*.json"))) == 1

    def test_importing_does_not_create_a_directory(self, tmp_path, monkeypatch):
        """The behaviour that made `data/` reappear after each deletion."""
        monkeypatch.setenv(cache.ENV_OVERRIDE, str(tmp_path / "unborn"))
        cache.cached_json("billing", "k", ttl=60.0)
        assert not (tmp_path / "unborn").exists()
