"""Regenerable artifacts live outside the working tree.

`data/` was deleted for good on 2026-08-04. Keeping it deleted is not a
one-time act: the caches recreated it every time anything enumerated boards,
which is why it survived two prunes and came back after both.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.shared import cache

WORKING_TREE = Path(__file__).resolve().parents[2]


class TestResolution:
    def test_an_explicit_override_wins(self, monkeypatch):
        """The node wrapper sets this to /mnt/work/cache: a Batch task's HOME is
        its own working directory, wiped with the task, so the default would
        re-canonicalise the river's 2.6M boards (~1 min) on every leg."""
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
